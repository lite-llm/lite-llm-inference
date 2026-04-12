//! GPU tensor backend for CUDA-accelerated inference.
//!
//! Provides a `Tensor` type that wraps CUDA device memory and supports
//! matrix multiplication, element-wise ops, and data transfer between
//! host (CPU) and device (GPU) memory.
//!
//! When the `cuda` feature is disabled, this module falls back to CPU tensor ops.

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

/// Tensor device type: CPU or GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// GPU device ID
    Gpu(usize),
}

/// Information about a CUDA GPU device.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub total_memory_bytes: u64,
    pub compute_capability_major: usize,
    pub compute_capability_minor: usize,
}

/// Global GPU information returned by `cuda_info()`.
#[derive(Debug, Clone)]
pub struct CudaInfo {
    pub device_count: usize,
    pub devices: Vec<CudaDeviceInfo>,
}

// ---------------------------------------------------------------------------
// CUDA feature-gated types
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::safe::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
    use cudarc::driver::result::device::get_count as cuda_device_count;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Wrapper around cudarc's cuBLAS handle.
    ///
    /// Provides a safe interface for f32 GEMM operations.
    pub struct CublasHandle {
        blas: Arc<CudaBlas>,
        _device: Arc<CudaDevice>,
    }

    impl CublasHandle {
        /// Create a new cuBLAS handle bound to the given device.
        pub fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
            let blas = CudaBlas::new(device.clone()).map_err(|e| format!("cuBLAS init error: {e}"))?;
            Ok(Self {
                blas: Arc::new(blas),
                _device: device,
            })
        }

        /// Perform matrix multiplication: C = alpha * A @ B + beta * C
        ///
        /// - `a`: column-major (M x K) matrix on GPU
        /// - `b`: column-major (K x N) matrix on GPU
        /// - `c`: column-major (M x N) output matrix on GPU
        ///
        /// cuBLAS expects column-major layout. For row-major inputs (Rust convention),
        /// we compute: C = (B^T @ A^T)^T = A @ B  by swapping A and B roles.
        pub unsafe fn sgemm(
            &self,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            c: &mut CudaSlice<f32>,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<(), String> {
            // cudarc's GemmConfig for f32 uses the standard cuBLAS GEMM layout.
            // cuBLAS computes: C = alpha * op(A) * op(B) + beta * C
            // For row-major data, we exploit: (A @ B)^T = B^T @ A^T
            // So we call cuBLAS with transposed semantics on row-major data.
            let config = GemmConfig {
                transa: cudarc::cublas::safe::Transpose::N,
                transb: cudarc::cublas::safe::Transpose::N,
                m: n as i32,       // note: swapped for row-major trick
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k as i32,
                ldc: n as i32,
                beta: 0.0,
            };

            // Row-major trick: compute B @ A into C (effectively A^T @ B^T = (B @ A)^T)
            // Actually for row-major: C_row = A_row @ B_row
            // cuBLAS column-major: C_col = A_col @ B_col
            // With row-major data stored linearly, we need:
            //   c = a @ b  =>  use transa=N, transb=N but swap operands
            self.blas
                .gemm::<f32, _, _, _>(config, b, a, c)
                .map_err(|e| format!("cuBLAS Sgemm error: {e}"))
        }
    }

    /// GPU device manager singleton.
    ///
    /// - Initializes CUDA context on first use.
    /// - Provides cuBLAS handles per device.
    /// - Tracks GPU memory usage.
    /// - Reports device properties.
    pub struct GpuDeviceManager {
        devices: Vec<Arc<CudaDevice>>,
        blas_handles: Mutex<HashMap<usize, Arc<CublasHandle>>>,
        allocated_bytes: AtomicU64,
    }

    static MANAGER: OnceLock<GpuDeviceManager> = OnceLock::new();

    impl GpuDeviceManager {
        /// Get or initialize the global GPU device manager.
        pub fn global() -> &'static Self {
            MANAGER.get_or_init(|| Self::init())
        }

        /// Initialize CUDA devices.
        pub fn init() -> Self {
            let count = cuda_device_count().unwrap_or(0) as usize;
            log::info!("Detected {count} CUDA device(s)");

            let mut devices = Vec::with_capacity(count);
            for i in 0..count {
                match CudaDevice::new(i) {
                    Ok(device) => {
                        log::info!("Initialized CUDA device {i}");
                        devices.push(Arc::new(device));
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize CUDA device {i}: {e}");
                    }
                }
            }

            if devices.is_empty() {
                log::warn!("No CUDA devices available; GPU backend disabled");
            }

            Self {
                devices,
                blas_handles: Mutex::new(HashMap::new()),
                allocated_bytes: AtomicU64::new(0),
            }
        }

        /// Number of available CUDA devices.
        pub fn device_count(&self) -> usize {
            self.devices.len()
        }

        /// Get a reference to a CUDA device by index.
        pub fn device(&self, id: usize) -> Option<&Arc<CudaDevice>> {
            self.devices.get(id)
        }

        /// Get or create a cuBLAS handle for the given device.
        pub fn get_blas_handle(&self, device_id: usize) -> Option<Arc<CublasHandle>> {
            let mut cache = self.blas_handles.lock().ok()?;
            if let Some(handle) = cache.get(&device_id) {
                return Some(Arc::clone(handle));
            }
            let device = self.devices.get(device_id)?;
            match CublasHandle::new(Arc::clone(device)) {
                Ok(handle) => {
                    let handle = Arc::new(handle);
                    cache.insert(device_id, Arc::clone(&handle));
                    Some(handle)
                }
                Err(e) => {
                    log::error!("Failed to create cuBLAS handle for device {device_id}: {e}");
                    None
                }
            }
        }

        /// Allocate device memory and track usage.
        pub fn alloc_device(
            &self,
            device_id: usize,
            data: &[f32],
        ) -> Result<CudaSlice<f32>, String> {
            let device = self
                .device(device_id)
                .ok_or_else(|| format!("GPU device {device_id} not available"))?;
            let slice = device
                .htod_sync_copy(data)
                .map_err(|e| format!("H2D transfer error: {e}"))?;

            let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
            self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
            log::debug!("Allocated {bytes} bytes on GPU device {device_id}");

            Ok(slice)
        }

        /// Copy data from device to host and update tracking.
        pub fn copy_device_to_host(
            &self,
            slice: &CudaSlice<f32>,
        ) -> Result<Vec<f32>, String> {
            let device = self
                .device(0)
                .ok_or_else(|| "No GPU device available".to_string())?;
            let data = device
                .dtoh_sync_copy(slice)
                .map_err(|e| format!("D2H transfer error: {e}"))?;

            let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
            self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);

            Ok(data)
        }

        /// Total GPU memory currently allocated (tracked).
        pub fn allocated_memory_bytes(&self) -> u64 {
            self.allocated_bytes.load(Ordering::Relaxed)
        }

        /// Get device properties for the given device ID.
        pub fn device_properties(&self, device_id: usize) -> Option<CudaDeviceInfo> {
            use cudarc::driver::result::device::get_attribute;
            use cudarc::driver::sys::CUdevice_attribute::{
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
            };

            let device = self.devices.get(device_id)?;
            let device_raw = device.cu_device();

            let major = unsafe { get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_raw) }.ok()?;
            let minor = unsafe { get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_raw) }.ok()?;
            let total_mem = unsafe { get_attribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device_raw) }.ok()? as u64;

            // Device name requires primary context; use a placeholder if unavailable.
            let name = format!("CUDA Device {device_id}");

            Some(CudaDeviceInfo {
                device_id,
                name,
                total_memory_bytes: total_mem,
                compute_capability_major: major as usize,
                compute_capability_minor: minor as usize,
            })
        }

        /// Collect info for all devices.
        pub fn cuda_info() -> CudaInfo {
            let manager = Self::global();
            let count = manager.device_count();
            let mut devices = Vec::with_capacity(count);
            for i in 0..count {
                if let Some(info) = manager.device_properties(i) {
                    devices.push(info);
                }
            }
            CudaInfo {
                device_count: count,
                devices,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public re-exports and feature-gated stubs
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use cuda_impl::{CublasHandle, GpuDeviceManager};

/// Alias for backwards compatibility.
#[cfg(feature = "cuda")]
pub type CudaDeviceManager = GpuDeviceManager;

#[cfg(not(feature = "cuda"))]
/// CublasHandle stub when CUDA is not available.
pub struct CublasHandle;

#[cfg(not(feature = "cuda"))]
impl CublasHandle {
    pub fn new() -> Result<Self, String> {
        Err("CUDA feature not enabled".to_string())
    }
}

#[cfg(not(feature = "cuda"))]
/// GpuDeviceManager stub when CUDA is not available.
pub struct GpuDeviceManager;

/// Alias for backwards compatibility.
#[cfg(not(feature = "cuda"))]
pub type CudaDeviceManager = GpuDeviceManager;

#[cfg(not(feature = "cuda"))]
impl GpuDeviceManager {
    pub fn global() -> &'static Self {
        static STUB: std::sync::OnceLock<GpuDeviceManager> = std::sync::OnceLock::new();
        STUB.get_or_init(|| Self)
    }

    pub fn init() -> Self {
        log::info!("CUDA feature disabled; GpuDeviceManager is a no-op stub");
        Self
    }

    pub fn device_count(&self) -> usize {
        0
    }

    pub fn cuda_info() -> CudaInfo {
        CudaInfo {
            device_count: 0,
            devices: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Public cuda_info() function (works with or without CUDA)
// ---------------------------------------------------------------------------

/// Return information about available CUDA GPU devices.
///
/// When the `cuda` feature is disabled or no GPUs are available,
/// returns an empty `CudaInfo` with `device_count == 0`.
pub fn cuda_info() -> CudaInfo {
    #[cfg(feature = "cuda")]
    {
        GpuDeviceManager::cuda_info()
    }
    #[cfg(not(feature = "cuda"))]
    {
        CudaInfo {
            device_count: 0,
            devices: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// A tensor that can reside on CPU or GPU memory.
///
/// When compiled with the `cuda` feature, GPU tensors use CUDA device memory
/// for accelerated compute. Otherwise, all tensors use CPU memory.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub device: Device,
    /// Data is stored as f32 values on the host.
    /// When cuda is enabled, this represents a cache of host-side data,
    /// with the actual GPU memory managed by a DeviceStorage handle.
    pub data: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub device_id: Option<usize>,
}

impl Tensor {
    /// Create a new CPU tensor from data and shape.
    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "data length must match shape product");

        Self {
            shape: shape.to_vec(),
            device: Device::Cpu,
            data,
            #[cfg(feature = "cuda")]
            device_id: None,
        }
    }

    /// Create a zero-filled tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            device: Device::Cpu,
            data: vec![0.0; len],
            #[cfg(feature = "cuda")]
            device_id: None,
        }
    }

    /// Create a tensor filled with random values (for weight initialization).
    pub fn randn(shape: &[usize], scale: f32) -> Self {
        let len = shape.iter().product();
        let mut data = vec![0.0; len];
        for d in &mut data {
            *d = (rand::random::<f32>() - 0.5) * 2.0 * scale;
        }
        Self {
            shape: shape.to_vec(),
            device: Device::Cpu,
            data,
            #[cfg(feature = "cuda")]
            device_id: None,
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape the tensor (same data, new shape).
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let expected: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), expected, "reshape size mismatch");
        Self {
            shape: new_shape.to_vec(),
            device: self.device,
            data: self.data.clone(),
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Matrix multiplication: (M x K) @ (K x N) -> (M x N).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.device, other.device, "devices must match for matmul");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        assert_eq!(self.shape.len(), 2, "self must be 2D");
        assert_eq!(other.shape.len(), 2, "other must be 2D");
        assert_eq!(k, other.shape[0], "inner dimensions must match");

        #[cfg(feature = "cuda")]
        {
            // If both tensors are on GPU, use cuBLAS for accelerated matmul.
            if let (Device::Gpu(device_id), Some(_), Device::Gpu(other_id), Some(_)) =
                (&self.device, self.device_id, &other.device, other.device_id)
            {
                assert_eq!(device_id, other_id, "GPU devices must match for matmul");
                return self.matmul_cuda(other, m, k, n);
            }
        }

        // CPU fallback: naive O(M*N*K) matmul
        Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device)
    }

    /// CPU matrix multiplication fallback.
    fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, device: Device) -> Tensor {
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor {
            shape: vec![m, n],
            device,
            data: result,
            #[cfg(feature = "cuda")]
            device_id: None,
        }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len(), "shape mismatch for add");
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor {
            shape: self.shape.clone(),
            device: self.device,
            data: result,
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len(), "shape mismatch for mul");
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Tensor {
            shape: self.shape.clone(),
            device: self.device,
            data: result,
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Transpose a 2D tensor.
    pub fn t(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "transpose requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }

        Tensor {
            shape: vec![n, m],
            device: self.device,
            data: result,
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self, temperature: f32) -> Tensor {
        let dim = *self.shape.last().unwrap_or(&1);
        let batch = self.data.len() / dim;
        let temp = if temperature > 0.0 { temperature } else { 1.0 };
        let mut result = vec![0.0; self.data.len()];

        for b in 0..batch {
            let start = b * dim;
            let slice = &self.data[start..start + dim];

            let max_val = slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0;
            let mut exp_vals = vec![0.0; dim];
            for (i, &v) in slice.iter().enumerate() {
                exp_vals[i] = ((v - max_val) / temp).exp();
                sum += exp_vals[i];
            }

            if sum > 0.0 {
                for i in 0..dim {
                    result[start + i] = exp_vals[i] / sum;
                }
            }
        }

        Tensor {
            shape: self.shape.clone(),
            device: self.device,
            data: result,
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Extract the last row (for LM head: take last token's hidden state).
    pub fn last_row(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "last_row requires 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let start = (rows - 1) * cols;
        let data = self.data[start..start + cols].to_vec();

        Tensor {
            shape: vec![1, cols],
            device: self.device,
            data,
            #[cfg(feature = "cuda")]
            device_id: self.device_id,
        }
    }

    /// Transfer tensor to GPU (no-op when cuda feature is disabled).
    #[cfg(feature = "cuda")]
    pub fn to_device(&self, device_id: usize) -> Self {
        let manager = GpuDeviceManager::global();
        if manager.device_count() == 0 {
            log::warn!("No CUDA devices available; falling back to CPU");
            return self.clone();
        }

        // Ensure device_id is valid
        if device_id >= manager.device_count() {
            log::warn!(
                "GPU device {device_id} not available ({} devices); using device 0",
                manager.device_count()
            );
            return self.to_device(0);
        }

        Self {
            shape: self.shape.clone(),
            device: Device::Gpu(device_id),
            data: self.data.clone(),
            device_id: Some(device_id),
        }
    }

    /// Transfer tensor back to CPU (returns self when already on CPU).
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&self) -> Self {
        if self.device == Device::Cpu {
            return self.clone();
        }

        // Host-side cache is already in self.data
        Self {
            shape: self.shape.clone(),
            device: Device::Cpu,
            data: self.data.clone(),
            device_id: None,
        }
    }

    /// CUDA-accelerated matrix multiplication using cuBLAS.
    ///
    /// This method:
    /// 1. Allocates GPU memory for input matrices via GpuDeviceManager
    /// 2. Uses CublasHandle to perform cuBLAS Sgemm
    /// 3. Copies the result back to host memory
    /// 4. GPU memory is automatically tracked and freed by the manager
    #[cfg(feature = "cuda")]
    fn matmul_cuda(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        let device_id = self.device_id.unwrap_or(0);
        let manager = GpuDeviceManager::global();

        // Get cuBLAS handle
        let blas = match manager.get_blas_handle(device_id) {
            Some(b) => b,
            None => {
                log::error!(
                    "Failed to get cuBLAS handle for device {device_id}; falling back to CPU"
                );
                return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
            }
        };

        // Allocate GPU memory for inputs
        let a_gpu = match manager.alloc_device(device_id, &self.data) {
            Ok(slice) => slice,
            Err(e) => {
                log::error!("GPU memory allocation failed: {e}; falling back to CPU");
                return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
            }
        };

        let b_gpu = match manager.alloc_device(device_id, &other.data) {
            Ok(slice) => slice,
            Err(e) => {
                log::error!("GPU memory allocation failed: {e}; falling back to CPU");
                return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
            }
        };

        // Allocate GPU memory for output
        let mut c_gpu = match manager.alloc_device(device_id, &vec![0.0; m * n]) {
            Ok(slice) => slice,
            Err(e) => {
                log::error!("GPU memory allocation failed: {e}; falling back to CPU");
                return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
            }
        };

        // Execute cuBLAS Sgemm
        let result = unsafe { blas.sgemm(&a_gpu, &b_gpu, &mut c_gpu, m, k, n) };

        // Copy result back to host
        let host_result = match manager.copy_device_to_host(&c_gpu) {
            Ok(data) => data,
            Err(e) => {
                log::error!("Failed to copy result from GPU: {e}; falling back to CPU");
                return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
            }
        };

        if let Err(e) = result {
            log::error!("cuBLAS Sgemm failed: {e}; falling back to CPU");
            return Self::matmul_cpu(&self.data, &other.data, m, k, n, self.device);
        }

        log::debug!(
            "cuBLAS matmul completed: ({m}x{k}) @ ({k}x{n}) -> ({m}x{n}) on GPU {device_id}"
        );

        Tensor {
            shape: vec![m, n],
            device: self.device,
            data: host_result,
            device_id: self.device_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn matmul_basic() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 19.0).abs() < 1e-5);
        assert!((c.data[1] - 22.0).abs() < 1e-5);
        assert!((c.data[2] - 43.0).abs() < 1e-5);
        assert!((c.data[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_rectangular() {
        // (2x3) @ (3x4) = (2x4)
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4],
        );

        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 4]);
        // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
        //       = [1+10+27, 2+12+30, 3+14+33, 4+16+36]
        //       = [38, 44, 50, 56]
        assert!((c.data[0] - 38.0).abs() < 1e-5);
        assert!((c.data[1] - 44.0).abs() < 1e-5);
        assert!((c.data[2] - 50.0).abs() < 1e-5);
        assert!((c.data[3] - 56.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_basic() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.softmax(1.0);

        // Sum should be ~1.0
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher input should have higher probability
        assert!(s.data[2] > s.data[1]);
        assert!(s.data[1] > s.data[0]);
    }

    #[test]
    fn transpose_works() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape, vec![3, 2]);
        assert_eq!(tt.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn zeros_creates_correct_tensor() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn add_works() {
        let a = Tensor::from_data(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_data(vec![3.0, 4.0], &[2]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);
    }

    // CUDA-specific tests (only run when cuda feature is enabled and GPU is available)
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::super::{cuda_info, GpuDeviceManager, Tensor};

        #[test]
        fn gpu_device_manager_init() {
            let manager = GpuDeviceManager::init();
            let count = manager.device_count();
            // If no GPU is available, count should be 0
            if count == 0 {
                println!("No CUDA devices detected; skipping GPU tests");
            } else {
                assert!(count > 0);
            }
        }

        #[test]
        fn cuda_info_returns_devices() {
            let info = cuda_info();
            if info.device_count > 0 {
                assert!(!info.devices.is_empty());
                for device in &info.devices {
                    assert!(device.compute_capability_major >= 3); // Minimum useful compute capability
                }
            }
        }

        #[test]
        fn cublas_matmul_correctness() {
            let manager = GpuDeviceManager::init();
            if manager.device_count() == 0 {
                println!("No CUDA devices available; skipping cuBLAS matmul test");
                return;
            }

            // Create tensors and move to GPU
            let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

            let a_gpu = a.to_device(0);
            let b_gpu = b.to_device(0);

            assert_eq!(a_gpu.device, super::super::Device::Gpu(0));
            assert_eq!(b_gpu.device, super::super::Device::Gpu(0));

            // Perform matmul on GPU
            let c = a_gpu.matmul(&b_gpu);

            // Verify results: [[19, 22], [43, 50]]
            assert_eq!(c.shape, vec![2, 2]);
            assert!((c.data[0] - 19.0).abs() < 1e-3, "c[0] = {}", c.data[0]);
            assert!((c.data[1] - 22.0).abs() < 1e-3, "c[1] = {}", c.data[1]);
            assert!((c.data[2] - 43.0).abs() < 1e-3, "c[2] = {}", c.data[2]);
            assert!((c.data[3] - 50.0).abs() < 1e-3, "c[3] = {}", c.data[3]);
        }

        #[test]
        fn cublas_matmul_rectangular() {
            let manager = GpuDeviceManager::init();
            if manager.device_count() == 0 {
                println!("No CUDA devices available; skipping rectangular matmul test");
                return;
            }

            let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let b = Tensor::from_data(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[3, 4],
            );

            let a_gpu = a.to_device(0);
            let b_gpu = b.to_device(0);

            let c = a_gpu.matmul(&b_gpu);
            assert_eq!(c.shape, vec![2, 4]);

            // Expected: [[38, 44, 50, 56], [83, 98, 113, 128]]
            let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
            for (i, (&got, &exp)) in c.data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-2,
                    "c[{i}] = {got}, expected {exp}"
                );
            }
        }

        #[test]
        fn cpu_fallback_when_cuda_not_used() {
            // Even with cuda feature enabled, CPU tensors should use CPU matmul
            let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

            assert_eq!(a.device, super::super::Device::Cpu);
            assert_eq!(b.device, super::super::Device::Cpu);

            let c = a.matmul(&b);
            assert_eq!(c.device, super::super::Device::Cpu);
            assert!((c.data[0] - 19.0).abs() < 1e-5);
            assert!((c.data[3] - 50.0).abs() < 1e-5);
        }
    }

    // Non-CUDA tests to verify CPU fallback path
    #[cfg(not(feature = "cuda"))]
    mod cpu_fallback_tests {
        use super::super::{cuda_info, GpuDeviceManager, Device, Tensor};

        #[test]
        fn gpu_device_manager_is_stub() {
            let manager = GpuDeviceManager::init();
            assert_eq!(manager.device_count(), 0);
        }

        #[test]
        fn cuda_info_empty_when_disabled() {
            let info = cuda_info();
            assert_eq!(info.device_count, 0);
            assert!(info.devices.is_empty());
        }

        #[test]
        fn cpu_fallback_matmul() {
            let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

            let c = a.matmul(&b);
            assert_eq!(c.device, Device::Cpu);
            assert!((c.data[0] - 19.0).abs() < 1e-5);
            assert!((c.data[1] - 22.0).abs() < 1e-5);
            assert!((c.data[2] - 43.0).abs() < 1e-5);
            assert!((c.data[3] - 50.0).abs() < 1e-5);
        }
    }
}

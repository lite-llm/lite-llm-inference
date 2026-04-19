use std::collections::BTreeMap;

use crate::error::{InferenceError, InferenceResult};
use crate::types::{SessionId, TenantId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTier {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvEntry {
    pub layer: u16,
    pub head: u16,
    pub position: u64,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub tier: KvTier,
}

impl KvEntry {
    /// Transfer this KV entry to GPU device memory (when `cuda` feature is enabled).
    ///
    /// Returns a `GpuKvEntry` containing the same data but with device-side
    /// allocations for key and value vectors.
    #[cfg(feature = "cuda")]
    pub fn to_gpu(&self, device_id: usize) -> InferenceResult<GpuKvEntry> {
        use crate::gpu_backend::GpuDeviceManager;

        let manager = GpuDeviceManager::global();
        if manager.device_count() == 0 {
            return Err(InferenceError::InvalidState(
                "no CUDA devices available for GPU KV cache",
            ));
        }

        let device_id = if device_id >= manager.device_count() {
            0
        } else {
            device_id
        };

        let key_gpu = manager.alloc_device(device_id, &self.key).map_err(|e| {
            InferenceError::IoError(format!("GPU key allocation failed: {}", e))
        })?;
        let value_gpu = manager
            .alloc_device(device_id, &self.value)
            .map_err(|e| {
                InferenceError::IoError(format!("GPU value allocation failed: {}", e))
            })?;

        Ok(GpuKvEntry {
            layer: self.layer,
            head: self.head,
            position: self.position,
            key_gpu,
            value_gpu,
            device_id,
            tier: self.tier,
            // Keep host-side cache for quick access
            key_host: self.key.clone(),
            value_host: self.value.clone(),
        })
    }

    /// No-op when CUDA is not enabled — returns self wrapped in Result.
    #[cfg(not(feature = "cuda"))]
    pub fn to_gpu(&self, _device_id: usize) -> InferenceResult<StubGpuKvEntry> {
        Err(InferenceError::InvalidState(
            "cuda feature not enabled; GPU KV cache unavailable",
        ))
    }
}

// ---------------------------------------------------------------------------
// GPU KV Cache Types
// ---------------------------------------------------------------------------

/// A GPU-resident KV cache entry with device memory allocations.
///
/// When the `cuda` feature is enabled, this type wraps `cudarc::driver::safe::CudaSlice`
/// handles for the key and value vectors, enabling GPU-resident KV state that avoids
/// repeated host-to-device transfers during inference.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GpuKvEntry {
    pub layer: u16,
    pub head: u16,
    pub position: u64,
    /// GPU device memory handle for the key vector.
    pub key_gpu: cudarc::driver::safe::CudaSlice<f32>,
    /// GPU device memory handle for the value vector.
    pub value_gpu: cudarc::driver::safe::CudaSlice<f32>,
    /// The GPU device ID where this entry is allocated.
    pub device_id: usize,
    pub tier: KvTier,
    /// Host-side cache for fallback and debugging.
    pub key_host: Vec<f32>,
    pub value_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuKvEntry {
    /// Transfer the GPU-resident entry back to CPU memory.
    ///
    /// This copies data from device memory back to host and returns a standard `KvEntry`.
    pub fn to_cpu(&self) -> InferenceResult<KvEntry> {
        use crate::gpu_backend::GpuDeviceManager;

        let manager = GpuDeviceManager::global();
        let key = manager.copy_device_to_host(&self.key_gpu).map_err(|e| {
            InferenceError::IoError(format!("GPU->CPU key transfer failed: {}", e))
        })?;
        let value = manager
            .copy_device_to_host(&self.value_gpu)
            .map_err(|e| {
                InferenceError::IoError(format!("GPU->CPU value transfer failed: {}", e))
            })?;

        Ok(KvEntry {
            layer: self.layer,
            head: self.head,
            position: self.position,
            key,
            value,
            tier: self.tier,
        })
    }
}

/// Stub type returned when `cuda` feature is disabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct StubGpuKvEntry;

// ---------------------------------------------------------------------------
// GpuKvCache
// ---------------------------------------------------------------------------

/// GPU-resident KV cache manager that keeps hot entries in device memory.
///
/// When the `cuda` feature is enabled, this cache automatically promotes
/// recently-accessed (hot) entries to GPU memory for faster inference.
/// Warm and cold entries remain in host memory.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GpuKvCache {
    /// GPU device ID for allocations.
    device_id: usize,
    /// Maximum number of entries to keep in GPU memory.
    max_gpu_entries: usize,
    /// Hot entries resident on GPU (keyed by position).
    gpu_entries: BTreeMap<u64, GpuKvEntry>,
    /// Fallback: all entries in host memory.
    host_entries: BTreeMap<u64, KvEntry>,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    /// Create a new GPU KV cache for the given device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The CUDA device ID to use for GPU allocations.
    /// * `max_gpu_entries` - Maximum number of entries to keep in GPU memory.
    ///   When exceeded, the oldest (lowest position) entries are evicted to host.
    pub fn new(device_id: usize, max_gpu_entries: usize) -> InferenceResult<Self> {
        use crate::gpu_backend::GpuDeviceManager;

        let manager = GpuDeviceManager::global();
        if manager.device_count() == 0 {
            return Err(InferenceError::InvalidState(
                "no CUDA devices available for GPU KV cache",
            ));
        }

        let device_id = if device_id >= manager.device_count() {
            0
        } else {
            device_id
        };

        if max_gpu_entries == 0 {
            return Err(InferenceError::InvalidConfig(
                "max_gpu_entries must be greater than zero",
            ));
        }

        Ok(Self {
            device_id,
            max_gpu_entries,
            gpu_entries: BTreeMap::new(),
            host_entries: BTreeMap::new(),
        })
    }

    /// Append a new KV entry to the cache, promoting it to GPU if hot.
    pub fn append(
        &mut self,
        layer: u16,
        head: u16,
        position: u64,
        key: &[f32],
        value: &[f32],
    ) -> InferenceResult<()> {
        if key.len() != value.len() {
            return Err(InferenceError::InvalidInput(
                "kv key/value lengths must be equal",
            ));
        }

        let entry = KvEntry {
            layer,
            head,
            position,
            key: key.to_vec(),
            value: value.to_vec(),
            tier: KvTier::Hot,
        };

        // Try to promote to GPU
        if self.gpu_entries.len() < self.max_gpu_entries {
            match entry.to_gpu(self.device_id) {
                Ok(gpu_entry) => {
                    self.gpu_entries.insert(position, gpu_entry);
                    return Ok(());
                }
                Err(_) => {
                    // Fall back to host storage
                }
            }
        }

        // Evict oldest GPU entry if at capacity
        if self.gpu_entries.len() >= self.max_gpu_entries {
            if let Some((&oldest_pos, oldest_gpu_entry)) = self.gpu_entries.first_key_value() {
                let host_entry = oldest_gpu_entry.to_cpu()?;
                self.gpu_entries.remove(&oldest_pos);
                self.host_entries.insert(oldest_pos, host_entry);
            }
        }

        // Try GPU promotion again after eviction
        if self.gpu_entries.len() < self.max_gpu_entries {
            if let Ok(gpu_entry) = entry.to_gpu(self.device_id) {
                self.gpu_entries.insert(position, gpu_entry);
                return Ok(());
            }
        }

        // Final fallback: store in host memory
        self.host_entries.insert(position, entry);
        Ok(())
    }

    /// Retrieve entries by position range, fetching from GPU or host.
    pub fn slice(
        &self,
        layer: u16,
        head: u16,
        from_position: u64,
        to_position_exclusive: u64,
    ) -> InferenceResult<Vec<KvEntry>> {
        if from_position > to_position_exclusive {
            return Err(InferenceError::InvalidInput(
                "from_position must be <= to_position_exclusive",
            ));
        }

        let mut results = Vec::new();

        // Check GPU entries first
        for (&pos, gpu_entry) in self.gpu_entries.range(from_position..to_position_exclusive) {
            if gpu_entry.layer == layer && gpu_entry.head == head {
                results.push(gpu_entry.to_cpu()?);
            }
        }

        // Check host entries
        for (&pos, entry) in self.host_entries.range(from_position..to_position_exclusive) {
            if entry.layer == layer && entry.head == head {
                results.push(entry.clone());
            }
        }

        results.sort_by(|a, b| a.position.cmp(&b.position));
        Ok(results)
    }

    /// Get the number of entries currently in GPU memory.
    pub fn gpu_entry_count(&self) -> usize {
        self.gpu_entries.len()
    }

    /// Get the total number of entries (GPU + host).
    pub fn total_entry_count(&self) -> usize {
        self.gpu_entries.len() + self.host_entries.len()
    }

    /// Remove an entry by position, freeing GPU memory if applicable.
    pub fn remove(&mut self, position: u64) {
        self.gpu_entries.remove(&position);
        self.host_entries.remove(&position);
    }
}

// ---------------------------------------------------------------------------
// CPU-only stub for GpuKvCache
// ---------------------------------------------------------------------------

/// Stub type when `cuda` feature is disabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct GpuKvCache;

#[cfg(not(feature = "cuda"))]
impl GpuKvCache {
    pub fn new(_device_id: usize, _max_gpu_entries: usize) -> InferenceResult<Self> {
        Err(InferenceError::InvalidState(
            "cuda feature not enabled; GPU KV cache unavailable",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub hot_token_limit: usize,
    pub warm_token_limit: usize,
    pub total_token_limit: usize,
    pub sliding_window_tokens: Option<usize>,
}

impl KvCacheConfig {
    pub fn validate(self) -> InferenceResult<Self> {
        if self.hot_token_limit == 0 {
            return Err(InferenceError::InvalidConfig(
                "hot_token_limit must be greater than zero",
            ));
        }
        if self.total_token_limit == 0 || self.total_token_limit < self.hot_token_limit {
            return Err(InferenceError::InvalidConfig(
                "total_token_limit must be >= hot_token_limit",
            ));
        }

        if self.hot_token_limit + self.warm_token_limit > self.total_token_limit {
            return Err(InferenceError::InvalidConfig(
                "hot + warm limits must not exceed total_token_limit",
            ));
        }

        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SessionCache {
    records: Vec<KvEntry>,
}

impl SessionCache {
    fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    config: KvCacheConfig,
    sessions: BTreeMap<(TenantId, SessionId), SessionCache>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> InferenceResult<Self> {
        Ok(Self {
            config: config.validate()?,
            sessions: BTreeMap::new(),
        })
    }

    pub fn append(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
        layer: u16,
        head: u16,
        position: u64,
        key: &[f32],
        value: &[f32],
    ) -> InferenceResult<()> {
        if key.len() != value.len() {
            return Err(InferenceError::InvalidInput(
                "kv key/value lengths must be equal",
            ));
        }

        let session = self
            .sessions
            .entry((tenant_id, session_id))
            .or_insert_with(SessionCache::new);

        session.records.push(KvEntry {
            layer,
            head,
            position,
            key: key.to_vec(),
            value: value.to_vec(),
            tier: KvTier::Hot,
        });

        self.rebalance_session(tenant_id, session_id)?;
        Ok(())
    }

    pub fn slice(
        &self,
        tenant_id: TenantId,
        session_id: SessionId,
        layer: u16,
        head: u16,
        from_position: u64,
        to_position_exclusive: u64,
    ) -> InferenceResult<Vec<KvEntry>> {
        if from_position > to_position_exclusive {
            return Err(InferenceError::InvalidInput(
                "from_position must be <= to_position_exclusive",
            ));
        }

        let session = self.sessions.get(&(tenant_id, session_id)).ok_or_else(|| {
            InferenceError::TenantViolation("session not found for tenant".to_owned())
        })?;

        let mut out = session
            .records
            .iter()
            .filter(|entry| entry.layer == layer)
            .filter(|entry| entry.head == head)
            .filter(|entry| {
                entry.position >= from_position && entry.position < to_position_exclusive
            })
            .cloned()
            .collect::<Vec<KvEntry>>();

        out.sort_by(|a, b| a.position.cmp(&b.position));
        Ok(out)
    }

    pub fn reset_session(&mut self, tenant_id: TenantId, session_id: SessionId) {
        self.sessions.remove(&(tenant_id, session_id));
    }

    pub fn session_len(&self, tenant_id: TenantId, session_id: SessionId) -> usize {
        self.sessions
            .get(&(tenant_id, session_id))
            .map(|session| session.records.len())
            .unwrap_or(0)
    }

    pub fn tier_counts(&self, tenant_id: TenantId, session_id: SessionId) -> (usize, usize, usize) {
        if let Some(session) = self.sessions.get(&(tenant_id, session_id)) {
            let hot = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Hot)
                .count();
            let warm = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Warm)
                .count();
            let cold = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Cold)
                .count();
            (hot, warm, cold)
        } else {
            (0, 0, 0)
        }
    }

    fn rebalance_session(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
    ) -> InferenceResult<()> {
        let session =
            self.sessions
                .get_mut(&(tenant_id, session_id))
                .ok_or(InferenceError::InvalidState(
                    "session missing during rebalance",
                ))?;

        session.records.sort_by(|a, b| {
            a.position
                .cmp(&b.position)
                .then(a.layer.cmp(&b.layer))
                .then(a.head.cmp(&b.head))
        });

        if let Some(window) = self.config.sliding_window_tokens {
            if window == 0 {
                return Err(InferenceError::InvalidConfig(
                    "sliding_window_tokens must be greater than zero when set",
                ));
            }

            if session.records.len() > window {
                let drop_count = session.records.len() - window;
                session.records.drain(0..drop_count);
            }
        }

        if session.records.len() > self.config.total_token_limit {
            let drop_count = session.records.len() - self.config.total_token_limit;
            session.records.drain(0..drop_count);
        }

        let total = session.records.len();
        for (idx, entry) in session.records.iter_mut().enumerate() {
            let reverse_idx = total - idx;
            entry.tier = if reverse_idx <= self.config.hot_token_limit {
                KvTier::Hot
            } else if reverse_idx <= self.config.hot_token_limit + self.config.warm_token_limit {
                KvTier::Warm
            } else {
                KvTier::Cold
            };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuKvCache, KvCache, KvCacheConfig, KvEntry, KvTier};

    fn cache() -> KvCache {
        KvCache::new(KvCacheConfig {
            hot_token_limit: 2,
            warm_token_limit: 2,
            total_token_limit: 5,
            sliding_window_tokens: None,
        })
        .expect("cache should initialize")
    }

    #[test]
    fn append_and_slice_preserve_order() {
        let mut cache = cache();
        for pos in 0..4 {
            cache
                .append(1, 10, 0, 0, pos, &[pos as f32], &[pos as f32 + 1.0])
                .expect("append should succeed");
        }

        let slice = cache
            .slice(1, 10, 0, 0, 0, 4)
            .expect("slice should succeed");
        let positions = slice
            .iter()
            .map(|entry| entry.position)
            .collect::<Vec<u64>>();
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn deterministic_tier_assignment_after_rebalance() {
        let mut cache = cache();
        for pos in 0..5 {
            cache
                .append(1, 10, 0, 0, pos, &[0.0], &[0.0])
                .expect("append should succeed");
        }

        let (hot, warm, cold) = cache.tier_counts(1, 10);
        assert_eq!((hot, warm, cold), (2, 2, 1));

        let entries = cache
            .slice(1, 10, 0, 0, 0, 10)
            .expect("slice should succeed");
        assert_eq!(entries.first().map(|entry| entry.tier), Some(KvTier::Cold));
        assert_eq!(entries.last().map(|entry| entry.tier), Some(KvTier::Hot));
    }

    #[test]
    fn total_limit_evicts_oldest_entries() {
        let mut cache = cache();
        for pos in 0..8 {
            cache
                .append(1, 10, 0, 0, pos, &[0.0], &[0.0])
                .expect("append should succeed");
        }

        let entries = cache
            .slice(1, 10, 0, 0, 0, 100)
            .expect("slice should succeed");
        let positions = entries
            .iter()
            .map(|entry| entry.position)
            .collect::<Vec<u64>>();
        assert_eq!(positions, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn tenant_isolation_keeps_sessions_separate() {
        let mut cache = cache();
        cache
            .append(1, 10, 0, 0, 0, &[1.0], &[2.0])
            .expect("append should succeed");
        cache
            .append(2, 10, 0, 0, 0, &[3.0], &[4.0])
            .expect("append should succeed");

        let a = cache
            .slice(1, 10, 0, 0, 0, 1)
            .expect("slice should succeed");
        let b = cache
            .slice(2, 10, 0, 0, 0, 1)
            .expect("slice should succeed");

        assert_ne!(a[0].key, b[0].key);

        cache.reset_session(1, 10);
        assert_eq!(cache.session_len(1, 10), 0);
        assert_eq!(cache.session_len(2, 10), 1);
    }

    // -----------------------------------------------------------------------
    // GPU KV Cache Tests
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn gpu_kv_cache_returns_error_when_cuda_disabled() {
        let result = GpuKvCache::new(0, 10);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn kv_entry_to_gpu_returns_error_when_cuda_disabled() {
        let entry = KvEntry {
            layer: 0,
            head: 0,
            position: 0,
            key: vec![1.0, 2.0],
            value: vec![3.0, 4.0],
            tier: KvTier::Hot,
        };
        let result = entry.to_gpu(0);
        assert!(result.is_err());
    }
}

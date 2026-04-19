//! Modern transformer architecture components matching 2024–2026 LLM designs.
//!
//! - **RoPE (Rotary Position Embeddings)**: encodes relative position information
//!   into the attention computation, enabling smooth extrapolation to long context windows.
//! - **RMSNorm**: replaces LayerNorm with a computationally cheaper normalization
//!   that provides equivalent training stability.
//! - **SwiGLU**: replaces standard GELU with SwiGLU gating for better expressivity.
//! - **GQA (Grouped-Query Attention)**: shares Key/Value heads across query groups
//!   to reduce KV-cache memory footprint during inference.

use crate::gpu_backend::Tensor;

/// Rotary Position Embedding (RoPE).
///
/// Applies rotary embeddings to query and key tensors before attention computation.
/// RoPE encodes the relative distance between tokens as rotation angles,
/// allowing the model to naturally handle sequences longer than training length.
///
/// Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    #[allow(dead_code)]
    dim: usize,
    max_seq_len: usize,
    #[allow(dead_code)]
    base: f32,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Create a new RoPE module.
    ///
    /// - `dim`: embedding dimension (must be even)
    /// - `max_seq_len`: maximum sequence length to pre-compute
    /// - `base`: frequency base (default 10000.0)
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(dim % 2 == 0, "RoPE dimension must be even");

        // Pre-compute cos/sin cache for all positions
        let mut cos_cache = vec![0.0; max_seq_len * dim];
        let mut sin_cache = vec![0.0; max_seq_len * dim];

        for pos in 0..max_seq_len {
            for i in 0..dim / 2 {
                let freq = 1.0 / base.powf((2 * i) as f32 / dim as f32);
                let theta = pos as f32 * freq;
                cos_cache[pos * dim + 2 * i] = theta.cos();
                cos_cache[pos * dim + 2 * i + 1] = theta.cos();
                sin_cache[pos * dim + 2 * i] = theta.sin();
                sin_cache[pos * dim + 2 * i + 1] = theta.sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            base,
            cos_cache: Tensor::from_data(cos_cache, &[max_seq_len, dim]),
            sin_cache: Tensor::from_data(sin_cache, &[max_seq_len, dim]),
        }
    }

    /// Apply rotary embeddings to a tensor of shape [seq_len, dim].
    ///
    /// For each position p and dimension pair (2i, 2i+1):
    ///   [x_2i, x_2i+1] → [x_2i * cos(p*θ_i) - x_2i+1 * sin(p*θ_i),
    ///                      x_2i * sin(p*θ_i) + x_2i+1 * cos(p*θ_i)]
    pub fn apply(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape.len(), 2, "input must be 2D [seq_len, dim]");
        let seq_len = x.shape[0];
        let dim = x.shape[1];
        assert_eq!(dim, self.dim, "dimension mismatch");
        assert!(seq_len <= self.max_seq_len, "sequence exceeds max length");

        let mut result = vec![0.0; seq_len * dim];

        for pos in 0..seq_len {
            for i in 0..dim / 2 {
                let x0 = x.data[pos * dim + 2 * i];
                let x1 = x.data[pos * dim + 2 * i + 1];
                let cos = self.cos_cache.data[pos * dim + 2 * i];
                let sin = self.sin_cache.data[pos * dim + 2 * i];

                result[pos * dim + 2 * i] = x0 * cos - x1 * sin;
                result[pos * dim + 2 * i + 1] = x0 * sin + x1 * cos;
            }
        }

        Tensor::from_data(result, &[seq_len, dim])
    }
}

/// Root Mean Square Layer Normalization (RMSNorm).
///
/// Normalizes activations using only the root mean square statistic,
/// omitting the mean-centering step of LayerNorm. This is computationally
/// cheaper and empirically provides equivalent training stability.
///
/// Reference: Zhang & Sennrich "Root Mean Square Layer Normalization" (2019)
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Vec<f32>,
    dim: usize,
    eps: f32,
}

impl RmsNorm {
    /// Create RMSNorm with learned weight vector.
    pub fn new(dim: usize, eps: f32) -> Self {
        // Initialize weights to 1.0 (identity transform)
        let weight = vec![1.0; dim];
        Self { weight, dim, eps }
    }

    /// Create RMSNorm with custom weights.
    pub fn with_weights(dim: usize, weights: Vec<f32>, eps: f32) -> Self {
        assert_eq!(weights.len(), dim, "weight length must match dimension");
        Self {
            weight: weights,
            dim,
            eps,
        }
    }

    /// Normalize a tensor of shape `[batch, dim]` or `[dim]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let dim = x.shape.last().copied().unwrap_or(x.data.len());
        let batch = x.data.len() / dim;
        let mut result = vec![0.0; x.data.len()];

        for b in 0..batch {
            let start = b * dim;
            let slice = &x.data[start..start + dim];

            // Compute RMS: sqrt(mean(x^2) + eps)
            let mean_square: f32 = slice.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rms = (mean_square + self.eps).sqrt();

            // Normalize and scale
            for i in 0..dim {
                result[start + i] = (slice[i] / rms) * self.weight[i];
            }
        }

        Tensor::from_data(result, &x.shape)
    }
}

/// SwiGLU Feed-Forward Network activation.
///
/// SwiGLU(x, W, V) = Swish(xW) ⊗ xV
/// where Swish(x) = x * sigmoid(x)
///
/// This replaces the standard GELU-based FFN and provides better
/// expressivity with similar compute cost.
///
/// Reference: Shazeer "GLU Variants Improve Transformer" (2020)
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    assert_eq!(gate.data.len(), up.data.len(), "gate and up must match");

    let dim = gate.data.len();
    let mut result = vec![0.0; dim];

    for i in 0..dim {
        // Swish(gate[i]) = gate[i] * sigmoid(gate[i])
        let swish = gate.data[i] / (1.0 + (-gate.data[i]).exp());
        result[i] = swish * up.data[i];
    }

    Tensor::from_data(result, gate.shape.as_slice())
}

/// Grouped-Query Attention (GQA) helper.
///
/// GQA shares Key and Value heads across groups of Query heads,
/// reducing the KV-cache size from O(n_heads) to O(n_kv_heads)
/// while maintaining most of the quality of full MHA.
///
/// Configuration: n_query_heads must be divisible by n_kv_heads.
///
/// Reference: Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models
///            from Multi-Head Checkpoints" (2023)
#[derive(Debug, Clone)]
pub struct GroupedQueryConfig {
    pub n_query_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl GroupedQueryConfig {
    pub fn new(n_query_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(
            n_query_heads % n_kv_heads == 0,
            "n_query_heads must be divisible by n_kv_heads"
        );
        Self {
            n_query_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// Number of query heads per KV head (the "group size").
    pub fn group_size(&self) -> usize {
        self.n_query_heads / self.n_kv_heads
    }

    /// Repeat KV heads to match query head count.
    /// Each KV head is repeated `group_size` times.
    pub fn repeat_kv(&self, kv: &Tensor) -> Tensor {
        assert_eq!(kv.shape.len(), 3, "KV must be [seq_len, n_kv_heads, head_dim]");
        let seq_len = kv.shape[0];
        let n_kv_heads = kv.shape[1];
        let head_dim = kv.shape[2];

        assert_eq!(n_kv_heads, self.n_kv_heads);
        assert_eq!(head_dim, self.head_dim);

        let group_size = self.group_size();
        let output_len = seq_len * self.n_query_heads * head_dim;
        let mut result = vec![0.0; output_len];

        for s in 0..seq_len {
            for kv_h in 0..n_kv_heads {
                for g in 0..group_size {
                    let q_h = kv_h * group_size + g;
                    for d in 0..head_dim {
                        let src_idx = (s * n_kv_heads * head_dim) + (kv_h * head_dim) + d;
                        let dst_idx = (s * self.n_query_heads * head_dim) + (q_h * head_dim) + d;
                        result[dst_idx] = kv.data[src_idx];
                    }
                }
            }
        }

        Tensor::from_data(result, &[seq_len, self.n_query_heads, head_dim])
    }
}

/// A modern FFN block using SwiGLU activation and RMSNorm.
///
/// This replaces the standard FeedForward (GELU + LayerNorm)
/// with the 2024-2026 standard architecture.
#[derive(Debug, Clone)]
pub struct ModernFeedForward {
    gate_proj: Tensor, // W_gate: [hidden_dim, intermediate_dim]
    up_proj: Tensor,   // W_up: [hidden_dim, intermediate_dim]
    down_proj: Tensor, // W_down: [intermediate_dim, hidden_dim]
    hidden_dim: usize,
    #[allow(dead_code)]
    intermediate_dim: usize,
    rms_norm: RmsNorm,
}

impl ModernFeedForward {
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        let scale = (1.0 / hidden_dim as f32).sqrt();
        Self {
            gate_proj: Tensor::randn(&[hidden_dim, intermediate_dim], scale),
            up_proj: Tensor::randn(&[hidden_dim, intermediate_dim], scale),
            down_proj: Tensor::randn(&[intermediate_dim, hidden_dim], scale),
            hidden_dim,
            intermediate_dim,
            rms_norm: RmsNorm::new(hidden_dim, 1e-5),
        }
    }

    /// Forward pass: RMSNorm(x) → SwiGLU(gate, up) → down_proj
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. RMSNorm pre-normalization
        let x_norm = self.rms_norm.forward(x);

        // 2. Project to gate and up spaces
        // x_norm: [seq_len, hidden_dim]
        // gate_proj: [hidden_dim, intermediate_dim]
        let gate = x_norm.reshape(&[x_norm.shape[0], self.hidden_dim]);
        let up = gate.clone(); // Same input for both

        // Naive projection (in production, use GPU matmul)
        let gate_proj_result = self.apply_linear(&gate, &self.gate_proj);
        let up_proj_result = self.apply_linear(&up, &self.up_proj);

        // 3. SwiGLU activation
        let activated = swiglu(&gate_proj_result, &up_proj_result);

        // 4. Down projection
        self.apply_linear(&activated, &self.down_proj)
    }

    fn apply_linear(&self, input: &Tensor, weight: &Tensor) -> Tensor {
        let seq_len = input.shape[0];
        let in_dim = self.hidden_dim;
        let out_dim = weight.shape[1];

        let mut result = vec![0.0; seq_len * out_dim];
        for s in 0..seq_len {
            for o in 0..out_dim {
                let mut sum = 0.0;
                for i in 0..in_dim {
                    sum += input.data[s * in_dim + i] * weight.data[i * out_dim + o];
                }
                result[s * out_dim + o] = sum;
            }
        }

        Tensor::from_data(result, &[seq_len, out_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::{GroupedQueryConfig, RmsNorm, RotaryEmbedding, Tensor};

    #[test]
    fn rope_apply_is_deterministic() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let y1 = rope.apply(&x);
        let y2 = rope.apply(&x);
        assert_eq!(y1.data, y2.data);
    }

    #[test]
    fn rope_preserves_norm() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let x = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], &[1, 4]);
        let y = rope.apply(&x);

        // RoPE is a rotation, so L2 norm should be preserved
        let norm_x: f32 = x.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_y: f32 = y.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_x - norm_y).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_zero_input_gives_zero() {
        let norm = RmsNorm::new(4, 1e-5);
        let x = Tensor::zeros(&[2, 4]);
        let y = norm.forward(&x);

        // All zeros should remain zeros
        assert!(y.data.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn rms_norm_normalizes_unit_variance() {
        let norm = RmsNorm::new(3, 1e-5);
        let x = Tensor::from_data(vec![3.0, 4.0, 0.0], &[1, 3]);
        let y = norm.forward(&x);

        // RMS should be close to 1.0 after normalization
        let rms: f32 = (y.data.iter().map(|v| v * v).sum::<f32>() / 3.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1);
    }

    #[test]
    fn gqa_repeat_kv_expands_correctly() {
        let config = GroupedQueryConfig::new(4, 2, 3); // 4 query heads, 2 KV heads
        assert_eq!(config.group_size(), 2);

        // Create a simple KV tensor: [seq_len=1, n_kv_heads=2, head_dim=3]
        let kv_data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let kv = Tensor::from_data(kv_data, &[1, 2, 3]);

        let repeated = config.repeat_kv(&kv);
        assert_eq!(repeated.shape, vec![1, 4, 3]);
        // KV head 0 is repeated to query heads 0,1
        // KV head 1 is repeated to query heads 2,3
        assert_eq!(&repeated.data[0..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&repeated.data[3..6], &[0.0, 1.0, 2.0]);
        assert_eq!(&repeated.data[6..9], &[3.0, 4.0, 5.0]);
        assert_eq!(&repeated.data[9..12], &[3.0, 4.0, 5.0]);
    }
}

use std::sync::Arc;

#[allow(unused_imports)]
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    weight: Vec<f32>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut weight = vec![0.0; vocab_size * embedding_dim];
        let scale = (1.0 / (embedding_dim as f32).sqrt()) * 0.1;

        for i in 0..weight.len() {
            weight[i] = (rand::random::<f32>() - 0.5) * 2.0 * scale;
        }

        Self {
            vocab_size,
            embedding_dim,
            weight,
        }
    }

    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let mut output = vec![0.0; seq_len * self.embedding_dim];

        for (i, &token_id) in token_ids.iter().enumerate() {
            let idx = (token_id as usize * self.embedding_dim)
                .min(self.weight.len() - self.embedding_dim);
            for j in 0..self.embedding_dim {
                output[i * self.embedding_dim + j] = self.weight[idx + j];
            }
        }

        output
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (1.0 / (in_features as f32).sqrt()) * 0.1;
        let mut weight = vec![0.0; in_features * out_features];
        let mut bias = vec![0.0; out_features];

        for i in 0..weight.len() {
            weight[i] = (rand::random::<f32>() - 0.5) * 2.0 * scale;
        }

        for i in 0..out_features {
            bias[i] = 0.0;
        }

        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch_size = input.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = self.bias[o];
                for i in 0..self.in_features {
                    sum += input[b * self.in_features + i] * self.weight[i * self.out_features + o];
                }
                output[b * self.out_features + o] = sum;
            }
        }

        output
    }
}

pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654;
    let c = 0.044714998453855515;
    let t = (sqrt_2_over_pi * (x + c * x * x * x)).tanh();
    x * 0.5 * (1.0 + t)
}

pub fn softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }

    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    let mut max_val = input[0];
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }

    let mut exp_vals: Vec<f32> = input
        .iter()
        .map(|&x| ((x - max_val) / temp).exp())
        .collect();
    let sum: f32 = exp_vals.iter().sum();

    if sum > 0.0 {
        for val in &mut exp_vals {
            *val /= sum;
        }
    }

    exp_vals
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    hidden_dim: usize,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_dim, intermediate_size),
            up_proj: Linear::new(hidden_dim, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_dim),
            hidden_dim,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let intermediate = self.gate_proj.forward(input);
        let up = self.up_proj.forward(input);

        let gated: Vec<f32> = intermediate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu(g) * u)
            .collect();
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        Self {
            q_proj: Linear::new(hidden_dim, hidden_dim),
            k_proj: Linear::new(hidden_dim, hidden_dim),
            v_proj: Linear::new(hidden_dim, hidden_dim),
            o_proj: Linear::new(hidden_dim, hidden_dim),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        _mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        let seq_len = q.len() / self.hidden_dim();
        let hidden = self.hidden_dim();
        let hd = self.head_dim;

        if seq_len == 0 {
            return q.clone();
        }

        let mut output = vec![0.0; q.len()];

        for h in 0..self.num_heads {
            for i in 0..seq_len.min(1) {
                let mut score_sum = 0.0f32;
                let mut weighted_sum = vec![0.0f32; hd.min(16)];

                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    let d_max = hd.min(16);
                    for d in 0..d_max {
                        let q_idx = i * hidden + h * hd + d;
                        let k_idx = j * hidden + h * hd + d;
                        if q_idx < q.len() && k_idx < k.len() {
                            score += q[q_idx] * k[k_idx];
                        }
                    }
                    if !score.is_nan() {
                        score_sum += score.exp();
                    }
                }

                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    let d_max = hd.min(16);
                    for d in 0..d_max {
                        let q_idx = i * hidden + h * hd + d;
                        let k_idx = j * hidden + h * hd + d;
                        if q_idx < q.len() && k_idx < k.len() {
                            score += q[q_idx] * k[k_idx];
                        }
                    }
                    if !score.is_nan() && score_sum > 0.0 {
                        let weight = score.exp() / score_sum;
                        let d_max = hd.min(16);
                        for d in 0..d_max {
                            let v_idx = j * hidden + h * hd + d;
                            if v_idx < v.len() {
                                weighted_sum[d] += weight * v[v_idx];
                            }
                        }
                    }
                }

                let d_max = hd.min(16);
                for d in 0..d_max {
                    output[i * hidden + h * hd + d] = weighted_sum[d];
                }
            }
        }

        self.o_proj.forward(&output)
    }

    fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    hidden_dim: usize,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            attention: Attention::new(hidden_dim, num_heads),
            feed_forward: FeedForward::new(hidden_dim, intermediate_size),
            hidden_dim,
        }
    }

    pub fn forward(&self, hidden_states: &[f32], mask: Option<&[f32]>) -> Vec<f32> {
        let attn_output = self
            .attention
            .forward(hidden_states, hidden_states, hidden_states, mask);

        let residual: Vec<f32> = hidden_states
            .iter()
            .zip(attn_output.iter())
            .map(|(&x, &y)| x + y * 0.1)
            .collect();

        let ffn_output = self.feed_forward.forward(&residual);

        residual
            .iter()
            .zip(ffn_output.iter())
            .map(|(&x, &y)| x + y * 0.1)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct LMHead {
    embedding: Arc<Embedding>,
    linear: Linear,
}

impl LMHead {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            embedding: Arc::new(Embedding::new(vocab_size, hidden_dim)),
            linear: Linear::new(hidden_dim, vocab_size),
        }
    }

    pub fn forward(&self, hidden_states: &[f32]) -> Vec<f32> {
        self.linear.forward(hidden_states)
    }

    pub fn embedding(&self) -> Arc<Embedding> {
        self.embedding.clone()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Model {
    embedding: Arc<Embedding>,
    layers: Vec<TransformerBlock>,
    lm_head: LMHead,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
}

impl Model {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        let intermediate_size = hidden_dim * 4;
        let embedding = Arc::new(Embedding::new(vocab_size, hidden_dim));

        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(hidden_dim, num_heads, intermediate_size))
            .collect();

        Self {
            embedding,
            layers,
            lm_head: LMHead::new(vocab_size, hidden_dim),
            hidden_dim,
            num_layers,
            vocab_size,
        }
    }

    pub fn forward(&self, input_ids: &[u32]) -> Vec<f32> {
        let hidden = self.embedding.forward(input_ids);

        let seq_len = input_ids.len().max(1);
        let mask = self.create_causal_mask(seq_len);

        let mut hidden = hidden;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask));
        }

        let last_idx = (seq_len - 1) * self.hidden_dim;
        let last_hidden: Vec<f32> = if last_idx < hidden.len() {
            hidden[last_idx..last_idx + self.hidden_dim].to_vec()
        } else {
            vec![0.0; self.hidden_dim]
        };

        self.lm_head.forward(&last_hidden)
    }

    fn create_causal_mask(&self, seq_len: usize) -> Vec<f32> {
        let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    mask[i * seq_len + j] = 0.0;
                }
            }
        }

        mask
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub model: Model,
}

impl ModelWeights {
    pub fn new_small() -> Self {
        Self {
            model: Model::new(512, 256, 4, 4),
        }
    }

    pub fn new_medium() -> Self {
        Self {
            model: Model::new(1024, 512, 8, 8),
        }
    }

    pub fn new_large() -> Self {
        Self {
            model: Model::new(2048, 768, 12, 12),
        }
    }
}

pub fn create_model(config: &str) -> ModelWeights {
    match config {
        "small" => ModelWeights::new_small(),
        "medium" => ModelWeights::new_medium(),
        "large" => ModelWeights::new_large(),
        _ => ModelWeights::new_small(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 64);
        let tokens = vec![1, 2, 3];
        let output = emb.forward(&tokens);
        assert_eq!(output.len(), 3 * 64);
    }

    #[test]
    fn test_gelu() {
        assert!(gelu(1.0) > 0.0);
    }

    #[test]
    fn test_model() {
        let model = Model::new(128, 64, 2, 2);
        let tokens = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 128);
    }
}

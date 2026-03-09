use std::sync::Arc;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::error::InferenceResult;
use crate::model::Model;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    TopK,
    TopP,
    Temperature,
}

#[derive(Debug, Clone)]
pub struct Sampler {
    method: SamplingMethod,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            method: SamplingMethod::Greedy,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            rng: StdRng::from_entropy(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_method(mut self, method: SamplingMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        match self.method {
            SamplingMethod::Greedy => self.sample_greedy(logits),
            SamplingMethod::TopK => self.sample_top_k(logits),
            SamplingMethod::TopP => self.sample_top_p(logits),
            SamplingMethod::Temperature => self.sample_temperature(logits),
        }
    }

    fn sample_greedy(&mut self, logits: &[f32]) -> u32 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            if logit > max_val {
                max_val = logit;
                max_idx = i;
            }
        }

        max_idx as u32
    }

    fn sample_temperature(&mut self, logits: &[f32]) -> u32 {
        let temp = if self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };

        let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
        self.sample_from_probs(&scaled) as u32
    }

    fn sample_top_k(&mut self, logits: &[f32]) -> u32 {
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.top_k.min(indexed.len());
        let top_k_logits: Vec<f32> = indexed.iter().take(k).map(|(_, l)| *l).collect();

        let sampled = self.sample_from_probs(&top_k_logits);
        indexed[sampled].0 as u32
    }

    fn sample_top_p(&mut self, logits: &[f32]) -> u32 {
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = sorted.len();

        for (i, &logit) in sorted.iter().enumerate() {
            let prob = logit.exp();
            cumsum += prob;
            if cumsum >= self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let top_p_logits: Vec<f32> = sorted.into_iter().take(cutoff_idx).collect();
        self.sample_from_probs(&top_p_logits) as u32
    }

    fn sample_from_probs(&self, logits: &[f32]) -> usize {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();

        let sum: f32 = exp_logits.iter().sum();
        if sum == 0.0 || sum.is_nan() {
            return 0;
        }

        let mut probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

        let r = rand::random::<f32>();
        let mut cumulative = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                return i;
            }
        }

        probs.len() - 1
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_length: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub stop_token_id: u32,
    pub seed: Option<u64>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            stop_token_id: 2,
            seed: None,
        }
    }
}

pub struct Generator {
    model: Model,
    tokenizer: Arc<Tokenizer>,
}

impl Generator {
    pub fn new(model: Model, tokenizer: Arc<Tokenizer>) -> Self {
        Self { model, tokenizer }
    }

    pub fn generate(&self, prompt: &str, options: &GenerateOptions) -> InferenceResult<String> {
        let input_ids = self.tokenizer.encode(prompt, true, false);

        let mut sampler = Sampler::new()
            .with_temperature(options.temperature)
            .with_top_k(options.top_k)
            .with_top_p(options.top_p);

        if let Some(seed) = options.seed {
            sampler = sampler.with_seed(seed);
        }

        let mut generated_ids = input_ids.clone();
        let mut current_position = 0;

        while current_position < options.max_length {
            let logits = self.model.forward(&generated_ids);

            let next_token = sampler.sample(&logits);

            if next_token == options.stop_token_id {
                break;
            }

            generated_ids.push(next_token);
            current_position += 1;
        }

        let output = self.tokenizer.decode(&generated_ids);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let mut sampler = Sampler::new().with_method(SamplingMethod::Greedy);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sampler.sample(&logits);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_temperature() {
        let mut sampler = Sampler::new().with_temperature(0.1);
        let logits = vec![1.0, 2.0, 3.0];
        let result = sampler.sample(&logits);
        assert!(result < 3);
    }

    #[test]
    fn test_top_k() {
        let mut sampler = Sampler::new().with_top_k(2);
        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let result = sampler.sample(&logits);
        assert!(result == 1 || result == 3 || result == 4);
    }
}

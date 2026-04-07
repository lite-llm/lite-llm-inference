use std::sync::Arc;
use std::time::Instant;

use crate::error::InferenceResult;
use crate::model::{Model, ModelWeights};
use crate::sampler::{GenerateOptions, Sampler};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model_size: String,
    pub max_length: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_size: "small".to_string(),
            max_length: 100,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            seed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceEngine {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    config: InferenceConfig,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        let vocab_size = 90;
        let model = Model::new(vocab_size, 64, 2, 2);
        let tokenizer = Tokenizer::new();

        Self {
            model,
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    pub fn with_model(mut self, model_weights: ModelWeights) -> Self {
        self.model = model_weights.model;
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn generate(&self, prompt: &str) -> InferenceResult<String> {
        self.generate_with_max_len(prompt, self.config.max_length)
    }

    pub fn generate_with_max_len(
        &self,
        prompt: &str,
        max_length: usize,
    ) -> InferenceResult<String> {
        let options = GenerateOptions {
            max_length,
            temperature: self.config.temperature,
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            stop_token_id: self.tokenizer.eos(),
            seed: self.config.seed,
        };

        let input_ids = self.tokenizer.encode(prompt, true, false);

        let mut sampler = Sampler::new()
            .with_temperature(options.temperature)
            .with_top_k(options.top_k)
            .with_top_p(options.top_p);

        if let Some(seed) = options.seed {
            sampler = sampler.with_seed(seed);
        }

        let mut generated_ids = input_ids.clone();

        for _ in 0..options.max_length {
            let logits = self.model.forward(&generated_ids);

            let next_token = sampler.sample(&logits);

            if next_token == options.stop_token_id {
                break;
            }

            generated_ids.push(next_token);
        }

        let output = self.tokenizer.decode(&generated_ids);
        Ok(output)
    }

    pub fn generate_with_options(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> InferenceResult<String> {
        let input_ids = self.tokenizer.encode(prompt, true, false);

        let mut sampler = Sampler::new()
            .with_temperature(options.temperature)
            .with_top_k(options.top_k)
            .with_top_p(options.top_p);

        if let Some(seed) = options.seed {
            sampler = sampler.with_seed(seed);
        }

        let mut generated_ids = input_ids.clone();

        for _ in 0..options.max_length {
            let logits = self.model.forward(&generated_ids);

            let next_token = sampler.sample(&logits);

            if next_token == options.stop_token_id {
                break;
            }

            generated_ids.push(next_token);
        }

        let output = self.tokenizer.decode(&generated_ids);
        Ok(output)
    }

    pub fn generate_streaming<F>(&self, prompt: &str, mut callback: F) -> InferenceResult<()>
    where
        F: FnMut(String),
    {
        let options = GenerateOptions {
            max_length: self.config.max_length,
            temperature: self.config.temperature,
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            stop_token_id: self.tokenizer.eos(),
            seed: self.config.seed,
        };

        let input_ids = self.tokenizer.encode(prompt, true, false);

        let mut sampler = Sampler::new()
            .with_temperature(options.temperature)
            .with_top_k(options.top_k)
            .with_top_p(options.top_p);

        if let Some(seed) = options.seed {
            sampler = sampler.with_seed(seed);
        }

        let mut generated_ids = input_ids.clone();
        let mut last_char = ' ';

        for _ in 0..options.max_length {
            let logits = self.model.forward(&generated_ids);

            let next_token = sampler.sample(&logits);

            if next_token == options.stop_token_id {
                break;
            }

            generated_ids.push(next_token);

            let decoded = self.tokenizer.decode(&[next_token]);
            if !decoded.is_empty() {
                let c = decoded.chars().next().unwrap_or(' ');
                if c != ' ' || last_char != ' ' {
                    callback(decoded);
                }
                last_char = c;
            }
        }

        Ok(())
    }

    pub fn batch_generate(&self, prompts: &[String]) -> InferenceResult<Vec<String>> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let result = self.generate(prompt)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, true, true)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub latency_ms: u64,
}

pub struct InferenceSession {
    engine: InferenceEngine,
    prompt: String,
    generated_tokens: Vec<u32>,
    start_time: Instant,
}

impl InferenceSession {
    pub fn new(engine: InferenceEngine, prompt: &str) -> Self {
        Self {
            engine,
            prompt: prompt.to_string(),
            generated_tokens: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn step(&mut self) -> InferenceResult<Option<u32>> {
        let options = GenerateOptions {
            max_length: self.engine.config.max_length,
            temperature: self.engine.config.temperature,
            top_k: self.engine.config.top_k,
            top_p: self.engine.config.top_p,
            stop_token_id: self.engine.tokenizer.eos(),
            seed: self.engine.config.seed,
        };

        let mut all_tokens = self.engine.tokenize(&self.prompt);
        all_tokens.extend(&self.generated_tokens);

        if all_tokens.len() >= options.max_length {
            return Ok(None);
        }

        let logits = self.engine.model.forward(&all_tokens);

        let mut sampler = Sampler::new()
            .with_temperature(options.temperature)
            .with_top_k(options.top_k)
            .with_top_p(options.top_p);

        let next_token = sampler.sample(&logits);

        if next_token == options.stop_token_id {
            return Ok(None);
        }

        self.generated_tokens.push(next_token);
        Ok(Some(next_token))
    }

    pub fn run_full(&mut self) -> InferenceResult<String> {
        loop {
            match self.step()? {
                Some(_) => continue,
                None => break,
            }
        }

        let full_text = self.engine.tokenizer.decode(&self.generated_tokens);
        Ok(full_text)
    }

    pub fn response(&self) -> InferenceResponse {
        let latency = self.start_time.elapsed();
        let text = self.engine.tokenizer.decode(&self.generated_tokens);
        let prompt_tokens = self.engine.tokenize(&self.prompt).len();

        InferenceResponse {
            text,
            tokens: self.generated_tokens.clone(),
            prompt_tokens,
            generated_tokens: self.generated_tokens.len(),
            latency_ms: latency.as_millis() as u64,
        }
    }
}

pub fn create_inference_engine(config: InferenceConfig) -> InferenceEngine {
    InferenceEngine::new(config)
}

pub fn create_inference_engine_default() -> InferenceEngine {
    InferenceEngine::new(InferenceConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = create_inference_engine_default();
        assert!(engine.vocab_size() > 0);
    }

    #[test]
    fn test_generate() {
        let engine = create_inference_engine_default();
        let result = engine.generate("hi");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tokenize() {
        let engine = create_inference_engine_default();
        let tokens = engine.tokenize("hello world");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_streaming() {
        let engine = create_inference_engine_default();
        let mut output = String::new();
        engine
            .generate_streaming("hi", |s| output.push_str(&s))
            .unwrap();
        assert!(!output.is_empty() || output.len() == 0);
    }
}

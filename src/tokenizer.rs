use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    unk_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    model_max_length: usize,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<pad>".to_string(), 3);

        reverse_vocab.insert(0, "<unk>".to_string());
        reverse_vocab.insert(1, "<s>".to_string());
        reverse_vocab.insert(2, "</s>".to_string());
        reverse_vocab.insert(3, "<pad>".to_string());

        let chars =
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()[]{}";
        for c in chars.chars() {
            let token = c.to_string();
            if !vocab.contains_key(&token) {
                let id = vocab.len() as u32;
                vocab.insert(token.clone(), id);
                reverse_vocab.insert(id, token);
            }
        }

        let unk = vocab["<unk>"];
        let bos = vocab["<s>"];
        let eos = vocab["</s>"];
        let pad = vocab["<pad>"];

        Self {
            vocab,
            reverse_vocab,
            unk_token_id: unk,
            bos_token_id: bos,
            eos_token_id: eos,
            pad_token_id: pad,
            model_max_length: 512,
        }
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.bos_token_id);
        }

        for c in text.chars() {
            let token = c.to_string();
            if let Some(&id) = self.vocab.get(&token) {
                tokens.push(id);
            } else {
                tokens.push(self.unk_token_id);
            }
        }

        if add_eos {
            tokens.push(self.eos_token_id);
        }

        tokens.truncate(self.model_max_length);
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();

        for &token_id in tokens {
            if token_id == self.eos_token_id {
                break;
            }
            if token_id == self.pad_token_id || token_id == self.bos_token_id {
                continue;
            }
            if let Some(text) = self.reverse_vocab.get(&token_id) {
                result.push_str(text);
            }
        }

        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn eos(&self) -> u32 {
        self.eos_token_id
    }

    pub fn pad(&self) -> u32 {
        self.pad_token_id
    }

    pub fn bos(&self) -> u32 {
        self.bos_token_id
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_tokenizer() -> Arc<Tokenizer> {
    Arc::new(Tokenizer::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = Tokenizer::new();
        let text = "hello world";
        let tokens = tokenizer.encode(text, true, true);
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.vocab_size() > 50);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.eos(), 2);
        assert_eq!(tokenizer.bos(), 1);
    }
}

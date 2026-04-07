use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    pub version: u32,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

impl ModelCheckpoint {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            version: 1,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        let magic: &[u8] = b"LITE_LLM_MODEL_V1";
        file.write_all(magic)?;
        file.write_all(&self.version.to_le_bytes())?;
        file.write_all(&self.vocab_size.to_le_bytes())?;
        file.write_all(&self.hidden_dim.to_le_bytes())?;
        file.write_all(&self.num_layers.to_le_bytes())?;
        let num_heads_bytes = (self.num_heads as u64).to_le_bytes();
        file.write_all(&num_heads_bytes)?;

        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 17];
        file.read_exact(&mut magic)?;
        let expected: &[u8] = b"LITE_LLM_MODEL_V1";
        if &magic != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid model file format",
            ));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        let mut vocab_bytes = [0u8; 8];
        file.read_exact(&mut vocab_bytes)?;
        let vocab_size = usize::from_le_bytes(vocab_bytes);

        let mut hidden_bytes = [0u8; 8];
        file.read_exact(&mut hidden_bytes)?;
        let hidden_dim = usize::from_le_bytes(hidden_bytes);

        let mut layer_bytes = [0u8; 8];
        file.read_exact(&mut layer_bytes)?;
        let num_layers = usize::from_le_bytes(layer_bytes);

        let mut head_bytes = [0u8; 8];
        file.read_exact(&mut head_bytes)?;
        let num_heads = usize::from_le_bytes(head_bytes);

        Ok(Self {
            version,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerCheckpoint {
    pub unk_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl TokenizerCheckpoint {
    pub fn new(unk: u32, bos: u32, eos: u32, pad: u32) -> Self {
        Self {
            unk_token_id: unk,
            bos_token_id: bos,
            eos_token_id: eos,
            pad_token_id: pad,
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let magic: &[u8] = b"LITE_LLM_TOKENIZER";
        file.write_all(magic)?;
        file.write_all(&self.unk_token_id.to_le_bytes())?;
        file.write_all(&self.bos_token_id.to_le_bytes())?;
        file.write_all(&self.eos_token_id.to_le_bytes())?;
        file.write_all(&self.pad_token_id.to_le_bytes())?;
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 17];
        file.read_exact(&mut magic)?;
        let expected: &[u8] = b"LITE_LLM_TOKENIZER";
        if &magic != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid tokenizer file format",
            ));
        }

        let mut unk_bytes = [0u8; 4];
        file.read_exact(&mut unk_bytes)?;
        let unk_token_id = u32::from_le_bytes(unk_bytes);

        let mut bos_bytes = [0u8; 4];
        file.read_exact(&mut bos_bytes)?;
        let bos_token_id = u32::from_le_bytes(bos_bytes);

        let mut eos_bytes = [0u8; 4];
        file.read_exact(&mut eos_bytes)?;
        let eos_token_id = u32::from_le_bytes(eos_bytes);

        let mut pad_bytes = [0u8; 4];
        file.read_exact(&mut pad_bytes)?;
        let pad_token_id = u32::from_le_bytes(pad_bytes);

        Ok(Self {
            unk_token_id,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }
}

pub fn save_checkpoint(
    dir: &Path,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;

    let model_checkpoint = ModelCheckpoint::new(vocab_size, hidden_dim, num_layers, num_heads);
    model_checkpoint.save(&dir.join("model.bin"))?;

    let tokenizer_checkpoint = TokenizerCheckpoint::new(0, 1, 2, 3);
    tokenizer_checkpoint.save(&dir.join("tokenizer.bin"))?;

    Ok(())
}

pub fn load_checkpoint(dir: &Path) -> std::io::Result<(usize, usize, usize, usize)> {
    let checkpoint = ModelCheckpoint::load(&dir.join("model.bin"))?;
    Ok((
        checkpoint.vocab_size,
        checkpoint.hidden_dim,
        checkpoint.num_layers,
        checkpoint.num_heads as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_save_load_checkpoint() {
        let temp_dir = temp_dir().join("lite_llm_test");

        save_checkpoint(&temp_dir, 90, 64, 2, 2).unwrap();

        let (vocab, hidden, _layers, _heads) = load_checkpoint(&temp_dir).unwrap();
        assert_eq!(vocab, 90);
        assert_eq!(hidden, 64);
        assert_eq!(hidden, 64);

        std::fs::remove_dir_all(temp_dir).ok();
    }
}

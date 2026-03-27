//! Native GGUF tokenizer for Llama-family BPE vocabularies.

use std::collections::HashMap;
use std::sync::LazyLock;

use fancy_regex::Regex;
use thiserror::Error;

use crate::formats::{GgufMetadata, MetadataValue};

static LLAMA3_SPLIT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    )
    .expect("valid llama3 split regex")
});

static BYTE_TO_UNICODE: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut bs = ['\0'; 256];
    let mut mapped = [false; 256];
    for b in 33u8..=126u8 {
        bs[b as usize] = b as char;
        mapped[b as usize] = true;
    }
    for b in 161u8..=172u8 {
        bs[b as usize] = b as char;
        mapped[b as usize] = true;
    }
    for b in 174u8..=255u8 {
        bs[b as usize] = b as char;
        mapped[b as usize] = true;
    }

    let mut n = 0;
    for b in 0..=255 {
        if !mapped[b as usize] {
            bs[b as usize] = std::char::from_u32(256 + n).unwrap();
            mapped[b as usize] = true;
            n += 1;
        }
    }
    bs
});

static UNICODE_TO_BYTE: LazyLock<HashMap<char, u8>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(256);
    for (b, &c) in BYTE_TO_UNICODE.iter().enumerate() {
        map.insert(c, b as u8);
    }
    map
});

/// Tokenizer errors
#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer: {0}")]
    LoadFailed(String),

    #[error("Failed to encode text: {0}")]
    EncodeFailed(String),

    #[error("Failed to decode tokens: {0}")]
    DecodeFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid GGUF tokenizer metadata: {0}")]
    InvalidGgufMetadata(String),
}

/// Result type for tokenizer operations
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Native GGUF tokenizer.
pub struct GgufTokenizer {
    /// Raw token bytes -> token id
    vocab: HashMap<Vec<u8>, u32>,
    /// token id -> raw token bytes
    id_to_token: Vec<Vec<u8>>,
    /// Merge score map for merged token bytes
    scores: HashMap<Vec<u8>, f32>,
}

impl GgufTokenizer {
    const LLAMA3_BOS_TOKEN_ID: u32 = 128_000;
    const LLAMA3_EOT_TOKEN_ID: u32 = 128_009;

    /// Loading from tokenizer.json is no longer supported.
    pub fn from_file<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(TokenizerError::LoadFailed(
            "tokenizer.json loading removed; use GGUF metadata tokenizer".to_string(),
        ))
    }

    /// Build tokenizer from GGUF embedded metadata.
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self> {
        let tokens = Self::metadata_token_bytes_array(metadata, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(TokenizerError::InvalidGgufMetadata(
                "tokenizer.ggml.tokens is empty".to_string(),
            ));
        }

        let mut scores =
            Self::metadata_f32_array(metadata, "tokenizer.ggml.scores").unwrap_or_default();
        if scores.len() < tokens.len() {
            scores.resize(tokens.len(), 0.0);
        } else if scores.len() > tokens.len() {
            scores.truncate(tokens.len());
        }

        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut id_to_token = Vec::with_capacity(tokens.len());
        let mut score_map = HashMap::with_capacity(tokens.len());

        for (id, tok) in tokens.into_iter().enumerate() {
            let token_id = id as u32;
            vocab.insert(tok.clone(), token_id);
            score_map.insert(tok.clone(), scores[id]);
            id_to_token.push(tok);
        }

        tracing::info!(
            "Loaded native GGUF tokenizer with vocabulary size: {}",
            id_to_token.len()
        );

        Ok(Self {
            vocab,
            id_to_token,
            scores: score_map,
        })
    }

    /// Encode text into token IDs.
    ///
    /// Uses Llama 3 pre-tokenization regex followed by byte-level BPE merges.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut out = Vec::new();

        if add_special_tokens {
            out.push(Self::LLAMA3_BOS_TOKEN_ID);
        }

        for mat in LLAMA3_SPLIT_RE.find_iter(text) {
            let chunk = mat
                .map_err(|e| TokenizerError::EncodeFailed(format!("regex match failure: {e}")))?
                .as_str();
            self.encode_chunk(chunk.as_bytes(), &mut out)?;
        }

        Ok(out)
    }

    fn encode_chunk(&self, bytes: &[u8], out: &mut Vec<u32>) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }

        let mut parts: Vec<Vec<u8>> = bytes
            .iter()
            .map(|&b| {
                let c = BYTE_TO_UNICODE[b as usize];
                let mut buf = [0; 4];
                c.encode_utf8(&mut buf).as_bytes().to_vec()
            })
            .collect();

        loop {
            if parts.len() < 2 {
                break;
            }

            let mut best_idx = None;
            let mut best_score = f32::NEG_INFINITY;

            for i in 0..(parts.len() - 1) {
                let mut merged = Vec::with_capacity(parts[i].len() + parts[i + 1].len());
                merged.extend_from_slice(&parts[i]);
                merged.extend_from_slice(&parts[i + 1]);

                if self.vocab.contains_key(&merged) {
                    let score = self.scores.get(&merged).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(i);
                    }
                }
            }

            let Some(i) = best_idx else {
                break;
            };

            let mut merged = Vec::with_capacity(parts[i].len() + parts[i + 1].len());
            merged.extend_from_slice(&parts[i]);
            merged.extend_from_slice(&parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }

        for part in parts {
            if let Some(id) = self.vocab.get(&part) {
                out.push(*id);
            } else {
                return Err(TokenizerError::EncodeFailed(format!(
                    "missing byte/token in vocab during BPE encode: {:?}",
                    part
                )));
            }
        }

        Ok(())
    }

    /// Decode a single token ID into text.
    pub fn decode(&self, token_id: u32) -> Result<String> {
        self.decode_batch(&[token_id])
    }

    /// Decode a sequence of token IDs into text.
    pub fn decode_batch(&self, token_ids: &[u32]) -> Result<String> {
        let mut raw_bytes = Vec::new();
        for &id in token_ids {
            if id == Self::LLAMA3_BOS_TOKEN_ID || id == Self::LLAMA3_EOT_TOKEN_ID {
                continue;
            }

            let mapped_tok = self.id_to_token.get(id as usize).ok_or_else(|| {
                TokenizerError::DecodeFailed(format!("token id out of range: {}", id))
            })?;

            if let Ok(s) = std::str::from_utf8(mapped_tok) {
                for c in s.chars() {
                    if let Some(&b) = UNICODE_TO_BYTE.get(&c) {
                        raw_bytes.push(b);
                    }
                }
            }
        }

        String::from_utf8(raw_bytes).map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Llama 3 BOS token.
    pub const fn bos_token_id(&self) -> Option<u32> {
        Some(Self::LLAMA3_BOS_TOKEN_ID)
    }

    /// Llama 3 EOT token.
    pub const fn eos_token_id(&self) -> Option<u32> {
        Some(Self::LLAMA3_EOT_TOKEN_ID)
    }

    /// No pad token in current native GGUF tokenizer path.
    pub const fn pad_token_id(&self) -> Option<u32> {
        None
    }

    /// Convert token ID to string (lossy UTF-8 for debugging).
    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.id_to_token
            .get(token_id as usize)
            .and_then(|b| String::from_utf8(b.clone()).ok())
    }

    fn metadata_array<'a>(metadata: &'a GgufMetadata, key: &str) -> Result<&'a [MetadataValue]> {
        match metadata.metadata.get(key) {
            Some(MetadataValue::Array(values)) => Ok(values),
            Some(_) => Err(TokenizerError::InvalidGgufMetadata(format!(
                "metadata key '{}' exists but is not an array",
                key
            ))),
            None => Err(TokenizerError::InvalidGgufMetadata(format!(
                "missing required metadata key '{}'",
                key
            ))),
        }
    }

    fn metadata_token_bytes_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<Vec<u8>>> {
        Self::metadata_array(metadata, key)?
            .iter()
            .map(|value| match value {
                MetadataValue::Bytes(bytes) => Ok(bytes.clone()),
                _ => Err(TokenizerError::InvalidGgufMetadata(format!(
                    "metadata key '{}' contains a non-string token entry",
                    key
                ))),
            })
            .collect()
    }

    fn metadata_f32_array(metadata: &GgufMetadata, key: &str) -> Result<Vec<f32>> {
        Self::metadata_array(metadata, key)?
            .iter()
            .map(|value| match value {
                MetadataValue::Float32(v) => Ok(*v),
                MetadataValue::Float64(v) => Ok(*v as f32),
                MetadataValue::Int8(v) => Ok(f32::from(*v)),
                MetadataValue::Int16(v) => Ok(f32::from(*v)),
                MetadataValue::Int32(v) => Ok(*v as f32),
                MetadataValue::Int64(v) => Ok(*v as f32),
                MetadataValue::UInt8(v) => Ok(f32::from(*v)),
                MetadataValue::UInt16(v) => Ok(f32::from(*v)),
                MetadataValue::UInt32(v) => Ok(*v as f32),
                MetadataValue::UInt64(v) => Ok(*v as f32),
                _ => Err(TokenizerError::InvalidGgufMetadata(format!(
                    "metadata key '{}' contains a non-numeric score entry",
                    key
                ))),
            })
            .collect()
    }
}

/// Backward-compatible alias used throughout the engine.
pub type Tokenizer = GgufTokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_error_display() {
        let err = TokenizerError::LoadFailed("test error".to_string());
        assert_eq!(err.to_string(), "Failed to load tokenizer: test error");
    }

    #[test]
    fn test_llama3_regex_compiles() {
        let chunks: Vec<String> = LLAMA3_SPLIT_RE
            .find_iter("Hello, world! 123")
            .map(|m| m.expect("regex match").as_str().to_string())
            .collect();
        assert!(!chunks.is_empty());
    }
}

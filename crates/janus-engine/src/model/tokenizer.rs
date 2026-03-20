//! Tokenizer for encoding and decoding text
//!
//! This module provides a wrapper around the HuggingFace `tokenizers` library
//! for converting between text and token IDs.

use std::path::Path;
use thiserror::Error;

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
}

/// Result type for tokenizer operations
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Tokenizer for text encoding and decoding
///
/// This wraps the HuggingFace tokenizers library to provide:
/// - Text → Token IDs encoding
/// - Token IDs → Text decoding
/// - Vocabulary information
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    vocab_size: usize,
}

impl Tokenizer {
    /// Load a tokenizer from a tokenizer.json file
    ///
    /// The tokenizer.json file should be in HuggingFace tokenizers format,
    /// typically found alongside model weights (e.g., from LLaMA, Mistral).
    ///
    /// # Arguments
    /// * `path` - Path to the tokenizer.json file
    ///
    /// # Returns
    /// A new Tokenizer instance
    ///
    /// # Example
    /// ```no_run
    /// use janus_engine::model::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_file("models/llama-7b/tokenizer.json")
    ///     .expect("Failed to load tokenizer");
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;

        let vocab_size = tokenizer.get_vocab_size(true);

        tracing::info!("Loaded tokenizer with vocabulary size: {}", vocab_size);

        Ok(Self {
            tokenizer,
            vocab_size,
        })
    }

    /// Encode text into token IDs
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_special_tokens` - Whether to add special tokens (BOS, EOS, etc.)
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::model::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    /// let tokens = tokenizer.encode("Hello, world!", true).unwrap();
    /// println!("Tokens: {:?}", tokens);
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a single token ID into text
    ///
    /// # Arguments
    /// * `token_id` - The token ID to decode
    ///
    /// # Returns
    /// The decoded text string
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::model::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    /// let text = tokenizer.decode(42).unwrap();
    /// println!("Token 42: {}", text);
    /// ```
    pub fn decode(&self, token_id: u32) -> Result<String> {
        self.decode_batch(&[token_id])
    }

    /// Decode a sequence of token IDs into text
    ///
    /// # Arguments
    /// * `token_ids` - Slice of token IDs to decode
    ///
    /// # Returns
    /// The decoded text string
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::model::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    /// let tokens = vec![1, 15043, 29892, 3186, 29991]; // "Hello, world!"
    /// let text = tokenizer.decode_batch(&tokens).unwrap();
    /// println!("Text: {}", text);
    /// ```
    pub fn decode_batch(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    /// Get the vocabulary size
    ///
    /// # Returns
    /// Number of tokens in the vocabulary
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the token ID for the beginning-of-sequence token
    ///
    /// # Returns
    /// BOS token ID, if it exists
    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<s>")
            .or_else(|| self.tokenizer.token_to_id("[BOS]"))
            .or_else(|| self.tokenizer.token_to_id("<bos>"))
    }

    /// Get the token ID for the end-of-sequence token
    ///
    /// # Returns
    /// EOS token ID, if it exists
    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("[EOS]"))
            .or_else(|| self.tokenizer.token_to_id("<eos>"))
    }

    /// Get the token ID for the padding token
    ///
    /// # Returns
    /// PAD token ID, if it exists
    pub fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<pad>")
            .or_else(|| self.tokenizer.token_to_id("[PAD]"))
    }

    /// Convert a token ID to its string representation
    ///
    /// This is useful for debugging or inspecting individual tokens.
    ///
    /// # Arguments
    /// * `token_id` - The token ID to convert
    ///
    /// # Returns
    /// The token string, if it exists in the vocabulary
    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.tokenizer.id_to_token(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_error_display() {
        let err = TokenizerError::LoadFailed("test error".to_string());
        assert_eq!(err.to_string(), "Failed to load tokenizer: test error");
    }
}

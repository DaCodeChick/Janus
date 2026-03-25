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
    const LLAMA_BOS_TOKEN_ID: u32 = 1;
    const LLAMA_DUMMY_PREFIX_SPACE_ID: u32 = 29_871;
    const MAX_SPECIAL_TOKEN_LITERAL_CHARS: usize = 256;

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
        // Fast path: no embedded special tokens in the input text.
        if !self.contains_embedded_special_tokens(text) {
            let encoding = self
                .tokenizer
                .encode(text, add_special_tokens)
                .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
            return Ok(self.ensure_bos_anchor(encoding.get_ids().to_vec(), add_special_tokens));
        }

        let mut token_ids = Vec::new();
        let mut segment_start = 0;
        let mut cursor = 0;

        while cursor < text.len() {
            if let Some((special_id, special_len)) = self.special_token_match(&text[cursor..]) {
                if segment_start < cursor {
                    let chunk = &text[segment_start..cursor];
                    let encoding = self
                        .tokenizer
                        .encode(chunk, false)
                        .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
                    self.extend_chunk_ids(&mut token_ids, chunk, encoding.get_ids());
                }

                token_ids.push(special_id);
                cursor += special_len;
                segment_start = cursor;
                continue;
            }

            let next_char_len = text[cursor..].chars().next().map_or(1, char::len_utf8);
            cursor += next_char_len;
        }

        if segment_start < text.len() {
            let chunk = &text[segment_start..];
            let encoding = self
                .tokenizer
                .encode(chunk, false)
                .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
            self.extend_chunk_ids(&mut token_ids, chunk, encoding.get_ids());
        }

        if add_special_tokens {
            let bos_token_id = self.bos_token_id().unwrap_or(Self::LLAMA_BOS_TOKEN_ID);
            if token_ids.first() != Some(&bos_token_id) {
                token_ids.insert(0, bos_token_id);
            }
            if let Some(eos_token_id) = self.eos_token_id() {
                token_ids.push(eos_token_id);
            }
        }

        Ok(self.ensure_bos_anchor(token_ids, add_special_tokens))
    }

    fn ensure_bos_anchor(&self, mut token_ids: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        if add_special_tokens && token_ids.first() != Some(&Self::LLAMA_BOS_TOKEN_ID) {
            token_ids.insert(0, Self::LLAMA_BOS_TOKEN_ID);
        }
        token_ids
    }

    fn extend_chunk_ids(&self, token_ids: &mut Vec<u32>, chunk: &str, chunk_ids: &[u32]) {
        if chunk_ids.is_empty() {
            return;
        }

        let mut start = 0;
        if chunk_ids.first() == Some(&Self::LLAMA_DUMMY_PREFIX_SPACE_ID) && !chunk.starts_with(' ')
        {
            start = 1;
        }

        token_ids.extend_from_slice(&chunk_ids[start..]);
    }

    fn contains_embedded_special_tokens(&self, text: &str) -> bool {
        let mut cursor = 0;
        while cursor < text.len() {
            if self.special_token_match(&text[cursor..]).is_some() {
                return true;
            }

            let next_char_len = text[cursor..].chars().next().map_or(1, char::len_utf8);
            cursor += next_char_len;
        }

        false
    }

    fn special_token_match(&self, input: &str) -> Option<(u32, usize)> {
        let mut best_match: Option<(u32, usize)> = None;

        if let Some((candidate, len)) = Self::angle_bracket_pipe_candidate(input) {
            if let Some(id) = self.special_token_id(candidate) {
                best_match = Some((id, len));
            }
        }

        if let Some((candidate, len)) = Self::angle_bracket_candidate(input) {
            if let Some(id) = self.special_token_id(candidate) {
                match best_match {
                    Some((_, best_len)) if best_len >= len => {}
                    _ => best_match = Some((id, len)),
                }
            }
        }

        if let Some((candidate, len)) = Self::square_bracket_candidate(input) {
            if let Some(id) = self.special_token_id(candidate) {
                match best_match {
                    Some((_, best_len)) if best_len >= len => {}
                    _ => best_match = Some((id, len)),
                }
            }
        }

        best_match
    }

    fn special_token_id(&self, token: &str) -> Option<u32> {
        let id = self.tokenizer.token_to_id(token)?;
        if self.tokenizer.id_to_token(id).as_deref() == Some(token) {
            Some(id)
        } else {
            None
        }
    }

    fn angle_bracket_pipe_candidate(input: &str) -> Option<(&str, usize)> {
        if !input.starts_with("<|") {
            return None;
        }

        let end_rel = input[2..].find("|>")?;
        let len = 2 + end_rel + 2;
        if len > Self::MAX_SPECIAL_TOKEN_LITERAL_CHARS {
            return None;
        }

        Some((&input[..len], len))
    }

    fn angle_bracket_candidate(input: &str) -> Option<(&str, usize)> {
        if !input.starts_with('<') {
            return None;
        }

        let end_rel = input[1..].find('>')?;
        let len = 1 + end_rel + 1;
        if len > Self::MAX_SPECIAL_TOKEN_LITERAL_CHARS {
            return None;
        }

        Some((&input[..len], len))
    }

    fn square_bracket_candidate(input: &str) -> Option<(&str, usize)> {
        if !input.starts_with('[') {
            return None;
        }

        let end_rel = input[1..].find(']')?;
        let len = 1 + end_rel + 1;
        if len > Self::MAX_SPECIAL_TOKEN_LITERAL_CHARS {
            return None;
        }

        Some((&input[..len], len))
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

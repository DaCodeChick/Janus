//! Transformer model components
//!
//! This module contains the building blocks for transformer-based language models:
//! - `block`: Individual transformer block layers
//! - `config`: Configuration parsing from HuggingFace config.json
//! - `transformer`: Complete transformer model with autoregressive generation
//! - `output`: Language model head for vocabulary projection
//! - `sampler`: Token sampling strategies
//! - `tokenizer`: Text encoding and decoding
//! - `speculative`: Speculative decoding for accelerated inference

pub mod block;
pub mod config;
pub mod output;
pub mod sampler;
pub mod tokenizer;
pub mod speculative;

#[path = "transformer/mod.rs"]
pub mod model;

pub use block::{TransformerBlock, TransformerBlockConfig};
pub use config::{HuggingFaceConfig, ConfigError};
pub use model::{Model, ModelConfig};
pub use output::{LMHead, LMHeadConfig};
pub use sampler::{Sampler, SamplerConfig};
pub use tokenizer::{Tokenizer, TokenizerError};
pub use speculative::{SpeculativeDecoder, SpeculativeConfig, SpeculativeStats};

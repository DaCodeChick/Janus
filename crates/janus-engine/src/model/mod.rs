//! Transformer model components
//!
//! This module contains the building blocks for transformer-based language models:
//! - `block`: Individual transformer block layers

pub mod block;

pub use block::{TransformerBlock, TransformerBlockConfig};

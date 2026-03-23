//! KV Cache for efficient autoregressive generation
//!
//! The KV cache stores Key and Value projections from previous tokens to avoid
//! recomputing them during text generation. This is critical for LLM performance.
//!
//! # KV Cache Compression
//!
//! To extend context length beyond the physical cache size, this implementation
//! supports compression of older KV cache entries using a sliding window approach:
//!
//! - **Recent tokens** (uncompressed window): Keep full precision for accuracy
//! - **Middle range** (compression zone): Apply 2:1 compression via averaging
//! - **Old tokens** (eviction zone): Remove to free space
//!
//! This allows effective context lengths of 2-4x the base cache size with minimal
//! quality degradation.

mod config;
mod core;
mod compression;

#[cfg(test)]
mod tests;

pub use config::CacheCompressionConfig;
pub use core::KVCache;

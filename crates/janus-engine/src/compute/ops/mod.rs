//! GPU compute operations for tensor math
//!
//! This module provides GPU-accelerated operations for transformer-based language models:
//! - Matrix operations (matmul, GEMM)
//! - Quantized operations (Q4_K GEMM)
//! - Activation functions (SiLU)
//! - Normalization (RMSNorm)
//! - Positional embeddings (RoPE)
//! - Attention mechanisms (scaled dot-product attention)
//! - Element-wise operations (add, multiply)

mod activation;
mod attention;
mod elementwise;
mod matmul;
mod quantized;
mod rope;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_q4k;

// Re-export all public functions
pub use activation::{rmsnorm, silu};
pub use attention::{compute_attention, compute_attention_static};
pub use elementwise::{add_tensors, elementwise_mul};
pub use matmul::{gemm, matmul};
pub use quantized::{gemm_q4_k, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};
pub use rope::rope;

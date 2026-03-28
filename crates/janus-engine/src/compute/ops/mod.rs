//! GPU compute operations for tensor math
//!
//! This module provides GPU-accelerated operations for transformer-based language models:
//! - Matrix operations (matmul, GEMM)
//! - Quantized operations (Q4_K, Q5_K, Q8_0 GEMM)
//! - Activation functions (SiLU)
//! - Normalization (RMSNorm)
//! - Positional embeddings (RoPE)
//! - Attention mechanisms (scaled dot-product attention)
//! - Element-wise operations (add, multiply)
//! - Sampling operations (argmax)

mod activation;
mod argmax;
mod attention;
mod elementwise;
mod matmul;
mod quantized;
mod rope;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_q4k;

#[cfg(test)]
mod tests_q5k_q8_0;

// Re-export all public functions
pub use activation::{rmsnorm, silu};
pub use argmax::argmax;
pub use attention::compute_attention;
pub use elementwise::{add_tensors, elementwise_mul};
pub use matmul::{gemm, gemm_f32, matmul};
pub use quantized::{
    gemm_auto,
    gemm_q4_k, gemm_q5_k, gemm_q8_0,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE,
    Q5K_BLOCK_BYTES, Q5K_BLOCK_SIZE,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
};
pub use rope::rope;

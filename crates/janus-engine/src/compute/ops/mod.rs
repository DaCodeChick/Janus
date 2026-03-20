//! GPU compute operations for tensor math
//!
//! This module provides GPU-accelerated operations for transformer-based language models:
//! - Matrix operations (matmul, GEMM)
//! - Activation functions (SiLU)
//! - Normalization (RMSNorm)
//! - Positional embeddings (RoPE)
//! - Attention mechanisms (scaled dot-product attention)
//! - Element-wise operations (add, multiply)

mod activation;
mod attention;
mod elementwise;
mod matmul;
mod rope;

#[cfg(test)]
mod tests;

// Re-export all public functions
pub use activation::{rmsnorm, silu};
pub use attention::compute_attention;
pub use elementwise::{add_tensors, elementwise_mul};
pub use matmul::{gemm, matmul};
pub use rope::rope;

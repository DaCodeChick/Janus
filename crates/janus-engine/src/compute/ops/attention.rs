//! Attention mechanisms and softmax operations
//!
//! This module implements scaled dot-product attention for transformer models.
//! The attention computation follows three steps:
//! 1. Compute attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
//! 2. Apply softmax to get attention probabilities
//! 3. Multiply probabilities by values to get output

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for Attention operations
/// Must match the layout in attention.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionUniforms {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    layer_idx: u32,
    max_seq_len: u32,
    num_layers: u32,
    _pad: [u32; 7], // Padding to 64 bytes (WGSL uniform alignment requirement)
}

/// Uniforms structure for Softmax operation
/// Must match the layout in softmax.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    seq_len: u32,
    num_heads: u32,
    batch_size: u32,
    max_seq_len: u32,
}

/// Compute scaled dot-product attention with Grouped Query Attention (GQA) support - Batched
///
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
///
/// This function implements batched multi-head attention with GQA by:
/// 1. Computing attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
/// 2. Applying softmax to get attention probabilities
/// 3. Multiplying probabilities by values to get output
///
/// For GQA, each KV head is shared across multiple query heads (num_heads / num_kv_heads).
/// Each sequence in the batch attends only to its own history (no cross-sequence attention).
///
/// # Arguments
/// * `engine` - The compute engine providing GPU access
/// * `encoder` - Shared command encoder for batching operations
/// * `pipeline_cache` - Pre-compiled shader cache
/// * `query` - Query tensor [batch_size, num_heads, head_dim]
/// * `key_cache` - Batched key cache [batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
/// * `value_cache` - Batched value cache [batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
/// * `output` - Output tensor [batch_size, num_heads, head_dim]
/// * `scores` - Pre-allocated buffer for attention scores [batch_size, num_heads, max_seq_len]
/// * `probs` - Pre-allocated buffer for attention probabilities [batch_size, num_heads, max_seq_len]
/// * `batch_size` - Number of sequences being processed in parallel
/// * `layer_idx` - Layer index for cache segmentation
/// * `seq_len` - Current sequence length (number of tokens processed so far, including current)
/// * `max_seq_len` - Maximum sequence length for cache sizing
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads (GQA support)
/// * `head_dim` - Dimension of each attention head
/// * `num_layers` - Total number of transformer layers (for KV cache indexing)
///
/// # Shaders
/// Uses `shaders/attention.wgsl` and `shaders/softmax.wgsl` for compute operations.
pub fn compute_attention(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    query: &wgpu::Buffer,
    key_cache: &wgpu::Buffer,
    value_cache: &wgpu::Buffer,
    output: &wgpu::Buffer,
    scores: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    batch_size: u32,
    layer_idx: u32,
    seq_len: u32,
    max_seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_layers: u32,
) -> Result<()> {
    let device = engine.device();
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Create uniforms
    let attention_uniforms = AttentionUniforms {
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        layer_idx,
        max_seq_len,
        num_layers,
        _pad: [0; 7],
    };
    let attention_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attention_uniforms"),
        contents: bytemuck::cast_slice(&[attention_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let softmax_uniforms = SoftmaxUniforms {
        seq_len,
        num_heads,
        batch_size: batch_size * num_heads, // Total number of softmax operations
        max_seq_len,
    };
    let softmax_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("softmax_uniforms"),
        contents: bytemuck::cast_slice(&[softmax_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // === Step 1: QK scores ===
    let qk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qk_bind_group"),
        layout: &pipeline_cache.attention_qk_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: query.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: key_cache.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: attention_uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // === Step 2: Softmax ===
    let softmax_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softmax_bind_group"),
        layout: &pipeline_cache.attention_softmax_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: probs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: softmax_uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // === Step 3: Apply attention ===
    let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("apply_bind_group"),
        layout: &pipeline_cache.attention_apply_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: probs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: value_cache.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: attention_uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // Record all three passes
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qk_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_cache.attention_qk_pipeline);
        compute_pass.set_bind_group(0, &qk_bind_group, &[]);
        // Dispatch batch_size * num_heads workgroups (one per batch item per head)
        compute_pass.dispatch_workgroups(batch_size * num_heads, 1, 1);
    }
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_cache.attention_softmax_pipeline);
        compute_pass.set_bind_group(0, &softmax_bind_group, &[]);
        // Dispatch batch_size * num_heads workgroups (one softmax per batch item per head)
        compute_pass.dispatch_workgroups(batch_size * num_heads, 1, 1);
    }
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apply_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_cache.attention_apply_pipeline);
        compute_pass.set_bind_group(0, &apply_bind_group, &[]);
        // Total elements: batch_size * num_heads * head_dim
        let total_elements = batch_size * num_heads * head_dim;
        let workgroup_count = (total_elements + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    Ok(())
}

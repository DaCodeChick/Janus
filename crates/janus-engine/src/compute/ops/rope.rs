//! Rotary Positional Embeddings (RoPE)
//!
//! This module implements RoPE, which encodes positional information by rotating
//! pairs of dimensions in the embedding space. This allows the model to understand
//! token ordering without explicit position embeddings.

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for RoPE (Rotary Positional Embeddings)
/// Must match the layout in rope.wgsl
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeUniforms {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    position: u32,
}

const _: [(); 16] = [(); std::mem::size_of::<RopeUniforms>()];

/// Apply Rotary Positional Embeddings (RoPE)
///
/// RoPE encodes positional information by rotating pairs of dimensions in the embedding space.
/// This allows the model to understand token ordering without explicit position embeddings.
///
/// This optimized version uses pre-computed sin/cos values from a cache buffer,
/// eliminating expensive trigonometric computations during inference.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Shared command encoder for recording GPU operations
/// * `pipeline_cache` - Pre-compiled shader cache
/// * `input` - GPU buffer containing input tensor ([batch_size, num_heads, head_dim] elements)
/// * `output` - Pre-allocated output buffer (same size as input)
/// * `rope_cache` - Pre-computed sin/cos values [max_seq_len * head_dim]
/// * `batch_size` - Number of sequences in the batch
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `position` - Starting position in the sequence
///
/// # Note
/// This function records commands to the encoder but does NOT submit them.
/// The caller is responsible for submitting the encoder.
pub fn rope(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    rope_cache: &wgpu::Buffer,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    position: u32,
) -> Result<()> {
    let device = engine.device();

    let uniforms = RopeUniforms {
        batch_size,
        num_heads,
        head_dim,
        position,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_bind_group"),
        layout: &pipeline_cache.rope_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rope_cache.as_entire_binding(),
            },
        ],
    });

    // Record compute pass to encoder
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_cache.rope_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup count (256 threads per workgroup)
        // Process batch_size * num_heads * head_dim elements
        let total_elements = batch_size * num_heads * head_dim;
        let workgroup_count = (total_elements + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    Ok(())
}

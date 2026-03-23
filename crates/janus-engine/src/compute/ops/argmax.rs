//! GPU-side argmax operation for greedy sampling
//!
//! This module provides a GPU-accelerated argmax operation that finds the index
//! of the maximum value in logits without transferring the full logits array to CPU.
//! This significantly reduces GPU-to-CPU bandwidth usage during greedy decoding.

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for argmax operation
/// Must match the layout in argmax.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxUniforms {
    vocab_size: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Find the index of the maximum value in logits (argmax) on GPU
///
/// This operation performs greedy sampling by finding the token with the highest
/// logit value without transferring the full logits array to CPU. This avoids
/// ~128KB GPU→CPU transfer per token for typical vocabulary sizes.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Command encoder to record GPU commands
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `logits` - GPU buffer containing logits ([batch_size * vocab_size] f32 array)
/// * `output` - Output buffer for token IDs ([batch_size] u32 array)
/// * `vocab_size` - Size of the vocabulary
/// * `batch_size` - Number of sequences in the batch
///
/// # Returns
/// Returns Ok(()) on success, or an error if the operation fails.
///
/// # Performance
/// For a vocabulary of 32K tokens:
/// - Without GPU argmax: ~128KB transfer + CPU sampling
/// - With GPU argmax: Only 4 bytes per batch item transfer
///
/// # Shader
/// Uses the WGSL shader at `shaders/argmax.wgsl` for compute operations.
pub fn argmax(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    logits: &wgpu::Buffer,
    output: &wgpu::Buffer,
    vocab_size: u32,
    batch_size: u32,
) -> Result<()> {
    let device = engine.device();

    // Create uniforms buffer
    let uniforms = ArgmaxUniforms {
        vocab_size,
        batch_size,
        _pad0: 0,
        _pad1: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("argmax_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group using cached layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("argmax_bind_group"),
        layout: &pipeline_cache.argmax_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: logits.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    // We use 1 workgroup per batch item in the x dimension
    // The workgroup size is 256 threads (defined in shader)
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("argmax_pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline_cache.argmax_pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    compute_pass.dispatch_workgroups(1, batch_size, 1);
    drop(compute_pass);

    Ok(())
}

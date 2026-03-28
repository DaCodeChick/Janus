//! Activation and normalization operations
//!
//! This module provides GPU-accelerated activation functions and normalization:
//! - `silu`: SiLU (Swish) activation function
//! - `rmsnorm`: Root Mean Square Layer Normalization

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for SiLU activation
/// Must match the layout in activations.wgsl
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluUniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniforms structure for RMSNorm
/// Must match the layout in activations.wgsl
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormUniforms {
    batch_size: u32,
    hidden_dim: u32,
    epsilon: f32,
    _pad: u32,
}

const _: [(); 16] = [(); std::mem::size_of::<SiluUniforms>()];
const _: [(); 16] = [(); std::mem::size_of::<RmsNormUniforms>()];

/// Apply Root Mean Square Normalization (RMSNorm) with learned gamma weights
///
/// Formula: output[i] = (input[i] / sqrt(mean(input^2) + epsilon)) * gamma[i]
///
/// For batched processing: Each sequence in the batch is normalized independently.
/// Input/Output: [batch_size, hidden_dim]
/// Weights (gamma): [hidden_dim] (shared across batch)
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Command encoder to record GPU commands
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `input` - GPU buffer containing input values ([batch_size, hidden_dim] f32 array)
/// * `output` - Output buffer (pre-allocated, same size as input)
/// * `weights` - GPU buffer containing gamma weights ([hidden_dim] f32 array, shared across batch)
/// * `batch_size` - Number of sequences in the batch
/// * `hidden_dim` - Hidden dimension (size per sequence)
/// * `epsilon` - Small constant for numerical stability (typically 1e-6)
///
/// # Shader
/// Uses the WGSL shader at `shaders/activations.wgsl` for compute operations.
pub fn rmsnorm(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    weights: &wgpu::Buffer,
    batch_size: u32,
    hidden_dim: u32,
    epsilon: f32,
) -> Result<()> {
    let device = engine.device();

    // Create uniforms buffer
    let uniforms = RmsNormUniforms {
        batch_size,
        hidden_dim,
        epsilon,
        _pad: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsnorm_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group using cached layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmsnorm_bind_group"),
        layout: &pipeline_cache.rmsnorm_layout,
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
                resource: weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // Record compute pass using cached pipeline
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_cache.rmsnorm_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Each workgroup processes one sequence (batch_size workgroups)
        compute_pass.dispatch_workgroups(batch_size, 1, 1);
    }

    Ok(())
}

/// Apply SiLU (Swish) activation function: f(x) = x * sigmoid(x)
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Command encoder to record GPU commands
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `input` - GPU buffer containing input values (f32 array)
/// * `output` - Output buffer (pre-allocated)
/// * `size` - Number of elements in the input
///
/// # Shader
/// Uses the WGSL shader at `shaders/activations.wgsl` for compute operations.
pub fn silu(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    size: u32,
) -> Result<()> {
    let device = engine.device();

    // Create uniforms buffer
    let uniforms = SiluUniforms {
        size,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("silu_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group using cached layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu_bind_group"),
        layout: &pipeline_cache.silu_layout,
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
        ],
    });

    // Record compute pass using cached pipeline
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_cache.silu_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size (256 threads per workgroup)
        let workgroup_count = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    Ok(())
}

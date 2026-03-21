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
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeUniforms {
    seq_len: u32,
    head_dim: u32,
    position: u32,
    theta_base: f32,
}

/// Apply Rotary Positional Embeddings (RoPE)
///
/// RoPE encodes positional information by rotating pairs of dimensions in the embedding space.
/// This allows the model to understand token ordering without explicit position embeddings.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Shared command encoder for recording GPU operations
/// * `pipeline_cache` - Pre-compiled shader cache
/// * `input` - GPU buffer containing input tensor (seq_len * head_dim elements)
/// * `output` - Pre-allocated output buffer (same size as input)
/// * `seq_len` - Sequence length (number of tokens)
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `position` - Starting position in the sequence
/// * `theta_base` - Base for frequency calculation (typically 10000.0)
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
    seq_len: u32,
    head_dim: u32,
    position: u32,
    theta_base: f32,
) -> Result<()> {
    let device = engine.device();

    // Use cached shader module
    let shader = &pipeline_cache.rope_shader;

    let uniforms = RopeUniforms {
        seq_len,
        head_dim,
        position,
        theta_base,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rope_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_bind_group"),
        layout: &bind_group_layout,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rope_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rope_pipeline"),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Record compute pass to encoder
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup count (256 threads per workgroup)
        let total_elements = seq_len * head_dim;
        let workgroup_count = (total_elements + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    Ok(())
}

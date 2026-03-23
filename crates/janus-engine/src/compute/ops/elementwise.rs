//! Element-wise tensor operations
//!
//! This module provides GPU-accelerated element-wise operations:
//! - `add_tensors`: Element-wise addition for residual connections
//! - `elementwise_mul`: Element-wise multiplication for gating mechanisms

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for element-wise operations
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseUniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Add two tensors element-wise: output = a + b
///
/// Used for residual connections in transformer blocks.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Command encoder to record GPU commands
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `tensor_a` - First input tensor GPU buffer
/// * `tensor_b` - Second input tensor GPU buffer
/// * `output` - Output buffer (pre-allocated)
/// * `size` - Number of elements in each tensor
pub fn add_tensors(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    tensor_a: &wgpu::Buffer,
    tensor_b: &wgpu::Buffer,
    output: &wgpu::Buffer,
    size: u32,
) -> Result<()> {
    let device = engine.device();

    let uniforms = ElementwiseUniforms {
        size,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("add_tensors_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add_tensors_bind_group"),
        layout: &pipeline_cache.add_tensors_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tensor_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tensor_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add_tensors_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_cache.add_tensors_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size + 255) / 256, 1, 1);
    }

    Ok(())
}

/// Multiply two tensors element-wise: output = a * b
///
/// Used for gating mechanisms in transformer FFN blocks.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Command encoder to record GPU commands
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `tensor_a` - First input tensor GPU buffer
/// * `tensor_b` - Second input tensor GPU buffer
/// * `output` - Output buffer (pre-allocated)
/// * `size` - Number of elements in each tensor
pub fn elementwise_mul(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    tensor_a: &wgpu::Buffer,
    tensor_b: &wgpu::Buffer,
    output: &wgpu::Buffer,
    size: u32,
) -> Result<()> {
    let device = engine.device();

    let uniforms = ElementwiseUniforms {
        size,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elementwise_mul_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("elementwise_mul_bind_group"),
        layout: &pipeline_cache.elementwise_mul_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tensor_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tensor_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("elementwise_mul_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_cache.elementwise_mul_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size + 255) / 256, 1, 1);
    }

    Ok(())
}

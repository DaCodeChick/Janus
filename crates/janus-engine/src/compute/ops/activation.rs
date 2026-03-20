//! Activation and normalization operations
//!
//! This module provides GPU-accelerated activation functions and normalization:
//! - `silu`: SiLU (Swish) activation function
//! - `rmsnorm`: Root Mean Square Layer Normalization

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use wgpu::util::DeviceExt;

/// Uniforms structure for SiLU activation
/// Must match the layout in activations.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluUniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Apply SiLU (Swish) activation function: f(x) = x * sigmoid(x)
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `input` - GPU buffer containing input values (f32 array)
/// * `size` - Number of elements in the input
///
/// # Returns
/// GPU buffer containing the output values after SiLU activation
///
/// # Shader
/// Uses the WGSL shader at `shaders/activations.wgsl` for compute operations.
pub async fn silu(
    engine: &ComputeEngine,
    input: &wgpu::Buffer,
    size: u32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/activations.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("activations_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (size * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("silu_bind_group_layout"),
        entries: &[
            // Input (storage, read-only)
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
            // Output (storage, read-write)
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
            // Uniforms
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

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu_bind_group"),
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("silu_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("silu_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("silu"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("silu_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size (256 threads per workgroup)
        let workgroup_count = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    // Wait for completion
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

/// Uniforms structure for RMSNorm
/// Must match the layout in activations.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormUniforms {
    size: u32,
    epsilon: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Apply Root Mean Square Normalization (RMSNorm) with learned gamma weights
///
/// Formula: output[i] = (input[i] / sqrt(mean(input^2) + epsilon)) * gamma[i]
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `input` - GPU buffer containing input values (f32 array)
/// * `weights` - GPU buffer containing gamma weights (f32 array, same size as input)
/// * `size` - Number of elements in the input
/// * `epsilon` - Small constant for numerical stability (typically 1e-6)
///
/// # Returns
/// GPU buffer containing the normalized output values
///
/// # Shader
/// Uses the WGSL shader at `shaders/activations.wgsl` for compute operations.
pub async fn rmsnorm(
    engine: &ComputeEngine,
    input: &wgpu::Buffer,
    weights: &wgpu::Buffer,
    size: u32,
    epsilon: f32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/activations.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("activations_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (size * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let uniforms = RmsNormUniforms {
        size,
        epsilon,
        _pad0: 0,
        _pad1: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsnorm_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rmsnorm_bind_group_layout"),
        entries: &[
            // Input (storage, read-only)
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
            // Output (storage, read-write)
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
            // Gamma weights (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmsnorm_bind_group"),
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
                resource: weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rmsnorm_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rmsnorm_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("rmsnorm"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rmsnorm_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // RMSNorm uses a single workgroup for the reduction
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    // Wait for completion
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

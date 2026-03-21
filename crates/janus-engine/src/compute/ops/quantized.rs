//! Quantized matrix operations
//!
//! This module provides GPU-accelerated operations for quantized tensors:
//! - Q4_K: 4-bit quantization with super-blocks of 256 elements
//! - Q5_K: 5-bit quantization with super-blocks of 256 elements
//! - Q8_0: 8-bit quantization with blocks of 32 elements

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use wgpu::util::DeviceExt;

/// Uniforms structure for Q4_K matrix-vector multiplication
/// Must match the layout in gemm_q4_k.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Q4KUniforms {
    m: u32,          // Number of rows in matrix A
    k: u32,          // Number of columns in matrix A (must be multiple of 256)
    num_blocks: u32, // K / 256
    _pad: u32,
}

/// Q4_K block size in elements (weights)
pub const Q4K_BLOCK_SIZE: usize = 256;

/// Q4_K block size in bytes
/// Structure:
/// - 128 bytes: 256 4-bit weights (2 weights per byte)
/// - 12 bytes: 8 6-bit scales (6 * 8 = 48 bits = 6 bytes) + padding
/// - 12 bytes: 8 6-bit mins (6 * 8 = 48 bits = 6 bytes) + padding  
/// - 4 bytes: super-scale (f16, 2 bytes) + super-min (f16, 2 bytes)
/// Total: 144 bytes per block
pub const Q4K_BLOCK_BYTES: usize = 144;

/// Q5_K block size in elements (weights)
pub const Q5K_BLOCK_SIZE: usize = 256;

/// Q5_K block size in bytes
/// Structure:
/// - 12 bytes: 8 6-bit scales + 8 6-bit mins packed
/// - 32 bytes: high bits for all 5-bit quants
/// - 128 bytes: 256 low 4-bit quants
/// - 4 bytes: super-scale (f16, 2 bytes) + super-min (f16, 2 bytes)
/// Total: 176 bytes per block
pub const Q5K_BLOCK_BYTES: usize = 176;

/// Q8_0 block size in elements (weights)
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Q8_0 block size in bytes
/// Structure:
/// - 2 bytes: scale factor (f16)
/// - 32 bytes: 32 int8 quantized values
/// Total: 34 bytes per block
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Perform matrix-vector multiplication with Q4_K quantized matrix: y = M * x
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `matrix_q4k` - GPU buffer containing the quantized matrix M in Q4_K format (rows × cols, row-major)
/// * `vector` - GPU buffer containing the input vector x (cols f32 elements)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix (must be multiple of 256)
///
/// # Returns
/// GPU buffer containing the output vector y (rows f32 elements)
///
/// # Panics
/// Panics if cols is not a multiple of 256 (Q4_K block size)
///
/// # Shader
/// Uses the WGSL shader at `shaders/gemm_q4_k.wgsl` for on-the-fly dequantization and compute.
pub async fn gemm_q4_k(
    engine: &ComputeEngine,
    matrix_q4k: &wgpu::Buffer,
    vector: &wgpu::Buffer,
    rows: u32,
    cols: u32,
) -> Result<wgpu::Buffer> {
    // Validate that cols is a multiple of block size
    if cols % Q4K_BLOCK_SIZE as u32 != 0 {
        return Err(crate::compute::error::ComputeError::InvalidDimensions(
            format!("Q4_K gemm requires cols to be multiple of {}, got {}", Q4K_BLOCK_SIZE, cols)
        ));
    }

    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/gemm_q4_k.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm_q4_k_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (rows * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm_q4_k_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let num_blocks = cols / Q4K_BLOCK_SIZE as u32;
    let uniforms = Q4KUniforms {
        m: rows,
        k: cols,
        num_blocks,
        _pad: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_q4_k_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gemm_q4_k_bind_group_layout"),
        entries: &[
            // Quantized matrix A (storage, read-only)
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
            // Vector B (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output vector C (storage, read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniforms (uniform buffer)
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
        label: Some("gemm_q4_k_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_q4k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vector.as_entire_binding(),
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

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gemm_q4_k_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_q4_k_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_q4_k_encoder"),
    });

    // Dispatch compute shader
    // Workgroup size is 64x1x1, so we need ceil(rows / 64) workgroups
    let num_workgroups = (rows + 63) / 64;
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_q4_k_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    Ok(output)
}

/// Uniforms structure for Q5_K matrix-vector multiplication
/// Must match the layout in gemm_q5_k.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Q5KUniforms {
    m: u32,          // Number of rows in matrix A
    k: u32,          // Number of columns in matrix A (must be multiple of 256)
    num_blocks: u32, // K / 256
    _pad: u32,
}

/// Perform matrix-vector multiplication with Q5_K quantized matrix: y = M * x
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `matrix_q5k` - GPU buffer containing the quantized matrix M in Q5_K format (rows × cols, row-major)
/// * `vector` - GPU buffer containing the input vector x (cols f32 elements)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix (must be multiple of 256)
///
/// # Returns
/// GPU buffer containing the output vector y (rows f32 elements)
///
/// # Panics
/// Panics if cols is not a multiple of 256 (Q5_K block size)
///
/// # Shader
/// Uses the WGSL shader at `shaders/gemm_q5_k.wgsl` for on-the-fly dequantization and compute.
pub async fn gemm_q5_k(
    engine: &ComputeEngine,
    matrix_q5k: &wgpu::Buffer,
    vector: &wgpu::Buffer,
    rows: u32,
    cols: u32,
) -> Result<wgpu::Buffer> {
    // Validate that cols is a multiple of block size
    if cols % Q5K_BLOCK_SIZE as u32 != 0 {
        return Err(crate::compute::error::ComputeError::InvalidDimensions(
            format!("Q5_K gemm requires cols to be multiple of {}, got {}", Q5K_BLOCK_SIZE, cols)
        ));
    }

    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/gemm_q5_k.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm_q5_k_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (rows * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm_q5_k_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let num_blocks = cols / Q5K_BLOCK_SIZE as u32;
    let uniforms = Q5KUniforms {
        m: rows,
        k: cols,
        num_blocks,
        _pad: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_q5_k_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gemm_q5_k_bind_group_layout"),
        entries: &[
            // Quantized matrix A (storage, read-only)
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
            // Vector B (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output vector C (storage, read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniforms (uniform buffer)
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
        label: Some("gemm_q5_k_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_q5k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vector.as_entire_binding(),
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

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gemm_q5_k_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_q5_k_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_q5_k_encoder"),
    });

    // Dispatch compute shader
    // Workgroup size is 64x1x1, so we need ceil(rows / 64) workgroups
    let num_workgroups = (rows + 63) / 64;
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_q5_k_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    Ok(output)
}

/// Uniforms structure for Q8_0 matrix-vector multiplication
/// Must match the layout in gemm_q8_0.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Q8_0Uniforms {
    m: u32,          // Number of rows in matrix A
    k: u32,          // Number of columns in matrix A (must be multiple of 32)
    num_blocks: u32, // K / 32
    _pad: u32,
}

/// Perform matrix-vector multiplication with Q8_0 quantized matrix: y = M * x
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `matrix_q8_0` - GPU buffer containing the quantized matrix M in Q8_0 format (rows × cols, row-major)
/// * `vector` - GPU buffer containing the input vector x (cols f32 elements)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix (must be multiple of 32)
///
/// # Returns
/// GPU buffer containing the output vector y (rows f32 elements)
///
/// # Panics
/// Panics if cols is not a multiple of 32 (Q8_0 block size)
///
/// # Shader
/// Uses the WGSL shader at `shaders/gemm_q8_0.wgsl` for on-the-fly dequantization and compute.
pub async fn gemm_q8_0(
    engine: &ComputeEngine,
    matrix_q8_0: &wgpu::Buffer,
    vector: &wgpu::Buffer,
    rows: u32,
    cols: u32,
) -> Result<wgpu::Buffer> {
    // Validate that cols is a multiple of block size
    if cols % Q8_0_BLOCK_SIZE as u32 != 0 {
        return Err(crate::compute::error::ComputeError::InvalidDimensions(
            format!("Q8_0 gemm requires cols to be multiple of {}, got {}", Q8_0_BLOCK_SIZE, cols)
        ));
    }

    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/gemm_q8_0.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm_q8_0_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (rows * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm_q8_0_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let num_blocks = cols / Q8_0_BLOCK_SIZE as u32;
    let uniforms = Q8_0Uniforms {
        m: rows,
        k: cols,
        num_blocks,
        _pad: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_q8_0_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gemm_q8_0_bind_group_layout"),
        entries: &[
            // Quantized matrix A (storage, read-only)
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
            // Vector B (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output vector C (storage, read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniforms (uniform buffer)
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
        label: Some("gemm_q8_0_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_q8_0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vector.as_entire_binding(),
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

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gemm_q8_0_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_q8_0_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_q8_0_encoder"),
    });

    // Dispatch compute shader
    // Workgroup size is 64x1x1, so we need ceil(rows / 64) workgroups
    let num_workgroups = (rows + 63) / 64;
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_q8_0_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    Ok(output)
}

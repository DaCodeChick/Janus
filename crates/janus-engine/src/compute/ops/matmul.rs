//! Matrix multiplication operations (matmul and GEMM)
//!
//! This module provides GPU-accelerated matrix operations:
//! - `matmul`: Matrix-vector multiplication (M × x = y)
//! - `gemm`: General matrix-matrix multiplication (A × B = C)

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Uniforms structure for matrix-vector multiplication
/// Must match the layout in matmul.wgsl
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatVecUniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Perform matrix-vector multiplication: y = M * x
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `matrix` - GPU buffer containing the matrix M (rows × cols, row-major f32 data)
/// * `vector` - GPU buffer containing the input vector x (cols f32 elements)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
///
/// # Returns
/// GPU buffer containing the output vector y (rows f32 elements)
///
/// # Shader
/// Uses the WGSL shader at `shaders/matmul.wgsl` for compute operations.
pub async fn matmul(
    engine: &ComputeEngine,
    matrix: &wgpu::Buffer,
    vector: &wgpu::Buffer,
    rows: u32,
    cols: u32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("../shaders/matmul.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (rows * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let uniforms = MatVecUniforms {
        rows,
        cols,
        _pad0: 0,
        _pad1: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul_bind_group_layout"),
        entries: &[
            // Matrix (storage, read-only)
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
            // Vector (storage, read-only)
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
            // Output (storage, read-write)
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
        label: Some("matmul_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix.as_entire_binding(),
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matmul_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size
        // Workgroup size is 256, so we need (rows + 255) / 256 workgroups
        let workgroup_count = (rows + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    Ok(output)
}

/// Uniforms structure for GEMM (General Matrix-Matrix Multiplication)
/// Must match the layout in gemm.wgsl
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmUniforms {
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
    _pad: [u32; 12], // Padding to 64 bytes (WGSL uniform alignment requirement)
}

const _: [(); 16] = [(); std::mem::size_of::<MatVecUniforms>()];
const _: [(); 64] = [(); std::mem::size_of::<GemmUniforms>()];

/// Perform matrix-matrix multiplication (GEMM): C = A * B
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `encoder` - Shared command encoder for recording GPU operations
/// * `pipeline_cache` - Pre-compiled pipeline cache
/// * `matrix_a` - GPU buffer containing matrix A ([batch_size, M, K], row-major f32 data)
/// * `matrix_b` - GPU buffer containing matrix B ([K, N], row-major packed f16 data, shared across batch)
/// * `output` - Pre-allocated output buffer for matrix C ([batch_size, M, N])
/// * `batch_size` - Number of sequences in the batch
/// * `m` - Number of rows in A (per batch item)
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B
///
/// # Note
/// This function records commands to the encoder but does NOT submit them.
/// The caller is responsible for submitting the encoder.
///
/// # Shader
/// Uses the WGSL shader at `shaders/gemm.wgsl` for compute operations.
pub fn gemm(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    matrix_a: &wgpu::Buffer,
    matrix_b: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    let device = engine.device();

    // Create uniforms buffer
    let uniforms = GemmUniforms {
        batch_size,
        m,
        k,
        n,
        _pad: [0; 12],
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group using cached layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_bind_group"),
        layout: &pipeline_cache.gemm_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_b.as_entire_binding(),
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

    // Record compute pass using cached pipeline
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_cache.gemm_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size (16x16 workgroups per batch item)
        let workgroup_count_x = (n + 15) / 16;
        let workgroup_count_y = (m + 15) / 16;
        let workgroup_count_z = batch_size;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z);
    }

    Ok(())
}

pub fn gemm_f32(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    matrix_a: &wgpu::Buffer,
    matrix_b: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    let device = engine.device();

    let uniforms = GemmUniforms {
        batch_size,
        m,
        k,
        n,
        _pad: [0; 12],
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_f32_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_f32_bind_group"),
        layout: &pipeline_cache.gemm_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_b.as_entire_binding(),
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

    let workgroup_count_x = (n + 15) / 16;
    let workgroup_count_y = (m + 15) / 16;

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gemm_f32_pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(&pipeline_cache.gemm_f32_pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, batch_size);

    Ok(())
}

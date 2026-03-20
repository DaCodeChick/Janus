//! GPU compute operations for tensor math

use super::engine::ComputeEngine;
use super::error::Result;
use wgpu::util::DeviceExt;

/// Uniforms structure for matrix-vector multiplication
/// Must match the layout in matmul.wgsl
#[repr(C)]
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
    let shader_source = include_str!("shaders/matmul.wgsl");
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

    // Wait for completion
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

/// Uniforms structure for GEMM (General Matrix-Matrix Multiplication)
/// Must match the layout in gemm.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmUniforms {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Perform matrix-matrix multiplication (GEMM): C = A * B
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `matrix_a` - GPU buffer containing matrix A (M × K, row-major f32 data)
/// * `matrix_b` - GPU buffer containing matrix B (K × N, row-major f32 data)
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A / rows in B
/// * `n` - Number of columns in B
///
/// # Returns
/// GPU buffer containing the output matrix C (M × N, row-major f32 elements)
///
/// # Shader
/// Uses the WGSL shader at `shaders/gemm.wgsl` for compute operations.
pub async fn gemm(
    engine: &ComputeEngine,
    matrix_a: &wgpu::Buffer,
    matrix_b: &wgpu::Buffer,
    m: u32,
    k: u32,
    n: u32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("shaders/gemm.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (m * n * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
    let uniforms = GemmUniforms {
        m,
        k,
        n,
        _pad: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gemm_bind_group_layout"),
        entries: &[
            // Matrix A (storage, read-only)
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
            // Matrix B (storage, read-only)
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
        label: Some("gemm_bind_group"),
        layout: &bind_group_layout,
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gemm_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size (16x16 workgroups)
        let workgroup_count_x = (n + 15) / 16;
        let workgroup_count_y = (m + 15) / 16;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    // Wait for completion
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

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
    let shader_source = include_str!("shaders/activations.wgsl");
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

/// Apply Root Mean Square Normalization (RMSNorm)
///
/// Formula: output[i] = input[i] / sqrt(mean(input^2) + epsilon)
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `input` - GPU buffer containing input values (f32 array)
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
    size: u32,
    epsilon: f32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("shaders/activations.wgsl");
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

/// Apply Rotary Positional Embeddings (RoPE) to queries or keys
///
/// RoPE encodes positional information by rotating pairs of dimensions in the embedding space.
/// This allows the model to understand token ordering without explicit position embeddings.
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `input` - GPU buffer containing input tensor (seq_len * head_dim elements)
/// * `seq_len` - Sequence length (number of tokens)
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `position` - Starting position in the sequence
/// * `theta_base` - Base for frequency calculation (typically 10000.0)
///
/// # Returns
/// GPU buffer containing the output tensor with rotary embeddings applied
///
/// # Shader
/// Uses the WGSL shader at `shaders/rope.wgsl` for compute operations.
pub async fn rope(
    engine: &ComputeEngine,
    input: &wgpu::Buffer,
    seq_len: u32,
    head_dim: u32,
    position: u32,
    theta_base: f32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Load the shader
    let shader_source = include_str!("shaders/rope.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rope_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create output buffer
    let output_size = (seq_len * head_dim * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms buffer
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

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rope_bind_group_layout"),
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rope_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: Default::default(),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rope_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rope_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup dispatch size (256 threads per workgroup)
        // Each thread processes one pair, so total pairs = (seq_len * head_dim) / 2
        let total_pairs = (seq_len * head_dim) / 2;
        let workgroup_count = (total_pairs + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Submit commands
    queue.submit(Some(encoder.finish()));

    // Wait for completion
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

/// Uniforms structure for Attention and Softmax operations
/// Must match the layout in attention.wgsl and softmax.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionUniforms {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
}

/// Uniforms structure for Softmax operation
/// Must match the layout in softmax.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    seq_len: u32,
    num_heads: u32,
    batch_size: u32,
    _pad: u32,
}

/// Compute scaled dot-product attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
///
/// This function implements multi-head attention by:
/// 1. Computing attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
/// 2. Applying softmax to get attention probabilities
/// 3. Multiplying probabilities by values to get output
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `query` - GPU buffer containing query tensor [num_heads * head_dim]
/// * `key_cache` - GPU buffer containing all cached keys [seq_len * num_heads * head_dim]
/// * `value_cache` - GPU buffer containing all cached values [seq_len * num_heads * head_dim]
/// * `seq_len` - Current sequence length (number of tokens in cache)
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension of each attention head
///
/// # Returns
/// GPU buffer containing attention output [num_heads * head_dim]
///
/// # Shaders
/// Uses `shaders/attention.wgsl` and `shaders/softmax.wgsl` for compute operations.
pub async fn compute_attention(
    engine: &ComputeEngine,
    query: &wgpu::Buffer,
    key_cache: &wgpu::Buffer,
    value_cache: &wgpu::Buffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Calculate scale factor: 1/sqrt(head_dim)
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Load shaders
    let attention_shader_source = include_str!("shaders/attention.wgsl");
    let softmax_shader_source = include_str!("shaders/softmax.wgsl");
    
    let attention_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attention_shader"),
        source: wgpu::ShaderSource::Wgsl(attention_shader_source.into()),
    });
    
    let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("softmax_shader"),
        source: wgpu::ShaderSource::Wgsl(softmax_shader_source.into()),
    });

    // Create intermediate buffers
    // Scores buffer: [num_heads * seq_len] - attention scores before softmax
    let scores_size = (num_heads * seq_len * std::mem::size_of::<f32>() as u32) as u64;
    let scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_scores"),
        size: scores_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Attention probabilities buffer: [num_heads * seq_len] - after softmax
    let probs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_probs"),
        size: scores_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Output buffer: [num_heads * head_dim]
    let output_size = (num_heads * head_dim * std::mem::size_of::<f32>() as u32) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attention_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniforms
    let attention_uniforms = AttentionUniforms {
        seq_len,
        num_heads,
        head_dim,
        scale,
    };
    let attention_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attention_uniforms"),
        contents: bytemuck::cast_slice(&[attention_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let softmax_uniforms = SoftmaxUniforms {
        seq_len,
        num_heads,
        batch_size: num_heads, // Each head is a separate batch
        _pad: 0,
    };
    let softmax_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("softmax_uniforms"),
        contents: bytemuck::cast_slice(&[softmax_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ===== Step 1: Compute Q * K^T scores =====
    
    let qk_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("qk_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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

    let qk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qk_bind_group"),
        layout: &qk_bind_group_layout,
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

    let qk_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("qk_pipeline_layout"),
        bind_group_layouts: &[Some(&qk_bind_group_layout)],
        immediate_size: Default::default(),
    });

    let qk_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("qk_pipeline"),
        layout: Some(&qk_pipeline_layout),
        module: &attention_shader,
        entry_point: Some("compute_qk_scores"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ===== Step 2: Apply Softmax =====
    
    let softmax_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("softmax_bind_group_layout"),
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

    let softmax_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softmax_bind_group"),
        layout: &softmax_bind_group_layout,
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

    let softmax_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("softmax_pipeline_layout"),
        bind_group_layouts: &[Some(&softmax_bind_group_layout)],
        immediate_size: Default::default(),
    });

    let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("softmax_pipeline"),
        layout: Some(&softmax_pipeline_layout),
        module: &softmax_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ===== Step 3: Apply attention weights to values =====
    
    let apply_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("apply_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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

    let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("apply_bind_group"),
        layout: &apply_bind_group_layout,
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

    let apply_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("apply_pipeline_layout"),
        bind_group_layouts: &[Some(&apply_bind_group_layout)],
        immediate_size: Default::default(),
    });

    let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apply_pipeline"),
        layout: Some(&apply_pipeline_layout),
        module: &attention_shader,
        entry_point: Some("apply_attention"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ===== Execute all three passes =====
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attention_encoder"),
    });

    // Pass 1: Compute QK scores
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qk_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&qk_pipeline);
        compute_pass.set_bind_group(0, &qk_bind_group, &[]);
        
        // One workgroup per head, 256 threads per workgroup
        compute_pass.dispatch_workgroups(num_heads, 1, 1);
    }

    // Pass 2: Apply softmax
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&softmax_pipeline);
        compute_pass.set_bind_group(0, &softmax_bind_group, &[]);
        
        // One workgroup per head (batch)
        compute_pass.dispatch_workgroups(num_heads, 1, 1);
    }

    // Pass 3: Apply attention to values
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apply_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&apply_pipeline);
        compute_pass.set_bind_group(0, &apply_bind_group, &[]);
        
        // Total threads = num_heads * head_dim, 256 threads per workgroup
        let total_threads = num_heads * head_dim;
        let workgroup_count = (total_threads + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Submit and wait
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to read a GPU buffer back to CPU
    async fn read_buffer_to_vec(engine: &ComputeEngine, buffer: &wgpu::Buffer, size: u64) -> Vec<f32> {
        // Create staging buffer for reading
        let staging_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        engine.queue().submit(Some(encoder.finish()));

        // Map staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());
        rx.await.unwrap().unwrap();

        // Read data
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    #[tokio::test]
    async fn test_matmul_simple() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create a simple 2x3 matrix:
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        let matrix_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Create a 3-element vector: [1.0, 2.0, 3.0]
        let vector_data: Vec<f32> = vec![1.0, 2.0, 3.0];

        // Expected output:
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14.0, 32.0]
        let expected: Vec<f32> = vec![14.0, 32.0];

        // Upload to GPU
        let matrix_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vector_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vector"),
            contents: bytemuck::cast_slice(&vector_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run matmul
        let output = matmul(&engine, &matrix_buffer, &vector_buffer, 2, 3)
            .await
            .expect("matmul failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, 2 * 4).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }
    }

    #[tokio::test]
    async fn test_matmul_4x4() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create a 4x4 matrix (row-major):
        // [1.0,  2.0,  3.0,  4.0]
        // [5.0,  6.0,  7.0,  8.0]
        // [9.0,  10.0, 11.0, 12.0]
        // [13.0, 14.0, 15.0, 16.0]
        let matrix_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        // Create a 4-element vector: [1.0, 2.0, 3.0, 4.0]
        let vector_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Expected output (manual calculation):
        // Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30.0
        // Row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70.0
        // Row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110.0
        // Row 3: 13*1 + 14*2 + 15*3 + 16*4 = 13 + 28 + 45 + 64 = 150.0
        let expected: Vec<f32> = vec![30.0, 70.0, 110.0, 150.0];

        // Upload to GPU
        let matrix_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix_4x4"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vector_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vector_4"),
            contents: bytemuck::cast_slice(&vector_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run matmul
        let output = matmul(&engine, &matrix_buffer, &vector_buffer, 4, 4)
            .await
            .expect("matmul failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, 4 * 4).await;

        // Check results
        assert_eq!(result.len(), expected.len(), "Output vector length mismatch");
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ 4x4 matrix-vector multiplication test passed!");
        println!("  Matrix (4x4) * Vector (4) = Result (4)");
        println!("  Result: {:?}", result);
        println!("  Expected: {:?}", expected);
    }

    #[tokio::test]
    async fn test_gemm_simple() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create matrix A (2x3):
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        let matrix_a: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];

        // Create matrix B (3x2):
        // [7.0,  8.0]
        // [9.0, 10.0]
        // [11.0, 12.0]
        let matrix_b: Vec<f32> = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];

        // Expected output C = A * B (2x2):
        // [1*7+2*9+3*11,  1*8+2*10+3*12]   = [58,  64]
        // [4*7+5*9+6*11,  4*8+5*10+6*12]   = [139, 154]
        let expected: Vec<f32> = vec![
            58.0, 64.0,
            139.0, 154.0,
        ];

        // Upload to GPU
        let buffer_a = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix_a"),
            contents: bytemuck::cast_slice(&matrix_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buffer_b = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix_b"),
            contents: bytemuck::cast_slice(&matrix_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run GEMM: C = A * B where A is 2x3, B is 3x2, C is 2x2
        let output = gemm(&engine, &buffer_a, &buffer_b, 2, 3, 2)
            .await
            .expect("gemm failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, 4 * 4).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ GEMM (2x3 * 3x2) test passed!");
        println!("  Result: {:?}", result);
        println!("  Expected: {:?}", expected);
    }

    #[tokio::test]
    async fn test_gemm_square() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create 3x3 identity-like matrices for easier verification
        // Matrix A (3x3):
        // [2.0, 0.0, 0.0]
        // [0.0, 3.0, 0.0]
        // [0.0, 0.0, 4.0]
        let matrix_a: Vec<f32> = vec![
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0,
        ];

        // Matrix B (3x3):
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        // [7.0, 8.0, 9.0]
        let matrix_b: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        // Expected C = A * B (3x3):
        // [2*1, 2*2, 2*3]     = [2,  4,  6]
        // [3*4, 3*5, 3*6]     = [12, 15, 18]
        // [4*7, 4*8, 4*9]     = [28, 32, 36]
        let expected: Vec<f32> = vec![
            2.0, 4.0, 6.0,
            12.0, 15.0, 18.0,
            28.0, 32.0, 36.0,
        ];

        // Upload to GPU
        let buffer_a = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix_a_3x3"),
            contents: bytemuck::cast_slice(&matrix_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buffer_b = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_matrix_b_3x3"),
            contents: bytemuck::cast_slice(&matrix_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run GEMM
        let output = gemm(&engine, &buffer_a, &buffer_b, 3, 3, 3)
            .await
            .expect("gemm failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, 9 * 4).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ GEMM (3x3 * 3x3) test passed!");
    }

    #[tokio::test]
    async fn test_silu_activation() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Test inputs
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        // Expected output: x * sigmoid(x) = x / (1 + exp(-x))
        // For x = -2.0: -2.0 * sigmoid(-2.0) = -2.0 / (1 + exp(2.0)) ≈ -0.2384
        // For x = -1.0: -1.0 * sigmoid(-1.0) = -1.0 / (1 + exp(1.0)) ≈ -0.2689
        // For x =  0.0:  0.0 * sigmoid(0.0) = 0.0
        // For x =  1.0:  1.0 * sigmoid(1.0) = 1.0 / (1 + exp(-1.0)) ≈ 0.7311
        // For x =  2.0:  2.0 * sigmoid(2.0) = 2.0 / (1 + exp(-2.0)) ≈ 1.7616
        let expected: Vec<f32> = vec![
            -0.23840584,
            -0.26894142,
            0.0,
            0.7310586,
            1.7615942,
        ];

        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_silu_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run SiLU
        let output = silu(&engine, &input_buffer, input.len() as u32)
            .await
            .expect("silu failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, (input.len() * 4) as u64).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ SiLU activation test passed!");
        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", result);
        println!("  Expected: {:?}", expected);
    }

    #[tokio::test]
    async fn test_rmsnorm() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Test input
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let epsilon = 1e-6;

        // Calculate expected output manually:
        // sum_of_squares = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        // mean_square = 30 / 4 = 7.5
        // rms = sqrt(7.5 + 1e-6) ≈ 2.7386127875258306
        // output[i] = input[i] / rms
        let sum_sq = 1.0_f32 + 4.0 + 9.0 + 16.0; // 30.0
        let mean_sq = sum_sq / 4.0; // 7.5
        let rms = (mean_sq + epsilon).sqrt(); // ~2.7386127875258306
        let expected: Vec<f32> = input.iter().map(|&x| x / rms).collect();

        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_rmsnorm_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run RMSNorm
        let output = rmsnorm(&engine, &input_buffer, input.len() as u32, epsilon)
            .await
            .expect("rmsnorm failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, (input.len() * 4) as u64).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ RMSNorm test passed!");
        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", result);
        println!("  Expected: {:?}", expected);
        println!("  RMS value: {}", rms);
    }

    #[tokio::test]
    async fn test_rmsnorm_larger() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Test with a larger array to verify parallel reduction
        let input: Vec<f32> = (1..=512).map(|x| x as f32).collect();
        let epsilon = 1e-6;

        // Calculate expected output
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let mean_sq = sum_sq / input.len() as f32;
        let rms = (mean_sq + epsilon).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| x / rms).collect();

        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_rmsnorm_large_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Run RMSNorm
        let output = rmsnorm(&engine, &input_buffer, input.len() as u32, epsilon)
            .await
            .expect("rmsnorm failed");

        // Read back result
        let result = read_buffer_to_vec(&engine, &output, (input.len() * 4) as u64).await;

        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-3,  // Slightly larger tolerance for accumulated error
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }

        println!("✓ RMSNorm (512 elements) test passed!");
        println!("  RMS value: {}", rms);
    }

    #[tokio::test]
    async fn test_rope_basic() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Test RoPE with a simple case: 1 token, 4-dim head
        // Input: [1.0, 0.0, 1.0, 0.0]
        // Position: 0
        // theta_base: 10000.0
        
        let seq_len = 1;
        let head_dim = 4;
        let position = 0;
        let theta_base = 10000.0;
        
        let input: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        
        // For position 0, the rotation angles should be 0
        // So output should be same as input (no rotation)
        let expected = input.clone();
        
        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_rope_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Run RoPE
        let output = rope(&engine, &input_buffer, seq_len, head_dim, position, theta_base)
            .await
            .expect("rope failed");
        
        // Read back result
        let result = read_buffer_to_vec(&engine, &output, (input.len() * 4) as u64).await;
        
        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }
        
        println!("✓ RoPE basic test passed!");
        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", result);
    }

    #[tokio::test]
    async fn test_rope_with_position() {
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Test RoPE with non-zero position
        let seq_len = 1;
        let head_dim = 2;  // Just one pair for simplicity
        let position = 1;
        let theta_base = 10000.0;
        
        // Input: [1.0, 0.0]
        let input: Vec<f32> = vec![1.0, 0.0];
        
        // Calculate expected output manually
        // theta = 10000^(0/2) = 1.0
        // angle = position / theta = 1.0 / 1.0 = 1.0 radian
        // cos(1.0) ≈ 0.5403
        // sin(1.0) ≈ 0.8415
        // output[0] = 1.0 * cos(1.0) - 0.0 * sin(1.0) = 0.5403
        // output[1] = 1.0 * sin(1.0) + 0.0 * cos(1.0) = 0.8415
        let expected: Vec<f32> = vec![0.5403023, 0.84147096];
        
        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_rope_position_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Run RoPE
        let output = rope(&engine, &input_buffer, seq_len, head_dim, position, theta_base)
            .await
            .expect("rope failed");
        
        // Read back result
        let result = read_buffer_to_vec(&engine, &output, (input.len() * 4) as u64).await;
        
        // Check results
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                res
            );
        }
        
        println!("✓ RoPE with position test passed!");
        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", result);
        println!("  Expected: {:?}", expected);
    }

    #[tokio::test]
    async fn test_kv_cache_initialization() {
        use super::super::cache::KVCache;
        
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create a KV cache for 1024 tokens, 8 heads, 64 dim per head
        let max_seq_len = 1024;
        let num_heads = 8;
        let head_dim = 64;
        
        let cache = KVCache::new(&engine, max_seq_len, num_heads, head_dim)
            .expect("Failed to create KV cache");
        
        // Verify dimensions
        assert_eq!(cache.max_seq_len(), max_seq_len);
        assert_eq!(cache.num_heads(), num_heads);
        assert_eq!(cache.head_dim(), head_dim);
        assert_eq!(cache.current_position(), 0);
        
        println!("✓ KV Cache initialization test passed!");
        println!("  Max sequence length: {}", max_seq_len);
        println!("  Number of heads: {}", num_heads);
        println!("  Head dimension: {}", head_dim);
        println!("  Total cache size: {:.2} MB", 
            (max_seq_len * num_heads * head_dim * 2 * 4) as f64 / (1024.0 * 1024.0));
    }

    #[tokio::test]
    async fn test_kv_cache_update() {
        use super::super::cache::KVCache;
        
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Create a small KV cache for testing
        let max_seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;
        
        let mut cache = KVCache::new(&engine, max_seq_len, num_heads, head_dim)
            .expect("Failed to create KV cache");
        
        // Create fake Key and Value tensors for one token
        // Shape: [num_heads * head_dim] = [2 * 4] = 8 elements
        let new_key: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let new_value: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        // Upload to GPU
        let key_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_new_key"),
            contents: bytemuck::cast_slice(&new_key),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let value_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_new_value"),
            contents: bytemuck::cast_slice(&new_value),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Update cache at position 0
        cache.update(&engine, &key_buffer, &value_buffer, 0)
            .await
            .expect("Failed to update cache");
        
        // Verify position was updated
        assert_eq!(cache.current_position(), 1);
        
        // Read back the key cache to verify it was written correctly
        let cache_size = (max_seq_len * num_heads * head_dim * 4) as u64;
        let key_cache_data = read_buffer_to_vec(&engine, cache.key_cache(), cache_size).await;
        
        // The first 8 elements should match our new_key
        for i in 0..8 {
            assert!(
                (key_cache_data[i] - new_key[i]).abs() < 1e-5,
                "Key cache mismatch at index {}: expected {}, got {}",
                i,
                new_key[i],
                key_cache_data[i]
            );
        }
        
        println!("✓ KV Cache update test passed!");
        println!("  Updated position 0 successfully");
        println!("  Current position: {}", cache.current_position());
    }

    #[tokio::test]
    async fn test_rope_and_cache_integration() {
        use super::super::cache::KVCache;
        
        // Skip test if no GPU available
        let engine = match ComputeEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("Skipping test: No GPU available");
                return;
            }
        };

        // Integration test: RoPE + KV Cache
        let max_seq_len = 1024;
        let num_heads = 8;
        let head_dim = 64;
        let position = 0;
        
        // Create KV cache
        let mut cache = KVCache::new(&engine, max_seq_len, num_heads, head_dim)
            .expect("Failed to create KV cache");
        
        // Create a fake Query/Key tensor (num_heads * head_dim = 512 elements)
        let input: Vec<f32> = (0..512).map(|i| (i as f32) / 100.0).collect();
        
        // Upload to GPU
        let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_integration_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Apply RoPE to get rotated embeddings
        let rope_output = rope(&engine, &input_buffer, 1, num_heads * head_dim, position, 10000.0)
            .await
            .expect("rope failed");
        
        // Store in cache at position 0
        // Note: We're using the same buffer for both key and value for simplicity
        cache.update(&engine, &rope_output, &rope_output, position)
            .await
            .expect("Failed to update cache");
        
        // Verify cache was updated
        assert_eq!(cache.current_position(), 1);
        
        println!("✓ RoPE + KV Cache integration test passed!");
        println!("  Applied RoPE to {} elements", input.len());
        println!("  Stored in cache at position {}", position);
        println!("  Cache current position: {}", cache.current_position());
    }

    #[tokio::test]
    async fn test_attention_simple() {
        // Simple attention test with 2 heads, 4 dimensions, 3 sequence positions
        let engine = ComputeEngine::new()
            .await
            .expect("Failed to create engine");
        
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        
        println!("\n=== Testing Scaled Dot-Product Attention ===");
        println!("Configuration:");
        println!("  num_heads = {}", num_heads);
        println!("  head_dim = {}", head_dim);
        println!("  seq_len = {}", seq_len);
        
        // Create simple test data
        // Query: [num_heads * head_dim] = [2 * 4] = 8 elements
        // For head 0: [1.0, 0.0, 0.0, 0.0]
        // For head 1: [0.0, 1.0, 0.0, 0.0]
        let query_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  // head 0
            0.0, 1.0, 0.0, 0.0,  // head 1
        ];
        
        // Keys: [seq_len * num_heads * head_dim] = [3 * 2 * 4] = 24 elements
        // Position 0, head 0: [1.0, 0.0, 0.0, 0.0]
        // Position 0, head 1: [0.0, 1.0, 0.0, 0.0]
        // Position 1, head 0: [0.5, 0.5, 0.0, 0.0]
        // Position 1, head 1: [0.5, 0.5, 0.0, 0.0]
        // Position 2, head 0: [0.0, 0.0, 1.0, 0.0]
        // Position 2, head 1: [0.0, 0.0, 1.0, 0.0]
        let key_data: Vec<f32> = vec![
            // Position 0
            1.0, 0.0, 0.0, 0.0,  // head 0
            0.0, 1.0, 0.0, 0.0,  // head 1
            // Position 1
            0.5, 0.5, 0.0, 0.0,  // head 0
            0.5, 0.5, 0.0, 0.0,  // head 1
            // Position 2
            0.0, 0.0, 1.0, 0.0,  // head 0
            0.0, 0.0, 1.0, 0.0,  // head 1
        ];
        
        // Values: [seq_len * num_heads * head_dim] = [3 * 2 * 4] = 24 elements
        // Each position has a distinct pattern to verify weighted sum
        let value_data: Vec<f32> = vec![
            // Position 0
            1.0, 0.0, 0.0, 0.0,  // head 0
            0.0, 1.0, 0.0, 0.0,  // head 1
            // Position 1
            2.0, 0.0, 0.0, 0.0,  // head 0
            0.0, 2.0, 0.0, 0.0,  // head 1
            // Position 2
            3.0, 0.0, 0.0, 0.0,  // head 0
            0.0, 3.0, 0.0, 0.0,  // head 1
        ];
        
        let device = engine.device();
        
        // Create GPU buffers
        let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("query"),
            contents: bytemuck::cast_slice(&query_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let key_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("keys"),
            contents: bytemuck::cast_slice(&key_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("values"),
            contents: bytemuck::cast_slice(&value_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Compute attention
        let output = compute_attention(
            &engine,
            &query_buffer,
            &key_buffer,
            &value_buffer,
            seq_len,
            num_heads,
            head_dim,
        )
        .await
        .expect("compute_attention failed");
        
        // Read result
        let result = read_buffer_to_vec(&engine, &output, (num_heads * head_dim * 4) as u64).await;
        
        println!("\nAttention output:");
        for h in 0..num_heads as usize {
            println!("  Head {}: {:?}", h, &result[h*head_dim as usize..(h+1)*head_dim as usize]);
        }
        
        // Verify output shape
        assert_eq!(result.len(), (num_heads * head_dim) as usize);
        
        // For head 0: Query [1,0,0,0] should match strongly with Key position 0 [1,0,0,0]
        // and weakly with position 1 [0.5,0.5,0,0], not at all with position 2 [0,0,1,0]
        // So the output should be dominated by Value position 0
        
        // For head 1: Query [0,1,0,0] should match strongly with Key position 0 [0,1,0,0]
        // and weakly with position 1 [0.5,0.5,0,0], not at all with position 2
        
        // The first dimension of each head should be positive (weighted sum of positive values)
        assert!(result[0] > 0.0, "Head 0, dim 0 should be positive");
        assert!(result[4] < 0.5, "Head 1, dim 0 should be small (near zero)");
        assert!(result[5] > 0.0, "Head 1, dim 1 should be positive");
        
        println!("✓ Attention mechanism test passed!");
        println!("  Processed query with {} heads, {} dimensions", num_heads, head_dim);
        println!("  Computed attention over {} sequence positions", seq_len);
    }

    #[tokio::test]
    async fn test_attention_with_cache() {
        // Test attention integrated with KVCache
        let engine = ComputeEngine::new()
            .await
            .expect("Failed to create engine");
        
        let num_heads = 2;
        let head_dim = 8;
        let max_seq_len = 128;
        let current_seq_len = 4;
        
        println!("\n=== Testing Attention with KVCache ===");
        println!("Configuration:");
        println!("  num_heads = {}", num_heads);
        println!("  head_dim = {}", head_dim);
        println!("  max_seq_len = {}", max_seq_len);
        println!("  current_seq_len = {}", current_seq_len);
        
        // Create KV cache
        let mut cache = super::super::cache::KVCache::new(&engine, max_seq_len, num_heads, head_dim)
            .expect("Failed to create cache");
        
        let device = engine.device();
        
        // Populate cache with some test data for first 4 positions
        for pos in 0..current_seq_len {
            // Create simple keys and values for this position
            let key_data: Vec<f32> = (0..num_heads * head_dim)
                .map(|i| (pos as f32 + 1.0) * (i % head_dim) as f32 / head_dim as f32)
                .collect();
            
            let value_data: Vec<f32> = (0..num_heads * head_dim)
                .map(|i| (pos as f32 + 1.0) / (i % head_dim + 1) as f32)
                .collect();
            
            let key_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("key_{}", pos)),
                contents: bytemuck::cast_slice(&key_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            
            let value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("value_{}", pos)),
                contents: bytemuck::cast_slice(&value_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            
            cache.update(&engine, &key_buffer, &value_buffer, pos)
                .await
                .expect("Failed to update cache");
        }
        
        println!("Populated cache with {} positions", current_seq_len);
        
        // Create a query
        let query_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| if i % head_dim == 0 { 1.0 } else { 0.1 })
            .collect();
        
        let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("query"),
            contents: bytemuck::cast_slice(&query_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Get cache buffers
        let (key_cache, value_cache) = cache.buffers();
        
        // Compute attention
        let output = compute_attention(
            &engine,
            &query_buffer,
            key_cache,
            value_cache,
            current_seq_len,
            num_heads,
            head_dim,
        )
        .await
        .expect("compute_attention failed");
        
        // Read result
        let result = read_buffer_to_vec(&engine, &output, (num_heads * head_dim * 4) as u64).await;
        
        println!("\nAttention output with cache:");
        for h in 0..num_heads as usize {
            let head_output = &result[h*head_dim as usize..(h+1)*head_dim as usize];
            println!("  Head {}: first 4 dims = [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     h, head_output[0], head_output[1], head_output[2], head_output[3]);
        }
        
        // Verify output shape
        assert_eq!(result.len(), (num_heads * head_dim) as usize);
        
        // All outputs should be finite (not NaN or Inf)
        for &val in &result {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
        
        println!("✓ Attention with KVCache test passed!");
        println!("  Successfully computed attention over cached K/V tensors");
        println!("  Output shape: [{}]", result.len());
    }
}

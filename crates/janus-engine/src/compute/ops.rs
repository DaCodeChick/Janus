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
}

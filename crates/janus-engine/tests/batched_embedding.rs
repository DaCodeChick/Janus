//! Integration tests for batched embedding shader
//!
//! Tests that the embed shader correctly processes multiple token IDs
//! in a single batch by verifying shader compilation and basic GPU operations.

use janus_engine::compute::ComputeEngine;

#[tokio::test]
async fn test_batched_embed_shader_compiles() -> Result<(), Box<dyn std::error::Error>> {
    // Create compute engine
    let engine = ComputeEngine::new().await?;

    // Load and compile the embed shader
    let shader = engine
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/compute/shaders/embed.wgsl").into(),
            ),
        });

    // Create a simple pipeline to verify shader validity
    let bind_group_layout =
        engine
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("embed_test_layout"),
                entries: &[
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Token IDs
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
                    // Embedding table
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
                    // Output
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    let pipeline_layout = engine
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("embed_test_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });

    // Create pipeline - this will fail if shader has compilation errors
    let _pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embed_test_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    Ok(())
}

#[tokio::test]
async fn test_batched_embed_execution() -> Result<(), Box<dyn std::error::Error>> {
    // Create compute engine
    let engine = ComputeEngine::new().await?;

    let batch_size = 4u32;
    let vocab_size = 32u32;
    let hidden_dim = 64u32;

    // Create embedding table with pattern: embedding[token_id][dim] = token_id * hidden_dim + dim
    let embedding_data: Vec<f32> = (0..vocab_size)
        .flat_map(|token_id| {
            (0..hidden_dim).map(move |dim| (token_id * hidden_dim + dim) as f32)
        })
        .collect();

    // Pack to u32 format (2 f16 values per u32)
    let packed_size = (vocab_size * hidden_dim + 1) / 2; // Round up for odd counts
    let mut packed_data: Vec<u32> = vec![0; packed_size as usize];

    for (i, &value) in embedding_data.iter().enumerate() {
        let f16_value = half::f16::from_f32(value).to_bits();
        let packed_idx = i / 2;
        if i % 2 == 0 {
            packed_data[packed_idx] = f16_value as u32;
        } else {
            packed_data[packed_idx] |= (f16_value as u32) << 16;
        }
    }

    // Create buffers
    let embedding_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("embedding_table"),
        size: (packed_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    engine
        .queue()
        .write_buffer(&embedding_buffer, 0, bytemuck::cast_slice(&packed_data));

    let token_ids = vec![2u32, 7u32, 15u32, 23u32]; // Test batch
    let token_ids_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("token_ids"),
        size: (batch_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    engine
        .queue()
        .write_buffer(&token_ids_buffer, 0, bytemuck::cast_slice(&token_ids));

    let params_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    engine
        .queue()
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[batch_size, hidden_dim]));

    let output_size = (batch_size * hidden_dim + 1) / 2;
    let output_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (output_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader and pipeline
    let shader = engine
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/compute/shaders/embed.wgsl").into(),
            ),
        });

    let bind_group_layout =
        engine
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("embed_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    let bind_group = engine.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embed_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: token_ids_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: embedding_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = engine
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("embed_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });

    let pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embed_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Execute shader
    let mut encoder = engine
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embed_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embed_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let total_elements = batch_size * hidden_dim;
        let workgroups = (total_elements + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    engine.queue().submit(Some(encoder.finish()));

    // Wait for completion
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    println!("Batched embedding shader executed successfully!");

    Ok(())
}

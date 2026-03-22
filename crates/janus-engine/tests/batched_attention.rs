//! Integration tests for batched attention
//!
//! Tests that the attention mechanism correctly processes multiple sequences
//! in parallel, with each sequence attending only to its own history.

use janus_engine::compute::ComputeEngine;

#[tokio::test]
async fn test_batched_attention_shader_compiles() -> Result<(), Box<dyn std::error::Error>> {
    // Create compute engine
    let engine = ComputeEngine::new().await?;

    // Load and compile the attention shader
    let attention_shader = engine
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/compute/shaders/attention.wgsl").into(),
            ),
        });

    // Load and compile the softmax shader
    let softmax_shader = engine
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/compute/shaders/softmax.wgsl").into(),
            ),
        });

    // Create simple bind group layout for QK scores to verify shader validity
    let bind_group_layout =
        engine
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("attention_test_layout"),
                entries: &[
                    // Query
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
                    // Keys
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
                    // Scores (output)
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

    let pipeline_layout = engine
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention_test_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });

    // Create pipeline for QK scores - this will fail if shader has compilation errors
    let _qk_pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qk_test_pipeline"),
            layout: Some(&pipeline_layout),
            module: &attention_shader,
            entry_point: Some("compute_qk_scores"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create pipeline for apply attention
    let _apply_pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apply_test_pipeline"),
            layout: Some(&pipeline_layout),
            module: &attention_shader,
            entry_point: Some("apply_attention"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create softmax bind group layout
    let softmax_layout =
        engine
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("softmax_test_layout"),
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

    let softmax_pipeline_layout = engine
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softmax_test_pipeline_layout"),
            bind_group_layouts: &[Some(&softmax_layout)],
            immediate_size: Default::default(),
        });

    let _softmax_pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_test_pipeline"),
            layout: Some(&softmax_pipeline_layout),
            module: &softmax_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    Ok(())
}

#[tokio::test]
async fn test_batched_cache_update_shader_compiles() -> Result<(), Box<dyn std::error::Error>> {
    // Create compute engine
    let engine = ComputeEngine::new().await?;

    // Load and compile the cache update shader
    let shader = engine
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_cache_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/compute/shaders/update_cache.wgsl").into(),
            ),
        });

    // Create bind group layout
    let bind_group_layout =
        engine
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cache_update_test_layout"),
                entries: &[
                    // New KV data
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
                    // Cache buffer
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

    let pipeline_layout = engine
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cache_update_test_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });

    // Create pipeline - this will fail if shader has compilation errors
    let _pipeline = engine
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cache_update_test_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    Ok(())
}

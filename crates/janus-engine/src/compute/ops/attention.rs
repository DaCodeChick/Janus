//! Attention mechanisms and softmax operations
//!
//! This module implements scaled dot-product attention for transformer models.
//! The attention computation follows three steps:
//! 1. Compute attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
//! 2. Apply softmax to get attention probabilities
//! 3. Multiply probabilities by values to get output

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use wgpu::util::DeviceExt;

/// Uniforms structure for Attention operations
/// Must match the layout in attention.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionUniforms {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    _pad: [u32; 3],
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

/// Compute scaled dot-product attention with Grouped Query Attention (GQA) support
///
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
///
/// This function implements multi-head attention with GQA by:
/// 1. Computing attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
/// 2. Applying softmax to get attention probabilities
/// 3. Multiplying probabilities by values to get output
///
/// For GQA, each KV head is shared across multiple query heads (num_heads / num_kv_heads).
///
/// # Arguments
/// * `engine` - The compute engine containing GPU device and queue
/// * `query` - GPU buffer containing query tensor [num_heads * head_dim]
/// * `key_cache` - GPU buffer containing all cached keys [seq_len * num_kv_heads * head_dim]
/// * `value_cache` - GPU buffer containing all cached values [seq_len * num_kv_heads * head_dim]
/// * `seq_len` - Current sequence length (number of tokens in cache)
/// * `num_heads` - Number of query attention heads
/// * `num_kv_heads` - Number of key-value attention heads (for GQA)
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
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<wgpu::Buffer> {
    let device = engine.device();
    let queue = engine.queue();

    // Calculate scale factor: 1/sqrt(head_dim)
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Load shaders
    let attention_shader_source = include_str!("../shaders/attention.wgsl");
    let softmax_shader_source = include_str!("../shaders/softmax.wgsl");
    
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
        num_kv_heads,
        head_dim,
        scale,
        _pad: [0; 3],
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

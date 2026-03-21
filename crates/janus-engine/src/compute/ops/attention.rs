//! Attention mechanisms and softmax operations
//!
//! This module implements scaled dot-product attention for transformer models.
//! The attention computation follows three steps:
//! 1. Compute attention scores (Q * K^T) scaled by 1/sqrt(head_dim)
//! 2. Apply softmax to get attention probabilities
//! 3. Multiply probabilities by values to get output

use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
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
    layer_idx: u32,
    max_seq_len: u32,
    _pad: u32,
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
/// * `engine` - The compute engine providing GPU access
/// * `encoder` - Shared command encoder for batching operations
/// * `pipeline_cache` - Pre-compiled shader cache
/// * `query` - Query tensor [num_heads, head_dim]
/// * `key_cache` - Segmented key cache [num_layers, max_seq_len, num_kv_heads, head_dim]
/// * `value_cache` - Segmented value cache [num_layers, max_seq_len, num_kv_heads, head_dim]
/// * `output` - Output tensor [num_heads, head_dim]
/// * `scores` - Pre-allocated buffer for attention scores [num_heads * max_seq_len]
/// * `probs` - Pre-allocated buffer for attention probabilities [num_heads * max_seq_len]
/// * `layer_idx` - Layer index for cache segmentation
/// * `seq_len` - Current sequence length (number of tokens processed so far, including current)
/// * `max_seq_len` - Maximum sequence length for cache sizing
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads (GQA support)
/// * `head_dim` - Dimension of each attention head
///
/// # Shaders
/// Uses `shaders/attention.wgsl` and `shaders/softmax.wgsl` for compute operations.
pub fn compute_attention(
    engine: &ComputeEngine,
    encoder: &mut wgpu::CommandEncoder,
    pipeline_cache: &PipelineCache,
    query: &wgpu::Buffer,
    key_cache: &wgpu::Buffer,
    value_cache: &wgpu::Buffer,
    output: &wgpu::Buffer,
    scores: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    layer_idx: u32,
    seq_len: u32,
    max_seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let device = engine.device();
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use cached shaders
    let attention_shader = &pipeline_cache.attention_shader;
    let softmax_shader = &pipeline_cache.softmax_shader;

    // Create uniforms
    let attention_uniforms = AttentionUniforms {
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        layer_idx,
        max_seq_len,
        _pad: 0,
    };
    let attention_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attention_uniforms"),
        contents: bytemuck::cast_slice(&[attention_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let softmax_uniforms = SoftmaxUniforms {
        seq_len,
        num_heads,
        batch_size: num_heads,
        _pad: 0,
    };
    let softmax_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("softmax_uniforms"),
        contents: bytemuck::cast_slice(&[softmax_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // === Step 1: QK scores ===
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

    // === Step 2: Softmax ===
    let softmax_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    // === Step 3: Apply attention ===
    let apply_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    // Record all three passes
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qk_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&qk_pipeline);
        compute_pass.set_bind_group(0, &qk_bind_group, &[]);
        compute_pass.dispatch_workgroups(num_heads, 1, 1);
    }
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&softmax_pipeline);
        compute_pass.set_bind_group(0, &softmax_bind_group, &[]);
        compute_pass.dispatch_workgroups(num_heads, 1, 1);
    }
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apply_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&apply_pipeline);
        compute_pass.set_bind_group(0, &apply_bind_group, &[]);
        let workgroup_count = (num_heads * head_dim + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    Ok(())
}

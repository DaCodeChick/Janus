//! Tests for GPU compute operations (migrated to encoder/pipeline-cache APIs)

use super::*;
use crate::compute::cache::KVCache;
use crate::compute::engine::ComputeEngine;
use crate::compute::pipeline_cache::PipelineCache;
use half::f16;
use wgpu::util::DeviceExt;

fn f32_to_f16_packed_u32(data: &[f32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len().div_ceil(2));
    let mut i = 0usize;
    while i < data.len() {
        let a = f16::from_f32(data[i]).to_bits() as u32;
        let b = if i + 1 < data.len() {
            f16::from_f32(data[i + 1]).to_bits() as u32
        } else {
            0u32
        };
        out.push(a | (b << 16));
        i += 2;
    }
    out
}

fn approx_eq_slice(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= eps,
            "index {} mismatch: expected {}, got {}",
            i,
            e,
            a
        );
    }
}

async fn read_buffer_to_vec(engine: &ComputeEngine, buffer: &wgpu::Buffer, size: u64) -> Vec<f32> {
    let staging_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    engine.queue().submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());
    match rx.await {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => panic!("failed to map staging buffer"),
    }

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    result
}

async fn read_u32_buffer(engine: &ComputeEngine, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_u32"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy_u32_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    engine.queue().submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());
    match rx.await {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => panic!("failed to map u32 staging buffer"),
    }

    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    result
}

#[tokio::test]
async fn test_matmul_simple() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };

    let matrix_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector_data = vec![1.0f32, 2.0, 3.0];
    let expected = vec![14.0f32, 32.0];

    let packed = f32_to_f16_packed_u32(&matrix_data);
    let matrix_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_matrix"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let vector_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_vector"),
        contents: bytemuck::cast_slice(&vector_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output = match matmul(&engine, &matrix_buffer, &vector_buffer, 2, 3).await {
        Ok(v) => v,
        Err(e) => panic!("matmul failed: {}", e),
    };
    let result = read_buffer_to_vec(&engine, &output, 2 * 4).await;
    approx_eq_slice(&result, &expected, 1e-3);
}

#[tokio::test]
async fn test_gemm_simple() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let matrix_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [1,2,3 ; 4,5,6]
    let matrix_b_f32 = vec![7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]; // [N=2,K=3], transposed usage
    let matrix_b_packed = f32_to_f16_packed_u32(&matrix_b_f32);
    let expected = vec![58.0f32, 64.0, 139.0, 154.0];

    let buffer_a = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_a"),
        contents: bytemuck::cast_slice(&matrix_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_b = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_b"),
        contents: bytemuck::cast_slice(&matrix_b_packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm_out"),
        size: (4 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_encoder"),
    });
    if let Err(e) = gemm(
        &engine,
        &mut encoder,
        &pipeline_cache,
        &buffer_a,
        &buffer_b,
        &output,
        1,
        2,
        3,
        2,
    ) {
        panic!("gemm failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let result = read_buffer_to_vec(&engine, &output, 4 * 4).await;
    approx_eq_slice(&result, &expected, 1e-2);
}

#[tokio::test]
async fn test_silu_and_rmsnorm() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
    let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("act_input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let silu_out = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("silu_out"),
        size: (input.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("silu_encoder"),
    });
    if let Err(e) = silu(
        &engine,
        &mut encoder,
        &pipeline_cache,
        &input_buffer,
        &silu_out,
        input.len() as u32,
    ) {
        panic!("silu failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let silu_vals = read_buffer_to_vec(&engine, &silu_out, (input.len() * 4) as u64).await;
    assert!(silu_vals.iter().all(|v| v.is_finite()));

    let gamma_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamma"),
        contents: bytemuck::cast_slice(&gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rms_out = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms_out"),
        size: (input.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rms_encoder"),
    });
    if let Err(e) = rmsnorm(
        &engine,
        &mut encoder,
        &pipeline_cache,
        &input_buffer,
        &rms_out,
        &gamma_buffer,
        1,
        input.len() as u32,
        1e-5,
    ) {
        panic!("rmsnorm failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let rms_vals = read_buffer_to_vec(&engine, &rms_out, (input.len() * 4) as u64).await;
    assert!(rms_vals.iter().all(|v| v.is_finite()));
}

#[tokio::test]
async fn test_elementwise_add_and_mul() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let expected_add = vec![11.0f32, 22.0, 33.0, 44.0];
    let expected_mul = vec![10.0f32, 40.0, 90.0, 160.0];

    let a_buf = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a_buf"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b_buf"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_add = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_add"),
        size: (a.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_mul = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_mul"),
        size: (a.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("elementwise_encoder"),
    });
    if let Err(e) = add_tensors(&engine, &mut encoder, &pipeline_cache, &a_buf, &b_buf, &out_add, a.len() as u32)
    {
        panic!("add_tensors failed: {}", e);
    }
    if let Err(e) = elementwise_mul(&engine, &mut encoder, &pipeline_cache, &a_buf, &b_buf, &out_mul, a.len() as u32)
    {
        panic!("elementwise_mul failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let add_vals = read_buffer_to_vec(&engine, &out_add, (a.len() * 4) as u64).await;
    let mul_vals = read_buffer_to_vec(&engine, &out_mul, (a.len() * 4) as u64).await;
    approx_eq_slice(&add_vals, &expected_add, 1e-5);
    approx_eq_slice(&mul_vals, &expected_mul, 1e-5);
}

#[tokio::test]
async fn test_rope_basic() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let input = vec![1.0f32, 0.0, 1.0, 0.0];
    let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_output"),
        size: (input.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Build a simple rope cache for head_dim=4 and max_seq_len=2 in [cos,sin,cos,sin...] layout
    let rope_cache_data = vec![
        1.0f32, 0.0, 1.0, 0.0, // pos 0
        0.5403023, 0.84147096, 0.99995, 0.009999833, // pos 1
    ];
    let rope_cache = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_cache_test"),
        contents: bytemuck::cast_slice(&rope_cache_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rope_encoder"),
    });
    if let Err(e) = rope(
        &engine,
        &mut encoder,
        &pipeline_cache,
        &input_buffer,
        &output_buffer,
        &rope_cache,
        1,
        1,
        4,
        0,
    ) {
        panic!("rope failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let result = read_buffer_to_vec(&engine, &output_buffer, (input.len() * 4) as u64).await;
    approx_eq_slice(&result, &input, 1e-4);
}

#[tokio::test]
async fn test_argmax_basic() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let logits = vec![-1.0f32, 0.5, 4.2, 3.9, 2.0];
    let logits_buf = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("argmax_logits"),
        contents: bytemuck::cast_slice(&logits),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmax_out"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("argmax_encoder"),
    });
    if let Err(e) = argmax(&engine, &mut encoder, &pipeline_cache, &logits_buf, &out, logits.len() as u32, 1)
    {
        panic!("argmax failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    let out_vals = read_u32_buffer(&engine, &out, 1).await;
    assert_eq!(out_vals[0], 2);
}

#[tokio::test]
async fn test_kv_cache_update_and_buffers() {
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping test: No GPU available");
            return;
        }
    };
    let pipeline_cache = PipelineCache::new(engine.device());

    let mut cache = match KVCache::new(&engine, 1, 1, 16, 2, 4) {
        Ok(c) => c,
        Err(e) => panic!("kv cache create failed: {}", e),
    };

    let token_dim = (2 * 4) as usize;
    let key = vec![0.1f32; token_dim];
    let value = vec![0.2f32; token_dim];
    let key_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kv_key"),
        contents: bytemuck::cast_slice(&key),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let value_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kv_value"),
        contents: bytemuck::cast_slice(&value),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mut encoder = engine.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("kv_update_encoder"),
    });
    if let Err(e) = cache.update(
        &engine,
        &mut encoder,
        &pipeline_cache,
        &key_buffer,
        &value_buffer,
        0,
        0,
    ) {
        panic!("kv update failed: {}", e);
    }
    engine.queue().submit(Some(encoder.finish()));
    let _ = engine.device().poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(cache.current_position(), 1);
    let (_k, _v) = cache.buffers();
}

//! Test Q4_K quantized matrix operations

use crate::compute::engine::ComputeEngine;
use crate::compute::ops::{gemm_q4_k, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

#[tokio::test]
async fn test_q4k_shader_compilation() {
    // This test validates that the Q4_K WGSL shader compiles correctly
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping Q4_K test (no GPU available)");
            return;
        }
    };

    println!("GPU initialized for Q4_K test: {:?}", engine.adapter_info().name);

    // Create pipeline cache
    let pipeline_cache = PipelineCache::new(engine.device());

    // Create dummy Q4_K data (1 block = 256 elements = 144 bytes)
    let num_blocks = 1;
    let rows = 1;
    let cols = Q4K_BLOCK_SIZE as u32;

    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES * num_blocks];
    let vector_data = vec![1.0f32; cols as usize];

    // Upload to GPU
    let matrix_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_q4k_matrix"),
        contents: &q4k_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let vector_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_q4k_vector"),
        contents: bytemuck::cast_slice(&vector_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Run Q4_K GEMM
    let result = gemm_q4_k(&engine, &pipeline_cache, &matrix_buffer, &vector_buffer, rows, cols).await;

    assert!(result.is_ok(), "Q4_K GEMM should compile and run successfully");
    println!("Q4_K shader compiled and executed successfully!");
}

#[test]
fn test_q4k_constants() {
    assert_eq!(Q4K_BLOCK_SIZE, 256, "Q4_K block should contain 256 elements");
    assert_eq!(Q4K_BLOCK_BYTES, 144, "Q4_K block should be 144 bytes");
}

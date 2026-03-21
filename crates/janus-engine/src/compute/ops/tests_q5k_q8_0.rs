//! Test Q5_K and Q8_0 quantized matrix operations

use crate::compute::engine::ComputeEngine;
use crate::compute::ops::{gemm_q5_k, gemm_q8_0, Q5K_BLOCK_BYTES, Q5K_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};

#[tokio::test]
async fn test_q5k_shader_compiles() {
    // This test validates that the Q5_K WGSL shader compiles correctly
    let engine = ComputeEngine::new().await;
    let engine = match engine {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping Q5_K test (no GPU available)");
            return;
        }
    };

    println!("GPU initialized for Q5_K test: {:?}", engine.adapter_info().name);

    // Create dummy Q5_K data (1 block = 256 elements = 176 bytes)
    let rows = 2u32;
    let cols = 256u32;
    let matrix_size = (rows * cols) as usize / Q5K_BLOCK_SIZE * Q5K_BLOCK_BYTES;
    let matrix_data = vec![0u8; matrix_size];
    
    let matrix_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_q5k_matrix"),
        size: matrix_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    engine.queue().write_buffer(&matrix_buffer, 0, &matrix_data);
    
    let vector_data = vec![1.0f32; cols as usize];
    let vector_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_q5k_vector"),
        size: (cols * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    engine.queue().write_buffer(&vector_buffer, 0, bytemuck::cast_slice(&vector_data));
    
    // Run Q5_K GEMM
    let result = gemm_q5_k(&engine, &matrix_buffer, &vector_buffer, rows, cols).await;
    
    assert!(result.is_ok(), "Q5_K GEMM should compile and run successfully");
    println!("Q5_K shader compiled and executed successfully!");
}

#[test]
fn test_q5k_constants() {
    assert_eq!(Q5K_BLOCK_SIZE, 256, "Q5_K block should contain 256 elements");
    assert_eq!(Q5K_BLOCK_BYTES, 176, "Q5_K block should be 176 bytes");
}

#[tokio::test]
async fn test_q8_0_shader_compiles() {
    // This test validates that the Q8_0 WGSL shader compiles correctly
    let engine = ComputeEngine::new().await;
    let engine = match engine {
        Ok(e) => e,
        Err(_) => {
            println!("Skipping Q8_0 test (no GPU available)");
            return;
        }
    };

    println!("GPU initialized for Q8_0 test: {:?}", engine.adapter_info().name);

    // Create dummy Q8_0 data (1 block = 32 elements = 34 bytes, padded to 36 bytes for u32 alignment)
    let rows = 4u32;
    let cols = 64u32; // 2 blocks
    let matrix_size = (rows * cols) as usize / Q8_0_BLOCK_SIZE * 36; // 36 bytes per block (padded)
    let matrix_data = vec![0u8; matrix_size];
    
    let matrix_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_q8_0_matrix"),
        size: matrix_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    engine.queue().write_buffer(&matrix_buffer, 0, &matrix_data);
    
    let vector_data = vec![1.0f32; cols as usize];
    let vector_buffer = engine.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_q8_0_vector"),
        size: (cols * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    engine.queue().write_buffer(&vector_buffer, 0, bytemuck::cast_slice(&vector_data));
    
    // Run Q8_0 GEMM
    let result = gemm_q8_0(&engine, &matrix_buffer, &vector_buffer, rows, cols).await;
    
    assert!(result.is_ok(), "Q8_0 GEMM should compile and run successfully");
    println!("Q8_0 shader compiled and executed successfully!");
}

#[test]
fn test_q8_0_constants() {
    assert_eq!(Q8_0_BLOCK_SIZE, 32, "Q8_0 block should contain 32 elements");
    assert_eq!(Q8_0_BLOCK_BYTES, 34, "Q8_0 block should be 34 bytes");
}

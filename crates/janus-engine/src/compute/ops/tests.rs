//! Tests for GPU compute operations

use super::*;
use crate::compute::cache::KVCache;
use crate::compute::engine::ComputeEngine;
use wgpu::util::DeviceExt;

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
    let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0]; // Identity gamma for simple test
    let epsilon = 1e-6;

    // Calculate expected output manually:
    // sum_of_squares = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
    // mean_square = 30 / 4 = 7.5
    // rms = sqrt(7.5 + 1e-6) ≈ 2.7386127875258306
    // output[i] = (input[i] / rms) * gamma[i]
    let sum_sq = 1.0_f32 + 4.0 + 9.0 + 16.0; // 30.0
    let mean_sq = sum_sq / 4.0; // 7.5
    let rms = (mean_sq + epsilon).sqrt(); // ~2.7386127875258306
    let expected: Vec<f32> = input.iter().zip(gamma.iter()).map(|(&x, &g)| (x / rms) * g).collect();

    // Upload to GPU
    let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_rmsnorm_input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gamma_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_rmsnorm_gamma"),
        contents: bytemuck::cast_slice(&gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Run RMSNorm
    let output = rmsnorm(&engine, &input_buffer, &gamma_buffer, input.len() as u32, epsilon)
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
    println!("  Gamma:    {:?}", gamma);
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
    let gamma: Vec<f32> = vec![1.0; 512]; // Identity gamma
    let epsilon = 1e-6;

    // Calculate expected output
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let mean_sq = sum_sq / input.len() as f32;
    let rms = (mean_sq + epsilon).sqrt();
    let expected: Vec<f32> = input.iter().zip(gamma.iter()).map(|(&x, &g)| (x / rms) * g).collect();

    // Upload to GPU
    let input_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_rmsnorm_large_input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gamma_buffer = engine.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_rmsnorm_large_gamma"),
        contents: bytemuck::cast_slice(&gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Run RMSNorm
    let output = rmsnorm(&engine, &input_buffer, &gamma_buffer, input.len() as u32, epsilon)
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
        num_heads, // num_kv_heads = num_heads for MHA
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
    let mut cache = KVCache::new(&engine, max_seq_len, num_heads, head_dim)
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
        num_heads, // num_kv_heads = num_heads for MHA
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

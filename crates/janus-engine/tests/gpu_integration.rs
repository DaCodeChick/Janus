//! Integration test for ComputeEngine and GGUF tensor allocation

use janus_engine::compute::ComputeEngine;
use janus_engine::gguf::{GGMLType, GGUFFile};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

/// Create a minimal valid GGUF file for testing
fn create_test_gguf_file(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("test_model.gguf");
    let mut file = File::create(&path).unwrap();

    // GGUF header
    file.write_all(b"GGUF").unwrap(); // Magic
    file.write_all(&2u32.to_le_bytes()).unwrap(); // Version 2
    file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
    file.write_all(&1u64.to_le_bytes()).unwrap(); // 1 metadata entry

    // Metadata: general.alignment = 32
    let key = "general.alignment";
    file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    file.write_all(key.as_bytes()).unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap(); // Type: UInt32
    file.write_all(&32u32.to_le_bytes()).unwrap(); // Value: 32

    // Tensor 1: "tensor_a" - 4x4 F32 matrix
    let tensor1_name = "tensor_a";
    file.write_all(&(tensor1_name.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(tensor1_name.as_bytes()).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap(); // 2 dimensions
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim[0] = 4
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim[1] = 4
    file.write_all(&(GGMLType::F32 as u32).to_le_bytes())
        .unwrap(); // F32
    file.write_all(&0u64.to_le_bytes()).unwrap(); // offset = 0

    // Tensor 2: "tensor_b" - 8 F32 values
    let tensor2_name = "tensor_b";
    file.write_all(&(tensor2_name.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(tensor2_name.as_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap(); // 1 dimension
    file.write_all(&8u64.to_le_bytes()).unwrap(); // dim[0] = 8
    file.write_all(&(GGMLType::F32 as u32).to_le_bytes())
        .unwrap(); // F32
    file.write_all(&64u64.to_le_bytes()).unwrap(); // offset = 64 (after tensor_a)

    // Pad to alignment (32 bytes)
    let current_pos = file.metadata().unwrap().len();
    let aligned_pos = ((current_pos + 31) / 32) * 32;
    let padding = aligned_pos - current_pos;
    file.write_all(&vec![0u8; padding as usize]).unwrap();

    // Tensor data
    // Tensor A: 16 F32 values (4x4 matrix)
    for i in 0..16 {
        file.write_all(&(i as f32).to_le_bytes()).unwrap();
    }

    // Tensor B: 8 F32 values
    for i in 0..8 {
        file.write_all(&((i + 100) as f32).to_le_bytes()).unwrap();
    }

    path
}

#[tokio::test]
async fn test_allocate_tensors_to_gpu() {
    // Create a temporary GGUF file
    let temp_dir = TempDir::new().unwrap();
    let gguf_path = create_test_gguf_file(&temp_dir);

    // Parse the GGUF file
    let gguf = GGUFFile::open(&gguf_path).expect("Failed to open GGUF file");

    // Verify we have 2 tensors
    assert_eq!(gguf.tensors().len(), 2);
    assert_eq!(gguf.tensors()[0].name, "tensor_a");
    assert_eq!(gguf.tensors()[1].name, "tensor_b");

    // Initialize the compute engine
    let engine = match ComputeEngine::new().await {
        Ok(e) => e,
        Err(e) => {
            println!("Skipping GPU test (no GPU available): {}", e);
            return;
        }
    };

    println!("GPU initialized: {:?}", engine.adapter_info().name);

    // Allocate tensors to GPU
    let tensor_buffers = engine
        .allocate_tensors(&gguf)
        .expect("Failed to allocate tensors");

    // Verify we have 2 buffers
    assert_eq!(tensor_buffers.len(), 2);
    assert!(tensor_buffers.contains_key("tensor_a"));
    assert!(tensor_buffers.contains_key("tensor_b"));

    // Verify buffer sizes
    let tensor_a_buffer = tensor_buffers.get("tensor_a").unwrap();
    let tensor_b_buffer = tensor_buffers.get("tensor_b").unwrap();

    assert_eq!(tensor_a_buffer.size(), 16 * 4); // 16 F32 values = 64 bytes
    assert_eq!(tensor_b_buffer.size(), 8 * 4); // 8 F32 values = 32 bytes

    println!("Successfully allocated {} tensors to GPU VRAM", tensor_buffers.len());
}

#[tokio::test]
async fn test_tensor_data_reading() {
    // Create a temporary GGUF file
    let temp_dir = TempDir::new().unwrap();
    let gguf_path = create_test_gguf_file(&temp_dir);

    // Parse the GGUF file
    let gguf = GGUFFile::open(&gguf_path).expect("Failed to open GGUF file");

    // Test reading tensor data
    let tensor_a = &gguf.tensors()[0];
    let tensor_a_data = gguf.get_tensor_data(tensor_a);

    // Verify size
    assert_eq!(tensor_a_data.len(), 64); // 16 F32 = 64 bytes

    // Verify first few values (as little-endian F32)
    let first_value = f32::from_le_bytes([
        tensor_a_data[0],
        tensor_a_data[1],
        tensor_a_data[2],
        tensor_a_data[3],
    ]);
    assert_eq!(first_value, 0.0);

    let second_value = f32::from_le_bytes([
        tensor_a_data[4],
        tensor_a_data[5],
        tensor_a_data[6],
        tensor_a_data[7],
    ]);
    assert_eq!(second_value, 1.0);
}

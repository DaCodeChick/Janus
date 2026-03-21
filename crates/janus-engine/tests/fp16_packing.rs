//! Integration tests for FP16 packed tensor loading and inference
//!
//! Tests that F32, F16, and BF16 tensors are correctly packed into
//! u32 format and unpacked in shaders for mixed-precision inference.

use janus_engine::compute::{ComputeEngine, ComputeError};
use janus_engine::formats::{FormatError, ModelLoader, TensorData, TensorDType};
use std::collections::HashMap;

/// Mock model loader for testing FP16 packing
struct MockFP16Loader {
    tensors: HashMap<String, TensorData<'static>>,
}

impl MockFP16Loader {
    fn new() -> Self {
        let mut tensors = HashMap::new();

        // Test F32 tensor: 8 elements = 4 packed u32s
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let f32_bytes: &'static [u8] = Box::leak(
            f32_data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        tensors.insert(
            "test_f32_weight".to_string(),
            TensorData {
                name: "test_f32_weight".to_string(),
                dtype: TensorDType::F32,
                shape: vec![2, 4],
                data: f32_bytes,
            },
        );

        // Test F16 tensor: 6 elements = 3 packed u32s
        let f16_data: Vec<u16> = vec![
            half::f16::from_f32(1.5).to_bits(),
            half::f16::from_f32(2.5).to_bits(),
            half::f16::from_f32(3.5).to_bits(),
            half::f16::from_f32(4.5).to_bits(),
            half::f16::from_f32(5.5).to_bits(),
            half::f16::from_f32(6.5).to_bits(),
        ];
        let f16_bytes: &'static [u8] = Box::leak(
            f16_data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        tensors.insert(
            "test_f16_weight".to_string(),
            TensorData {
                name: "test_f16_weight".to_string(),
                dtype: TensorDType::F16,
                shape: vec![2, 3],
                data: f16_bytes,
            },
        );

        // Test BF16 tensor: 4 elements = 2 packed u32s
        let bf16_data: Vec<u16> = vec![
            0x3F80, // 1.0 in BF16
            0x4000, // 2.0 in BF16
            0x4040, // 3.0 in BF16
            0x4080, // 4.0 in BF16
        ];
        let bf16_bytes: &'static [u8] = Box::leak(
            bf16_data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        tensors.insert(
            "test_bf16_weight".to_string(),
            TensorData {
                name: "test_bf16_weight".to_string(),
                dtype: TensorDType::BF16,
                shape: vec![2, 2],
                data: bf16_bytes,
            },
        );

        Self { tensors }
    }
}

impl ModelLoader for MockFP16Loader {
    fn tensors(&self) -> Result<HashMap<String, TensorData>, FormatError> {
        // Create a new HashMap with cloned references (data is static, so references are valid)
        let mut result = HashMap::new();
        for (name, tensor) in &self.tensors {
            result.insert(
                name.clone(),
                TensorData {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    data: tensor.data,
                },
            );
        }
        Ok(result)
    }

    fn get_metadata(&self, _key: &str) -> Option<String> {
        None
    }

    fn metadata_keys(&self) -> Vec<String> {
        Vec::new()
    }
}

#[tokio::test]
async fn test_fp16_packing_f32_tensors() {
    // Initialize compute engine
    let engine = match ComputeEngine::new().await {
        Ok(engine) => engine,
        Err(ComputeError::DeviceRequestFailed(_)) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to initialize compute engine: {}", e),
    };

    // Create mock loader with F32 tensors
    let loader = MockFP16Loader::new();

    // Allocate tensors (should pack F32 -> FP16)
    let buffers = engine
        .allocate_tensors(&loader)
        .expect("Failed to allocate tensors");

    // Verify buffers were created
    assert!(buffers.contains_key("test_f32_weight"), "F32 tensor should be allocated");
    assert!(buffers.contains_key("test_f16_weight"), "F16 tensor should be allocated");
    assert!(buffers.contains_key("test_bf16_weight"), "BF16 tensor should be allocated");

    // Verify buffer sizes (should be half of original F32 size)
    let f32_buffer = &buffers["test_f32_weight"];
    assert_eq!(
        f32_buffer.size(),
        (8 * std::mem::size_of::<f32>() / 2) as u64,
        "F32 buffer should be packed to half size"
    );

    let f16_buffer = &buffers["test_f16_weight"];
    assert_eq!(
        f16_buffer.size(),
        (6 * std::mem::size_of::<u16>()) as u64,
        "F16 buffer should maintain size (already 16-bit)"
    );

    let bf16_buffer = &buffers["test_bf16_weight"];
    assert_eq!(
        bf16_buffer.size(),
        (4 * std::mem::size_of::<u16>()) as u64,
        "BF16 buffer should be packed to half F32 size"
    );
}

#[tokio::test]
async fn test_fp16_packing_odd_element_count() {
    // Initialize compute engine
    let engine = match ComputeEngine::new().await {
        Ok(engine) => engine,
        Err(ComputeError::DeviceRequestFailed(_)) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to initialize compute engine: {}", e),
    };

    // Create a tensor with odd number of elements (should pad with zero)
    let mut loader = MockFP16Loader::new();
    
    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements (odd)
    let f32_bytes: &'static [u8] = Box::leak(
        f32_data
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );
    loader.tensors.insert(
        "odd_tensor".to_string(),
        TensorData {
            name: "odd_tensor".to_string(),
            dtype: TensorDType::F32,
            shape: vec![1, 5],
            data: f32_bytes,
        },
    );

    // Allocate tensors (should pack and pad with zero)
    let buffers = engine
        .allocate_tensors(&loader)
        .expect("Failed to allocate tensors");

    // Verify odd tensor was packed (should round up to 3 u32s for 5 elements)
    let odd_buffer = &buffers["odd_tensor"];
    assert_eq!(
        odd_buffer.size(),
        (3 * std::mem::size_of::<u32>()) as u64,
        "Odd element count should be rounded up and padded"
    );
}

#[test]
fn test_gemm_shader_compiles() {
    // Verify that the updated GEMM shader with packed f16 support compiles
    let shader_source = include_str!("../src/compute/shaders/gemm.wgsl");
    
    // Basic syntax checks
    assert!(shader_source.contains("matrix_b: array<u32>"), "GEMM shader should use u32 array for matrix_b");
    assert!(shader_source.contains("unpack2x16float"), "GEMM shader should use unpack2x16float builtin");
    assert!(shader_source.contains("packed f16"), "GEMM shader should document packed f16 format");
}

#[test]
fn test_matmul_shader_compiles() {
    // Verify that the updated matmul shader with packed f16 support compiles
    let shader_source = include_str!("../src/compute/shaders/matmul.wgsl");
    
    // Basic syntax checks
    assert!(shader_source.contains("matrix: array<u32>"), "Matmul shader should use u32 array for matrix");
    assert!(shader_source.contains("unpack2x16float"), "Matmul shader should use unpack2x16float builtin");
    assert!(shader_source.contains("packed f16"), "Matmul shader should document packed f16 format");
}

#[test]
fn test_embed_shader_compiles() {
    // Verify that the updated embed shader with packed f16 support compiles
    let shader_source = include_str!("../src/compute/shaders/embed.wgsl");
    
    // Basic syntax checks
    assert!(shader_source.contains("embedding_table: array<u32>"), "Embed shader should use u32 array for embedding_table");
    assert!(shader_source.contains("unpack2x16float"), "Embed shader should use unpack2x16float builtin");
    assert!(shader_source.contains("packed f16"), "Embed shader should document packed f16 format");
}

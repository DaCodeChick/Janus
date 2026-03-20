//! ComputeEngine for initializing GPU and loading model tensors into VRAM

use super::error::{ComputeError, Result};
use crate::formats::{ModelLoader, TensorDType};
use bytemuck;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// GPU compute engine for LLM inference
pub struct ComputeEngine {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl ComputeEngine {
    /// Initialize a new compute engine with the highest-performance GPU available
    pub async fn new() -> Result<Self> {
        // Create WGPU instance
        let instance = wgpu::Instance::default();

        // Request the highest-performance adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| ComputeError::DeviceRequestFailed(format!("Adapter request failed: {}", e)))?;

        // Log adapter info
        let adapter_info = adapter.get_info();
        tracing::info!(
            "Selected GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("janus_compute_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .map_err(|e: wgpu::RequestDeviceError| ComputeError::DeviceRequestFailed(e.to_string()))?;

        tracing::info!("GPU device initialized successfully");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Get reference to the WGPU instance
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Get reference to the WGPU adapter
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Get reference to the WGPU device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to the WGPU queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Allocate tensors from a model file to GPU buffers
    ///
    /// Accepts any ModelLoader implementation (GGUF, Safetensors, etc.)
    /// and creates GPU buffers for all tensors with zero-copy from mmap.
    ///
    /// # Supported Data Types
    /// - **F32**: Direct zero-copy transfer to GPU
    /// - **Q4_K**: Quantized format, dequantized on-the-fly in shader
    /// - **Other types**: Skipped with warning (BF16, other quantization formats not yet supported)
    pub fn allocate_tensors<L: ModelLoader>(&self, loader: &L) -> Result<HashMap<String, wgpu::Buffer>> {
        let tensors = loader.tensors()
            .map_err(|e| ComputeError::Other(format!("Failed to load tensors: {}", e)))?;
        
        let mut tensor_buffers = HashMap::new();

        tracing::info!("Allocating {} tensors to GPU VRAM", tensors.len());

        let mut total_bytes = 0u64;
        let mut skipped_count = 0;
        let mut f32_count = 0;
        let mut q4k_count = 0;

        for (name, tensor) in tensors {
            match tensor.dtype {
                TensorDType::F32 => {
                    // Direct zero-copy transfer for F32 tensors
                    let size_bytes = tensor.data.len();

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}", name)),
                        contents: tensor.data,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += size_bytes as u64;
                    f32_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (F32)",
                        name,
                        size_bytes
                    );

                    tensor_buffers.insert(name, buffer);
                }

                TensorDType::Q4_K => {
                    // Q4_K: Keep quantized format, dequantize on-the-fly in shader
                    let size_bytes = tensor.data.len();

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q4k", name)),
                        contents: tensor.data,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += size_bytes as u64;
                    q4k_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q4_K quantized, on-the-fly dequantization)",
                        name,
                        size_bytes
                    );

                    tensor_buffers.insert(name, buffer);
                }

                _ => {
                    // Skip unsupported types (other quantized formats will be added later)
                    tracing::warn!(
                        "Skipping tensor '{}' with type {:?} (not yet supported)",
                        name,
                        tensor.dtype
                    );
                    skipped_count += 1;
                }
            }
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "Successfully allocated {} tensors ({:.2} MB) to GPU VRAM: {} F32, {} BF16->F32, {} Q4_K, {} skipped",
            tensor_buffers.len(),
            total_mb,
            f32_direct_count,
            bf16_converted_count,
            q4k_count,
            skipped_count
        );

        Ok(tensor_buffers)
    }

    /// Get adapter information
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Get device limits
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Submit command buffers to the GPU queue
    pub fn submit<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        self.queue.submit(command_buffers);
    }

    /// Wait for all GPU operations to complete
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_initialization() {
        // This test requires a GPU to be available
        let result = ComputeEngine::new().await;
        
        // We can't guarantee GPU availability in CI, so just log the result
        match result {
            Ok(engine) => {
                let info = engine.adapter_info();
                println!("GPU found: {} ({:?})", info.name, info.backend);
                
                let limits = engine.limits();
                println!("Max buffer size: {}", limits.max_buffer_size);
            }
            Err(e) => {
                println!("No GPU available (this is okay in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        // Test BF16 to F32 conversion
        // BF16 is F32 with the lower 16 bits truncated
        
        // Test case 1: 1.0 in BF16
        // F32: 0x3F800000 (binary: 0011_1111_1000_0000_0000_0000_0000_0000)
        // BF16: 0x3F80 (upper 16 bits only)
        let bf16_bytes_1 = vec![0x80, 0x3F]; // Little-endian: 0x3F80
        let f32_result_1 = ComputeEngine::bf16_to_f32(&bf16_bytes_1);
        assert_eq!(f32_result_1.len(), 1);
        assert_eq!(f32_result_1[0], 1.0);

        // Test case 2: 2.0 in BF16
        // F32: 0x40000000
        // BF16: 0x4000
        let bf16_bytes_2 = vec![0x00, 0x40]; // Little-endian: 0x4000
        let f32_result_2 = ComputeEngine::bf16_to_f32(&bf16_bytes_2);
        assert_eq!(f32_result_2.len(), 1);
        assert_eq!(f32_result_2[0], 2.0);

        // Test case 3: 0.5 in BF16
        // F32: 0x3F000000
        // BF16: 0x3F00
        let bf16_bytes_3 = vec![0x00, 0x3F]; // Little-endian: 0x3F00
        let f32_result_3 = ComputeEngine::bf16_to_f32(&bf16_bytes_3);
        assert_eq!(f32_result_3.len(), 1);
        assert_eq!(f32_result_3[0], 0.5);

        // Test case 4: Multiple values
        let bf16_bytes_multi = vec![
            0x80, 0x3F, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3F, // 0.5
        ];
        let f32_result_multi = ComputeEngine::bf16_to_f32(&bf16_bytes_multi);
        assert_eq!(f32_result_multi.len(), 3);
        assert_eq!(f32_result_multi[0], 1.0);
        assert_eq!(f32_result_multi[1], 2.0);
        assert_eq!(f32_result_multi[2], 0.5);
    }

    #[test]
    fn test_bf16_to_f32_negative_values() {
        // Test negative values
        // -1.0 in F32: 0xBF800000
        // -1.0 in BF16: 0xBF80
        let bf16_bytes = vec![0x80, 0xBF]; // Little-endian: 0xBF80
        let f32_result = ComputeEngine::bf16_to_f32(&bf16_bytes);
        assert_eq!(f32_result.len(), 1);
        assert_eq!(f32_result[0], -1.0);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        // Test zero
        let bf16_bytes = vec![0x00, 0x00]; // 0.0
        let f32_result = ComputeEngine::bf16_to_f32(&bf16_bytes);
        assert_eq!(f32_result.len(), 1);
        assert_eq!(f32_result[0], 0.0);
    }
}

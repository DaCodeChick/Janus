//! ComputeEngine for initializing GPU and loading model tensors into VRAM

use super::error::{ComputeError, Result};
use crate::formats::{ModelLoader, TensorDType};
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

        // Get adapter limits and use them for the device
        let adapter_limits = adapter.limits();
        tracing::info!(
            "Adapter limits: max_storage_buffer_binding_size={}, max_buffer_size={}",
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_buffer_size
        );

        // Request device and queue with maximum available limits
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("janus_compute_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .map_err(|e: wgpu::RequestDeviceError| ComputeError::DeviceRequestFailed(e.to_string()))?;

        tracing::info!(
            "GPU device initialized with max_storage_buffer_binding_size={}",
            device.limits().max_storage_buffer_binding_size
        );

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

    /// Convert BF16 bytes to F32 on CPU before GPU upload
    ///
    /// BF16 (Brain Floating Point) is F32 with the lower 16 bits truncated.
    /// To convert: take the 16-bit BF16 value and pad with 16 zero bits.
    fn bf16_to_f32(bf16_data: &[u8]) -> Vec<f32> {
        let num_elements = bf16_data.len() / 2;
        let mut f32_data = Vec::with_capacity(num_elements);

        for i in 0..num_elements {
            // Read BF16 as u16 (little-endian)
            let bf16 = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
            
            // Convert to F32 by padding with 16 zero bits
            // BF16: [sign bit][8 exp bits][7 mantissa bits]
            // F32:  [sign bit][8 exp bits][23 mantissa bits]
            let f32_bits = (bf16 as u32) << 16;
            let f32_value = f32::from_bits(f32_bits);
            
            f32_data.push(f32_value);
        }

        f32_data
    }

    /// Allocate tensors from a model file to GPU buffers
    ///
    /// Accepts any ModelLoader implementation (GGUF, Safetensors, etc.)
    /// and creates GPU buffers for all tensors with zero-copy from mmap.
    ///
    /// # Supported Data Types
    /// - **F32**: Direct zero-copy transfer to GPU
    /// - **BF16**: Converted to F32 on CPU, then uploaded to GPU
    /// - **Q4_K**: Quantized format, dequantized on-the-fly in shader
    /// - **Other types**: Skipped with warning
    pub fn allocate_tensors<L: ModelLoader>(&self, loader: &L) -> Result<HashMap<String, wgpu::Buffer>> {
        let tensors = loader.tensors()
            .map_err(|e| ComputeError::Other(format!("Failed to load tensors: {}", e)))?;
        
        let mut tensor_buffers = HashMap::new();

        tracing::info!("Allocating {} tensors to GPU VRAM", tensors.len());

        let mut total_bytes = 0u64;
        let mut skipped_count = 0;
        let mut f32_count = 0;
        let mut bf16_count = 0;
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

                TensorDType::BF16 => {
                    // Convert BF16 to F32 on CPU, then upload
                    let f32_data = Self::bf16_to_f32(tensor.data);
                    let f32_bytes = bytemuck::cast_slice(&f32_data);
                    
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_bf16_to_f32", name)),
                        contents: f32_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += f32_bytes.len() as u64;
                    bf16_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (BF16 -> F32, {} elements)",
                        name,
                        f32_bytes.len(),
                        f32_data.len()
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
            f32_count,
            bf16_count,
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
}

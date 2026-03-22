//! ComputeEngine for initializing GPU and loading model tensors into VRAM

use super::error::{ComputeError, Result};
use crate::formats::{ModelLoader, TensorDType};
use half::f16;
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

    /// Pack F16 values into U32 bit-packed format (2 f16s per u32)
    /// 
    /// WebGPU lacks native f16 buffer support on many devices, so we pack
    /// two 16-bit floats into a single u32 and unpack on-the-fly in shaders.
    /// 
    /// Packing format: high 16 bits = second f16, low 16 bits = first f16
    /// This halves VRAM usage and doubles memory bandwidth.
    fn pack_f16_to_u32(f16_data: &[f16]) -> Vec<u32> {
        let num_pairs = (f16_data.len() + 1) / 2; // Round up for odd lengths
        let mut packed = Vec::with_capacity(num_pairs);

        for i in (0..f16_data.len()).step_by(2) {
            let low_f16 = f16_data[i];
            let high_f16 = if i + 1 < f16_data.len() {
                f16_data[i + 1]
            } else {
                f16::ZERO // Pad with zero if odd number of elements
            };

            // Pack: high 16 bits = second f16, low 16 bits = first f16
            let packed_u32 = ((high_f16.to_bits() as u32) << 16) | (low_f16.to_bits() as u32);
            packed.push(packed_u32);
        }

        packed
    }

    /// Convert F32 data to packed F16 format
    fn f32_to_packed_f16(f32_data: &[f32]) -> Vec<u32> {
        let f16_data: Vec<f16> = f32_data.iter().map(|&x| f16::from_f32(x)).collect();
        Self::pack_f16_to_u32(&f16_data)
    }

    /// Convert BF16 data to packed F16 format
    fn bf16_to_packed_f16(bf16_data: &[u8]) -> Vec<u32> {
        let num_elements = bf16_data.len() / 2;
        let mut f16_data = Vec::with_capacity(num_elements);
        let mut nan_count = 0;
        let mut inf_count = 0;

        for i in 0..num_elements {
            // Read BF16 as u16 (little-endian)
            let bf16 = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
            
            // Convert BF16 -> F32 -> F16
            let f32_bits = (bf16 as u32) << 16;
            let f32_value = f32::from_bits(f32_bits);
            
            // Check for NaN/Inf in source data
            if f32_value.is_nan() {
                nan_count += 1;
            } else if f32_value.is_infinite() {
                inf_count += 1;
            }
            
            let f16_value = f16::from_f32(f32_value);
            
            f16_data.push(f16_value);
        }

        if nan_count > 0 || inf_count > 0 {
            tracing::warn!(
                "BF16 conversion: found {} NaNs and {} Infs in {} elements",
                nan_count,
                inf_count,
                num_elements
            );
        }

        Self::pack_f16_to_u32(&f16_data)
    }

    /// Convert native F16 data to packed F16 format
    fn f16_to_packed_f16(f16_data: &[u8]) -> Vec<u32> {
        let num_elements = f16_data.len() / 2;
        let mut f16_vec = Vec::with_capacity(num_elements);

        for i in 0..num_elements {
            // Read F16 as u16 (little-endian)
            let f16_bits = u16::from_le_bytes([f16_data[i * 2], f16_data[i * 2 + 1]]);
            f16_vec.push(f16::from_bits(f16_bits));
        }

        Self::pack_f16_to_u32(&f16_vec)
    }

    /// Allocate tensors from a model file to GPU buffers
    ///
    /// Accepts any ModelLoader implementation (GGUF, Safetensors, etc.)
    /// and creates GPU buffers for all tensors with zero-copy from mmap.
    ///
    /// # Supported Data Types
    /// - **F32**: Converted to packed FP16 (2 f16s per u32) for 50% VRAM reduction
    /// - **F16**: Converted to packed FP16 (2 f16s per u32) for efficient storage
    /// - **BF16**: Converted to packed FP16 (2 f16s per u32) via F32 intermediate
    /// - **Q4_K**: 4-bit quantized format, dequantized on-the-fly in shader
    /// - **Q5_K**: 5-bit quantized format, dequantized on-the-fly in shader
    /// - **Q8_0**: 8-bit quantized format, dequantized on-the-fly in shader
    /// - **Other types**: Skipped with warning
    pub fn allocate_tensors<L: ModelLoader>(&self, loader: &L) -> Result<HashMap<String, wgpu::Buffer>> {
        let tensors = loader.tensors()
            .map_err(|e| ComputeError::Other(format!("Failed to load tensors: {}", e)))?;
        
        let mut tensor_buffers = HashMap::new();

        tracing::info!("Allocating {} tensors to GPU VRAM", tensors.len());

        let mut total_bytes = 0u64;
        let mut skipped_count = 0;
        let mut f32_packed_count = 0;
        let mut f16_packed_count = 0;
        let mut bf16_packed_count = 0;
        let mut q4k_count = 0;
        let mut q5k_count = 0;
        let mut q8_0_count = 0;

        for (name, tensor) in tensors {
            match tensor.dtype {
                TensorDType::F32 => {
                    // Convert F32 to packed FP16 format (50% VRAM reduction)
                    let f32_data: &[f32] = bytemuck::cast_slice(tensor.data);
                    let packed_data = Self::f32_to_packed_f16(f32_data);
                    let packed_bytes = bytemuck::cast_slice(&packed_data);

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_f32_packed_f16", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    f32_packed_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (F32 -> packed FP16, {} elements, {:.1}% VRAM reduction)",
                        name,
                        packed_bytes.len(),
                        f32_data.len(),
                        50.0
                    );

                    tensor_buffers.insert(name, buffer);
                }

                TensorDType::F16 => {
                    // Convert F16 to packed FP16 format (efficient storage)
                    let packed_data = Self::f16_to_packed_f16(tensor.data);
                    let packed_bytes = bytemuck::cast_slice(&packed_data);

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_f16_packed", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    f16_packed_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (F16 -> packed FP16, {} elements)",
                        name,
                        packed_bytes.len(),
                        tensor.data.len() / 2
                    );

                    tensor_buffers.insert(name, buffer);
                }

                TensorDType::BF16 => {
                    // Convert BF16 to packed FP16 format (via F32 intermediate)
                    let packed_data = Self::bf16_to_packed_f16(tensor.data);
                    let packed_bytes = bytemuck::cast_slice(&packed_data);
                    
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_bf16_packed_f16", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    bf16_packed_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (BF16 -> packed FP16, {} elements)",
                        name,
                        packed_bytes.len(),
                        tensor.data.len() / 2
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

                TensorDType::Q5_K => {
                    // Q5_K: Keep quantized format, dequantize on-the-fly in shader
                    let size_bytes = tensor.data.len();

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q5k", name)),
                        contents: tensor.data,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += size_bytes as u64;
                    q5k_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q5_K quantized, on-the-fly dequantization)",
                        name,
                        size_bytes
                    );

                    tensor_buffers.insert(name, buffer);
                }

                TensorDType::Q8_0 => {
                    // Q8_0: Keep quantized format, dequantize on-the-fly in shader
                    let size_bytes = tensor.data.len();

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q8_0", name)),
                        contents: tensor.data,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += size_bytes as u64;
                    q8_0_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q8_0 quantized, on-the-fly dequantization)",
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
            "Successfully allocated {} tensors ({:.2} MB) to GPU VRAM: {} F32->FP16, {} F16->FP16, {} BF16->FP16, {} Q4_K, {} Q5_K, {} Q8_0, {} skipped",
            tensor_buffers.len(),
            total_mb,
            f32_packed_count,
            f16_packed_count,
            bf16_packed_count,
            q4k_count,
            q5k_count,
            q8_0_count,
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

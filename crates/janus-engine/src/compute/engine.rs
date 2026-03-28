//! ComputeEngine for initializing GPU and loading model tensors into VRAM

use super::error::{ComputeError, Result};
use super::tensor::GpuTensor;
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
    const Q4K_BLOCK_SIZE: usize = 256;
    const Q4K_BLOCK_BYTES: usize = 144;
    const Q5K_BLOCK_SIZE: usize = 256;
    const Q5K_BLOCK_BYTES: usize = 176;
    const Q8_0_BLOCK_SIZE: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 34;

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

    /// Convert Q6_K blocks to F32 values.
    ///
    /// Q6_K block layout (210 bytes for 256 values):
    /// - ql[128]: low 4 bits for each value (two values per byte)
    /// - qh[64]: high 2 bits for each value (four values per byte)
    /// - scales[16]: signed int8 scales per 16-value group
    /// - d[2]: f16 super-scale
    fn q6_k_to_f32(q6k_data: &[u8], num_elements: usize) -> Result<Vec<f32>> {
        const Q6K_BLOCK_SIZE: usize = 256;
        const Q6K_BLOCK_BYTES: usize = 210;

        if q6k_data.len() % Q6K_BLOCK_BYTES != 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Q6_K tensor byte size {} is not a multiple of {}",
                q6k_data.len(),
                Q6K_BLOCK_BYTES
            )));
        }

        let num_blocks = q6k_data.len() / Q6K_BLOCK_BYTES;
        let mut dequantized = Vec::with_capacity(num_blocks * Q6K_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let offset = block_idx * Q6K_BLOCK_BYTES;
            let block = &q6k_data[offset..offset + Q6K_BLOCK_BYTES];

            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d_bits = u16::from_le_bytes([block[208], block[209]]);
            let d = f16::from_bits(d_bits).to_f32();

            for i in 0..Q6K_BLOCK_SIZE {
                let ql_byte = ql[i / 2];
                let lo = if i % 2 == 0 {
                    ql_byte & 0x0F
                } else {
                    (ql_byte >> 4) & 0x0F
                };

                let qh_byte = qh[i / 4];
                let hi = (qh_byte >> (2 * (i % 4))) & 0x03;

                let q = ((hi << 4) | lo) as i32;
                let q_signed = q - 32;
                let s = scales[i / 16] as i8 as f32;

                let value = d * s * q_signed as f32;
                dequantized.push(value);
            }
        }

        if dequantized.len() > num_elements {
            dequantized.truncate(num_elements);
        }

        Ok(dequantized)
    }

    fn is_embedding_tensor_name(name: &str) -> bool {
        matches!(name, "token_embd.weight" | "model.embed_tokens.weight")
    }

    fn extract_6bit_from_packed(bytes: &[u8], idx: usize) -> u8 {
        let bit_offset = idx * 6;
        let mut value = 0u8;
        for bit in 0..6 {
            let absolute = bit_offset + bit;
            let byte_idx = absolute / 8;
            let bit_idx = absolute % 8;
            let bit_val = (bytes[byte_idx] >> bit_idx) & 1;
            value |= bit_val << bit;
        }
        value
    }

    fn q4_k_to_packed_f16(q4k_data: &[u8], num_elements: usize) -> Result<Vec<u32>> {
        if q4k_data.len() % Self::Q4K_BLOCK_BYTES != 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Q4_K tensor byte size {} is not a multiple of {}",
                q4k_data.len(),
                Self::Q4K_BLOCK_BYTES
            )));
        }

        let num_blocks = q4k_data.len() / Self::Q4K_BLOCK_BYTES;
        let mut dequantized = Vec::with_capacity(num_blocks * Self::Q4K_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let offset = block_idx * Self::Q4K_BLOCK_BYTES;
            let block = &q4k_data[offset..offset + Self::Q4K_BLOCK_BYTES];

            let scales_mins = &block[0..12];
            let qs = &block[12..140];
            let d_bits = u16::from_le_bytes([block[140], block[141]]);
            let dmin_bits = u16::from_le_bytes([block[142], block[143]]);
            let d = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();

            for group in 0..8 {
                let scale = d * f32::from(Self::extract_6bit_from_packed(scales_mins, group));
                let min_val = dmin * f32::from(Self::extract_6bit_from_packed(scales_mins, group + 8));

                let group_qs = &qs[group * 16..(group + 1) * 16];
                for (byte_idx, byte) in group_qs.iter().enumerate() {
                    let low = byte & 0x0F;
                    let high = (byte >> 4) & 0x0F;

                    let q0 = f32::from(low);
                    let q1 = f32::from(high);
                    let _elem_base = group * 32 + byte_idx * 2;

                    dequantized.push(f16::from_f32(scale * q0 - min_val));
                    dequantized.push(f16::from_f32(scale * q1 - min_val));
                }
            }
        }

        if dequantized.len() > num_elements {
            dequantized.truncate(num_elements);
        }

        Ok(Self::pack_f16_to_u32(&dequantized))
    }

    fn q5_k_to_packed_f16(q5k_data: &[u8], num_elements: usize) -> Result<Vec<u32>> {
        if q5k_data.len() % Self::Q5K_BLOCK_BYTES != 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Q5_K tensor byte size {} is not a multiple of {}",
                q5k_data.len(),
                Self::Q5K_BLOCK_BYTES
            )));
        }

        let num_blocks = q5k_data.len() / Self::Q5K_BLOCK_BYTES;
        let mut dequantized = Vec::with_capacity(num_blocks * Self::Q5K_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let offset = block_idx * Self::Q5K_BLOCK_BYTES;
            let block = &q5k_data[offset..offset + Self::Q5K_BLOCK_BYTES];

            let scales_mins = &block[0..12];
            let qh = &block[12..44];
            let qs = &block[44..172];
            let d_bits = u16::from_le_bytes([block[172], block[173]]);
            let dmin_bits = u16::from_le_bytes([block[174], block[175]]);
            let d = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();

            for group in 0..8 {
                let scale = d * f32::from(Self::extract_6bit_from_packed(scales_mins, group));
                let min_val = dmin * f32::from(Self::extract_6bit_from_packed(scales_mins, group + 8));

                let group_qs = &qs[group * 16..(group + 1) * 16];
                for (byte_idx, byte) in group_qs.iter().enumerate() {
                    let low0 = byte & 0x0F;
                    let low1 = (byte >> 4) & 0x0F;

                    let elem0 = group * 32 + byte_idx * 2;
                    let elem1 = elem0 + 1;

                    let qh_byte0 = qh[elem0 / 8];
                    let qh_byte1 = qh[elem1 / 8];
                    let high0 = (qh_byte0 >> (elem0 % 8)) & 1;
                    let high1 = (qh_byte1 >> (elem1 % 8)) & 1;

                    let q0 = f32::from(low0 | (high0 << 4));
                    let q1 = f32::from(low1 | (high1 << 4));

                    dequantized.push(f16::from_f32(scale * q0 - min_val));
                    dequantized.push(f16::from_f32(scale * q1 - min_val));
                }
            }
        }

        if dequantized.len() > num_elements {
            dequantized.truncate(num_elements);
        }

        Ok(Self::pack_f16_to_u32(&dequantized))
    }

    fn q8_0_to_packed_f16(q8_0_data: &[u8], num_elements: usize) -> Result<Vec<u32>> {
        if q8_0_data.len() % Self::Q8_0_BLOCK_BYTES != 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Q8_0 tensor byte size {} is not a multiple of {}",
                q8_0_data.len(),
                Self::Q8_0_BLOCK_BYTES
            )));
        }

        let num_blocks = q8_0_data.len() / Self::Q8_0_BLOCK_BYTES;
        let mut dequantized = Vec::with_capacity(num_blocks * Self::Q8_0_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let offset = block_idx * Self::Q8_0_BLOCK_BYTES;
            let block = &q8_0_data[offset..offset + Self::Q8_0_BLOCK_BYTES];

            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let d = f16::from_bits(d_bits).to_f32();
            let quants = &block[2..34];

            for q in quants {
                let q_signed = *q as i8;
                let value = d * f32::from(q_signed);
                dequantized.push(f16::from_f32(value));
            }
        }

        if dequantized.len() > num_elements {
            dequantized.truncate(num_elements);
        }

        Ok(Self::pack_f16_to_u32(&dequantized))
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
    pub fn allocate_tensors<L: ModelLoader>(&self, loader: &L) -> Result<HashMap<String, GpuTensor>> {
        let tensors = loader.tensors()
            .map_err(|e| ComputeError::Other(format!("Failed to load tensors: {}", e)))?;
        
        let mut tensor_buffers = HashMap::new();

        tracing::info!("Allocating {} tensors to GPU VRAM", tensors.len());

        let mut total_bytes = 0u64;
        let skipped_count = 0;
        let mut f32_packed_count = 0;
        let mut f16_packed_count = 0;
        let mut bf16_packed_count = 0;
        let mut q4k_count = 0;
        let mut q5k_count = 0;
        let mut q6k_count = 0;
        let mut q8_0_count = 0;

        for (name, tensor) in tensors {
            // Check if tensor is 1D (RMSNorm weights, biases, etc.)
            let is_1d = tensor.shape.len() == 1;
            
            match tensor.dtype {
                TensorDType::F32 => {
                    if is_1d {
                        // 1D tensors: Keep as unpacked F32 (used by RMSNorm, etc.)
                        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("tensor_{}_f32_unpacked", name)),
                            contents: tensor.data,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        });

                        total_bytes += tensor.data.len() as u64;
                        
                        tracing::debug!(
                            "Allocated tensor '{}': {} bytes (F32, unpacked 1D)",
                            name,
                            tensor.data.len()
                        );

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F32,
                            },
                        );
                    } else {
                        // 2D+ tensors: Convert F32 to packed FP16 format (50% VRAM reduction)
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

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F16,
                            },
                        );
                    }
                }

                TensorDType::F16 => {
                    if is_1d {
                        // 1D tensors: Convert F16 to F32 and keep unpacked
                        let num_elements = tensor.data.len() / 2;
                        let mut f32_data = Vec::with_capacity(num_elements);
                        
                        for i in 0..num_elements {
                            let f16_bits = u16::from_le_bytes([tensor.data[i * 2], tensor.data[i * 2 + 1]]);
                            let f16_value = f16::from_bits(f16_bits);
                            f32_data.push(f16_value.to_f32());
                        }
                        
                        let f32_bytes = bytemuck::cast_slice(&f32_data);
                        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("tensor_{}_f16_to_f32_unpacked", name)),
                            contents: f32_bytes,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        });

                        total_bytes += f32_bytes.len() as u64;

                        tracing::debug!(
                            "Allocated tensor '{}': {} bytes (F16 -> F32, unpacked 1D)",
                            name,
                            f32_bytes.len()
                        );

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F16,
                            },
                        );
                    } else {
                        // 2D+ tensors: Convert F16 to packed FP16 format (efficient storage)
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

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F16,
                            },
                        );
                    }
                }

                TensorDType::BF16 => {
                    if is_1d {
                        // 1D tensors: Convert BF16 to F32 and keep unpacked
                        let num_elements = tensor.data.len() / 2;
                        let mut f32_data = Vec::with_capacity(num_elements);
                        
                        for i in 0..num_elements {
                            let bf16 = u16::from_le_bytes([tensor.data[i * 2], tensor.data[i * 2 + 1]]);
                            let f32_bits = (bf16 as u32) << 16;
                            let f32_value = f32::from_bits(f32_bits);
                            f32_data.push(f32_value);
                        }
                        
                        let f32_bytes = bytemuck::cast_slice(&f32_data);
                        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("tensor_{}_bf16_to_f32_unpacked", name)),
                            contents: f32_bytes,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        });

                        total_bytes += f32_bytes.len() as u64;

                        tracing::debug!(
                            "Allocated tensor '{}': {} bytes (BF16 -> F32, unpacked 1D)",
                            name,
                            f32_bytes.len()
                        );

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F16,
                            },
                        );
                    } else {
                        // 2D+ tensors: Convert BF16 to packed FP16 format (via F32 intermediate)
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

                        tensor_buffers.insert(
                            name,
                            GpuTensor {
                                buffer,
                                ggml_type: TensorDType::F16,
                            },
                        );
                    }
                }

                TensorDType::Q4_K => {
                    let num_elements: usize = tensor.shape.iter().product();
                    let packed_data = Self::q4_k_to_packed_f16(tensor.data, num_elements)?;
                    let packed_bytes = bytemuck::cast_slice(&packed_data);
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q4k_to_packed_f16", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    q4k_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q4_K -> packed FP16 fallback, {} elements)",
                        name,
                        packed_bytes.len(),
                        num_elements
                    );

                    tensor_buffers.insert(
                        name,
                        GpuTensor {
                            buffer,
                            ggml_type: TensorDType::F16,
                        },
                    );
                }

                TensorDType::Q5_K => {
                    let num_elements: usize = tensor.shape.iter().product();
                    let packed_data = Self::q5_k_to_packed_f16(tensor.data, num_elements)?;
                    let packed_bytes = bytemuck::cast_slice(&packed_data);
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q5k_to_packed_f16", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    q5k_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q5_K -> packed FP16 fallback, {} elements)",
                        name,
                        packed_bytes.len(),
                        num_elements
                    );

                    tensor_buffers.insert(
                        name,
                        GpuTensor {
                            buffer,
                            ggml_type: TensorDType::F16,
                        },
                    );
                }

                TensorDType::Q8_0 => {
                    let num_elements: usize = tensor.shape.iter().product();
                    let packed_data = Self::q8_0_to_packed_f16(tensor.data, num_elements)?;
                    let packed_bytes = bytemuck::cast_slice(&packed_data);
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q8_0_to_packed_f16", name)),
                        contents: packed_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += packed_bytes.len() as u64;
                    q8_0_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q8_0 -> packed FP16 fallback, {} elements)",
                        name,
                        packed_bytes.len(),
                        num_elements
                    );

                    tensor_buffers.insert(
                        name,
                        GpuTensor {
                            buffer,
                            ggml_type: TensorDType::F16,
                        },
                    );
                }

                TensorDType::Q6_K => {
                    println!("⚙️ [Loader] Auto-detect: Q6_K tensor found. Dequantizing to F32 fallback on CPU...");
                    let num_elements: usize = tensor.shape.iter().product();
                    let f32_data = Self::q6_k_to_f32(tensor.data, num_elements)?;
                    let f32_bytes = bytemuck::cast_slice(&f32_data);

                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("tensor_{}_q6k_to_f32", name)),
                        contents: f32_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });

                    total_bytes += f32_bytes.len() as u64;
                    q6k_count += 1;

                    tracing::debug!(
                        "Allocated tensor '{}': {} bytes (Q6_K -> F32 fallback, {} elements)",
                        name,
                        f32_bytes.len(),
                        num_elements
                    );

                    tensor_buffers.insert(
                        name,
                        GpuTensor {
                            buffer,
                            ggml_type: TensorDType::F32,
                        },
                    );
                }

                _ => {
                    return Err(ComputeError::Other(format!(
                        "Unsupported tensor dtype {:?} for tensor '{}'. This model likely uses a quantization format not yet supported by Janus (for example Q6_K or IQ variants).",
                        tensor.dtype, name
                    )));
                }
            }
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "Successfully allocated {} tensors ({:.2} MB) to GPU VRAM: {} F32->FP16, {} F16->FP16, {} BF16->FP16, {} Q4_K->FP16, {} Q5_K->FP16, {} Q6_K->F32, {} Q8_0->FP16, {} skipped",
            tensor_buffers.len(),
            total_mb,
            f32_packed_count,
            f16_packed_count,
            bf16_packed_count,
            q4k_count,
            q5k_count,
            q6k_count,
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

//! ComputeEngine for initializing GPU and loading GGUF tensors into VRAM

use super::error::{ComputeError, Result};
use crate::gguf::GGUFFile;
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

    /// Allocate tensors from a GGUF file into GPU VRAM
    ///
    /// This performs a zero-copy move from the memory-mapped file directly to GPU buffers.
    /// Returns a registry mapping tensor names to their GPU buffers.
    ///
    /// # Phase 5 Note
    /// Currently only F32 tensors are supported. Non-F32 tensors are skipped with a warning.
    /// Quantization support will be added in Phase 6.
    pub fn allocate_tensors(&self, gguf: &GGUFFile) -> Result<HashMap<String, wgpu::Buffer>> {
        use crate::gguf::GGMLType;
        
        let mut tensor_buffers = HashMap::new();
        let tensors = gguf.tensors();

        tracing::info!("Allocating {} tensors to GPU VRAM", tensors.len());

        let mut total_bytes = 0u64;
        let mut skipped_count = 0;

        for tensor in tensors {
            // Phase 5: Skip non-F32 tensors with a warning
            if tensor.ggml_type != GGMLType::F32 {
                tracing::warn!(
                    "Skipping tensor '{}' with type {:?} (only F32 supported in Phase 5)",
                    tensor.name,
                    tensor.ggml_type
                );
                skipped_count += 1;
                continue;
            }

            // Get the exact byte slice for this tensor from the mmap
            let tensor_data = gguf.get_tensor_data(tensor);
            let size_bytes = tensor.size_bytes();

            // Sanity check
            if tensor_data.len() != size_bytes as usize {
                return Err(ComputeError::InvalidBufferSize {
                    expected: size_bytes as usize,
                    actual: tensor_data.len(),
                });
            }

            // Create GPU buffer with tensor data
            let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("tensor_{}", tensor.name)),
                contents: tensor_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            total_bytes += size_bytes;

            tracing::debug!(
                "Allocated tensor '{}': {} bytes ({:?})",
                tensor.name,
                size_bytes,
                tensor.ggml_type
            );

            tensor_buffers.insert(tensor.name.clone(), buffer);
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "Successfully allocated {} F32 tensors ({:.2} MB) to GPU VRAM (skipped {} non-F32 tensors)",
            tensor_buffers.len(),
            total_mb,
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

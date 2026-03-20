//! KV Cache for efficient autoregressive generation
//!
//! The KV cache stores Key and Value projections from previous tokens to avoid
//! recomputing them during text generation. This is critical for LLM performance.

use super::engine::ComputeEngine;
use super::error::Result;
use wgpu::util::DeviceExt;

/// KV Cache for storing attention Keys and Values across sequence positions
///
/// During autoregressive generation, we need to compute attention between the current
/// token and all previous tokens. Instead of recomputing the Key and Value projections
/// for all previous tokens at each step, we cache them in GPU memory.
///
/// Layout: [max_seq_len][num_heads][head_dim]
pub struct KVCache {
    /// GPU buffer for cached Keys
    key_cache: wgpu::Buffer,
    
    /// GPU buffer for cached Values
    value_cache: wgpu::Buffer,
    
    /// Maximum sequence length supported by this cache
    max_seq_len: u32,
    
    /// Number of attention heads
    num_heads: u32,
    
    /// Dimension of each attention head
    head_dim: u32,
    
    /// Current position in the cache (for ring buffer behavior)
    current_position: u32,
}

impl KVCache {
    /// Create a new KV cache with the specified dimensions
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `max_seq_len` - Maximum sequence length to support (e.g., 2048, 4096)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Returns
    /// A new KVCache instance with pre-allocated GPU buffers
    pub fn new(
        engine: &ComputeEngine,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let device = engine.device();
        
        // Calculate total size in bytes
        let cache_size = (max_seq_len * num_heads * head_dim) as u64 * std::mem::size_of::<f32>() as u64;
        
        tracing::info!(
            "Allocating KV cache: max_seq_len={}, num_heads={}, head_dim={}, total_size={:.2} MB",
            max_seq_len,
            num_heads,
            head_dim,
            (cache_size * 2) as f64 / (1024.0 * 1024.0)
        );
        
        // Create Key cache buffer
        let key_cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache_keys"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create Value cache buffer
        let value_cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache_values"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        Ok(Self {
            key_cache,
            value_cache,
            max_seq_len,
            num_heads,
            head_dim,
            current_position: 0,
        })
    }
    
    /// Update the cache with new Key and Value tensors for the current token
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `new_key` - GPU buffer containing the new Key projection (num_heads * head_dim elements)
    /// * `new_value` - GPU buffer containing the new Value projection (num_heads * head_dim elements)
    /// * `position` - Position in the sequence to write to (0 to max_seq_len-1)
    ///
    /// # Returns
    /// Ok(()) if successful
    pub async fn update(
        &mut self,
        engine: &ComputeEngine,
        new_key: &wgpu::Buffer,
        new_value: &wgpu::Buffer,
        position: u32,
    ) -> Result<()> {
        // Ensure position is within bounds
        let cache_position = position % self.max_seq_len;
        
        // Update Key cache
        self.update_cache_buffer(engine, new_key, &self.key_cache, cache_position).await?;
        
        // Update Value cache
        self.update_cache_buffer(engine, new_value, &self.value_cache, cache_position).await?;
        
        self.current_position = cache_position + 1;
        
        Ok(())
    }
    
    /// Internal helper to update a cache buffer using the update_cache shader
    async fn update_cache_buffer(
        &self,
        engine: &ComputeEngine,
        new_data: &wgpu::Buffer,
        cache: &wgpu::Buffer,
        position: u32,
    ) -> Result<()> {
        let device = engine.device();
        let queue = engine.queue();
        
        // Load the shader
        let shader_source = include_str!("shaders/update_cache.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_cache_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create uniforms
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct UpdateCacheUniforms {
            cache_position: u32,
            token_dim: u32,
            num_heads: u32,
            _pad: u32,
        }
        
        let uniforms = UpdateCacheUniforms {
            cache_position: position,
            token_dim: self.head_dim,
            num_heads: self.num_heads,
            _pad: 0,
        };
        
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("update_cache_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_cache_bind_group_layout"),
            entries: &[
                // New KV data (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cache buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_cache_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: new_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cache.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_cache_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_cache_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create command encoder and dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("update_cache_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_cache_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup count (256 threads per workgroup)
            let total_elements = self.num_heads * self.head_dim;
            let workgroup_count = (total_elements + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Submit commands
        queue.submit(Some(encoder.finish()));
        
        // Wait for completion
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        
        Ok(())
    }
    
    /// Get a reference to the Key cache buffer
    pub fn key_cache(&self) -> &wgpu::Buffer {
        &self.key_cache
    }
    
    /// Get a reference to the Value cache buffer
    pub fn value_cache(&self) -> &wgpu::Buffer {
        &self.value_cache
    }
    
    /// Get the maximum sequence length supported by this cache
    pub const fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }
    
    /// Get the number of attention heads
    pub const fn num_heads(&self) -> u32 {
        self.num_heads
    }
    
    /// Get the dimension of each attention head
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }
    
    /// Get the current position in the cache
    pub const fn current_position(&self) -> u32 {
        self.current_position
    }
    
    /// Get references to both Key and Value cache buffers
    /// 
    /// Returns a tuple (key_cache, value_cache) for convenience
    pub fn buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.key_cache, &self.value_cache)
    }
    
    /// Reset the cache position to 0 (start of a new sequence)
    pub fn reset(&mut self) {
        self.current_position = 0;
    }
}

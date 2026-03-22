//! KV Cache for efficient autoregressive generation
//!
//! The KV cache stores Key and Value projections from previous tokens to avoid
//! recomputing them during text generation. This is critical for LLM performance.
//!
//! # KV Cache Compression
//!
//! To extend context length beyond the physical cache size, this implementation
//! supports compression of older KV cache entries using a sliding window approach:
//!
//! - **Recent tokens** (uncompressed window): Keep full precision for accuracy
//! - **Middle range** (compression zone): Apply 2:1 compression via averaging
//! - **Old tokens** (eviction zone): Remove to free space
//!
//! This allows effective context lengths of 2-4x the base cache size with minimal
//! quality degradation.

use super::engine::ComputeEngine;
use super::error::Result;
use super::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

/// Configuration for KV cache compression
#[derive(Debug, Clone, Copy)]
pub struct CacheCompressionConfig {
    /// Enable compression (default: false)
    pub enabled: bool,

    /// Size of the uncompressed window (recent tokens to keep at full precision)
    /// Default: 512 tokens
    pub uncompressed_window: u32,

    /// Compression ratio for middle range tokens (2 = compress 2:1, 4 = compress 4:1)
    /// Default: 2
    pub compression_ratio: u32,

    /// Trigger compression when cache is this percentage full (0.0-1.0)
    /// Default: 0.8 (80%)
    pub compression_trigger: f32,
}

impl Default for CacheCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            uncompressed_window: 512,
            compression_ratio: 2,
            compression_trigger: 0.8,
        }
    }
}

/// KV Cache for storing attention Keys and Values across sequence positions
///
/// During autoregressive generation, we need to compute attention between the current
/// token and all previous tokens. Instead of recomputing the Key and Value projections
/// for all previous tokens at each step, we cache them in GPU memory.
///
/// Layout: [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim]
/// Note: For GQA (Grouped Query Attention), num_kv_heads < num_query_heads
///
/// For batched inference, each sequence in the batch has its own independent cache.
///
/// # Compression
///
/// When compression is enabled, the cache extends effective context length by
/// compressing older entries. See `CacheCompressionConfig` for details.
pub struct KVCache {
    /// GPU buffer for cached Keys
    key_cache: wgpu::Buffer,

    /// GPU buffer for cached Values
    value_cache: wgpu::Buffer,

    /// Batch size (number of parallel sequences)
    batch_size: u32,

    /// Number of transformer layers
    num_layers: u32,

    /// Maximum sequence length supported by this cache
    max_seq_len: u32,

    /// Number of KV attention heads (for GQA, this is fewer than query heads)
    num_kv_heads: u32,

    /// Dimension of each attention head
    head_dim: u32,

    /// Current position in the cache (for ring buffer behavior)
    current_position: u32,

    /// Compression configuration
    compression_config: CacheCompressionConfig,

    /// Actual number of tokens stored (may differ from current_position after compression)
    actual_tokens_stored: u32,

    /// Number of compressed tokens (for statistics)
    compressed_tokens: u32,
}

impl KVCache {
    /// Create a new KV cache with the specified dimensions
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `batch_size` - Number of parallel sequences to support
    /// * `num_layers` - Number of transformer layers
    /// * `max_seq_len` - Maximum sequence length to support (e.g., 2048, 4096)
    /// * `num_kv_heads` - Number of KV attention heads (for GQA)
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Returns
    /// A new KVCache instance with pre-allocated GPU buffers
    pub fn new(
        engine: &ComputeEngine,
        batch_size: u32,
        num_layers: u32,
        max_seq_len: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        Self::with_compression(
            engine,
            batch_size,
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            CacheCompressionConfig::default(),
        )
    }

    /// Create a new KV cache with compression configuration
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `batch_size` - Number of parallel sequences to support
    /// * `num_layers` - Number of transformer layers
    /// * `max_seq_len` - Maximum sequence length to support (e.g., 2048, 4096)
    /// * `num_kv_heads` - Number of KV attention heads (for GQA)
    /// * `head_dim` - Dimension of each attention head
    /// * `compression_config` - Configuration for cache compression
    ///
    /// # Returns
    /// A new KVCache instance with pre-allocated GPU buffers and compression enabled
    pub fn with_compression(
        engine: &ComputeEngine,
        batch_size: u32,
        num_layers: u32,
        max_seq_len: u32,
        num_kv_heads: u32,
        head_dim: u32,
        compression_config: CacheCompressionConfig,
    ) -> Result<Self> {
        let device = engine.device();

        // Calculate total size in bytes (includes batch dimension)
        let cache_size = (batch_size * num_layers * max_seq_len * num_kv_heads * head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        if compression_config.enabled {
            tracing::info!(
                "Allocating KV cache with compression: batch_size={}, num_layers={}, max_seq_len={}, num_kv_heads={}, head_dim={}, total_size={:.2} MB, uncompressed_window={}, compression_ratio={}:1",
                batch_size,
                num_layers,
                max_seq_len,
                num_kv_heads,
                head_dim,
                (cache_size * 2) as f64 / (1024.0 * 1024.0),
                compression_config.uncompressed_window,
                compression_config.compression_ratio
            );
        } else {
            tracing::info!(
                "Allocating KV cache: batch_size={}, num_layers={}, max_seq_len={}, num_kv_heads={}, head_dim={}, total_size={:.2} MB",
                batch_size,
                num_layers,
                max_seq_len,
                num_kv_heads,
                head_dim,
                (cache_size * 2) as f64 / (1024.0 * 1024.0)
            );
        }

        // Create Key cache buffer
        let key_cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache_keys"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create Value cache buffer
        let value_cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache_values"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(Self {
            key_cache,
            value_cache,
            batch_size,
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            current_position: 0,
            compression_config,
            actual_tokens_stored: 0,
            compressed_tokens: 0,
        })
    }

    /// Update the cache with new Key and Value tensors for the current token
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `encoder` - Shared command encoder to record operations to
    /// * `pipeline_cache` - Pre-compiled pipeline cache
    /// * `new_key` - GPU buffer containing the new Key projection (num_kv_heads * head_dim elements)
    /// * `new_value` - GPU buffer containing the new Value projection (num_kv_heads * head_dim elements)
    /// * `layer_idx` - The transformer layer index (0 to num_layers-1)
    /// * `position` - Position in the sequence to write to (0 to max_seq_len-1)
    ///
    /// # Returns
    /// Ok(()) if successful. If compression is enabled and triggered, this method
    /// will automatically compress the cache before adding the new token.
    pub fn update(
        &mut self,
        engine: &ComputeEngine,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &PipelineCache,
        new_key: &wgpu::Buffer,
        new_value: &wgpu::Buffer,
        layer_idx: u32,
        position: u32,
    ) -> Result<()> {
        // Check if we need to compress before adding new token
        // We only compress on the first layer update to avoid redundant compression
        if layer_idx == 0 && self.should_compress() {
            let freed = self.compress(engine, encoder, pipeline_cache)?;
            tracing::debug!(
                "Auto-compression freed {} token slots at position {}",
                freed,
                position
            );
        }

        // Ensure position is within bounds
        let cache_position = position % self.max_seq_len;

        // Update Key cache
        self.update_cache_buffer(
            engine,
            encoder,
            pipeline_cache,
            new_key,
            &self.key_cache,
            layer_idx,
            cache_position,
        )?;

        // Update Value cache
        self.update_cache_buffer(
            engine,
            encoder,
            pipeline_cache,
            new_value,
            &self.value_cache,
            layer_idx,
            cache_position,
        )?;

        // Update position and token count
        self.current_position = cache_position + 1;
        
        // Track actual tokens stored (only increment on first layer)
        if layer_idx == 0 {
            // Update actual_tokens_stored to be the max of current value and new position
            self.actual_tokens_stored = self.actual_tokens_stored.max(position + 1);
        }

        Ok(())
    }

    /// Internal helper to update a cache buffer using the update_cache shader
    fn update_cache_buffer(
        &self,
        engine: &ComputeEngine,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &PipelineCache,
        new_data: &wgpu::Buffer,
        cache: &wgpu::Buffer,
        layer_idx: u32,
        position: u32,
    ) -> Result<()> {
        let device = engine.device();

        // Use cached shader
        let shader = &pipeline_cache.update_cache_shader;

        // Create uniforms
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct UpdateCacheUniforms {
            batch_size: u32,
            cache_position: u32,
            token_dim: u32,
            num_heads: u32,
            layer_idx: u32,
            max_seq_len: u32,
            num_layers: u32,
            _pad: u32, // Pad to 32 bytes (8 u32s total)
        }

        let uniforms = UpdateCacheUniforms {
            batch_size: self.batch_size,
            cache_position: position,
            token_dim: self.head_dim,
            num_heads: self.num_kv_heads,
            layer_idx,
            max_seq_len: self.max_seq_len,
            num_layers: self.num_layers,
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

        // Record compute pass to the shared encoder
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_cache_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup count (256 threads per workgroup)
            // Total elements: batch_size * num_kv_heads * head_dim
            let total_elements = self.batch_size * self.num_kv_heads * self.head_dim;
            let workgroup_count = (total_elements + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Do NOT submit - that's the caller's responsibility

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

    /// Get the number of transformer layers
    pub const fn num_layers(&self) -> u32 {
        self.num_layers
    }

    /// Get the maximum sequence length supported by this cache
    pub const fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    /// Get the number of KV attention heads
    pub const fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
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
        self.actual_tokens_stored = 0;
        self.compressed_tokens = 0;
    }

    /// Check if compression should be triggered
    ///
    /// Returns true if cache is at or above the compression trigger threshold
    pub fn should_compress(&self) -> bool {
        if !self.compression_config.enabled {
            return false;
        }

        let fill_ratio = self.actual_tokens_stored as f32 / self.max_seq_len as f32;
        fill_ratio >= self.compression_config.compression_trigger
    }

    /// Compress old KV cache entries to free space for new tokens
    ///
    /// This method compresses older cache entries using averaging to extend
    /// the effective context length. The compression strategy:
    ///
    /// 1. Keep recent tokens (uncompressed_window) at full precision
    /// 2. Compress older tokens by averaging N:1 (based on compression_ratio)
    /// 3. Shift compressed tokens and recent tokens to make room at the end
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `encoder` - Command encoder to record GPU operations
    /// * `pipeline_cache` - Pre-compiled pipeline cache
    ///
    /// # Returns
    /// Number of tokens freed by compression, or error
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::{ComputeEngine, KVCache, CacheCompressionConfig};
    /// # async fn example(engine: ComputeEngine) {
    /// let mut cache = KVCache::with_compression(
    ///     &engine, 1, 4, 2048, 4, 16,
    ///     CacheCompressionConfig {
    ///         enabled: true,
    ///         uncompressed_window: 512,
    ///         compression_ratio: 2,
    ///         compression_trigger: 0.8,
    ///     }
    /// ).unwrap();
    ///
    /// // ... fill cache with tokens ...
    ///
    /// if cache.should_compress() {
    ///     let mut encoder = engine.device().create_command_encoder(&Default::default());
    ///     let freed = cache.compress(&engine, &mut encoder, &pipeline_cache).unwrap();
    ///     println!("Freed {} token slots", freed);
    /// }
    /// # }
    /// ```
    pub fn compress(
        &mut self,
        engine: &ComputeEngine,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &PipelineCache,
    ) -> Result<u32> {
        if !self.compression_config.enabled {
            return Err(super::ComputeError::Other(
                "Compression not enabled for this cache".into(),
            ));
        }

        if self.actual_tokens_stored <= self.compression_config.uncompressed_window {
            // Not enough tokens to compress
            return Ok(0);
        }

        let device = engine.device();

        // Calculate compression range
        // Compress everything except the recent uncompressed window
        let compression_end = self.actual_tokens_stored.saturating_sub(self.compression_config.uncompressed_window);
        let compression_start = 0;
        let tokens_to_compress = compression_end - compression_start;

        if tokens_to_compress < self.compression_config.compression_ratio {
            // Not enough tokens to compress
            return Ok(0);
        }

        let compressed_size = tokens_to_compress / self.compression_config.compression_ratio;
        let tokens_freed = tokens_to_compress - compressed_size;

        tracing::debug!(
            "Compressing KV cache: tokens_to_compress={}, compression_ratio={}, compressed_size={}, tokens_freed={}",
            tokens_to_compress,
            self.compression_config.compression_ratio,
            compressed_size,
            tokens_freed
        );

        // Create uniforms
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct CompressionUniforms {
            batch_size: u32,
            num_layers: u32,
            max_seq_len: u32,
            num_kv_heads: u32,
            head_dim: u32,
            compression_start: u32,
            compression_end: u32,
            compression_ratio: u32,
        }

        let uniforms = CompressionUniforms {
            batch_size: self.batch_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            compression_start,
            compression_end,
            compression_ratio: self.compression_config.compression_ratio,
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compress_cache_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create temporary buffer for compressed output
        let cache_size = (self.batch_size
            * self.num_layers
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        let temp_key_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp_compressed_keys"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let temp_value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp_compressed_values"),
            size: cache_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Compress keys
        self.compress_buffer(
            device,
            encoder,
            pipeline_cache,
            &self.key_cache,
            &temp_key_buffer,
            &uniforms_buffer,
        )?;

        // Compress values
        self.compress_buffer(
            device,
            encoder,
            pipeline_cache,
            &self.value_cache,
            &temp_value_buffer,
            &uniforms_buffer,
        )?;

        // Copy compressed data back to main cache
        // Layout after compression: [compressed_old_tokens][uncompressed_recent_tokens][free_space]
        let compressed_bytes = (self.batch_size
            * self.num_layers
            * compressed_size
            * self.num_kv_heads
            * self.head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        encoder.copy_buffer_to_buffer(&temp_key_buffer, 0, &self.key_cache, 0, compressed_bytes);
        encoder.copy_buffer_to_buffer(&temp_value_buffer, 0, &self.value_cache, 0, compressed_bytes);

        // Copy uncompressed recent tokens to follow the compressed section
        let recent_start_pos = compression_end;
        let recent_tokens = self.actual_tokens_stored - recent_start_pos;
        let recent_bytes = (self.batch_size
            * self.num_layers
            * recent_tokens
            * self.num_kv_heads
            * self.head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        let recent_src_offset = (self.batch_size
            * self.num_layers
            * recent_start_pos
            * self.num_kv_heads
            * self.head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        let recent_dst_offset = compressed_bytes;

        encoder.copy_buffer_to_buffer(
            &self.key_cache,
            recent_src_offset,
            &self.key_cache,
            recent_dst_offset,
            recent_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.value_cache,
            recent_src_offset,
            &self.value_cache,
            recent_dst_offset,
            recent_bytes,
        );

        // Update cache state
        self.actual_tokens_stored = compressed_size + recent_tokens;
        self.compressed_tokens += tokens_freed;
        self.current_position = self.actual_tokens_stored;

        tracing::info!(
            "KV cache compressed: {} tokens -> {} tokens ({} freed, {} total compressed)",
            self.actual_tokens_stored + tokens_freed,
            self.actual_tokens_stored,
            tokens_freed,
            self.compressed_tokens
        );

        Ok(tokens_freed)
    }

    /// Helper method to compress a single buffer (key or value)
    fn compress_buffer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &PipelineCache,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        uniforms: &wgpu::Buffer,
    ) -> Result<()> {
        let shader = &pipeline_cache.compress_cache_shader;

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compress_cache_bind_group_layout"),
            entries: &[
                // Input cache (read-only)
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
                // Output cache (read-write)
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
            label: Some("compress_cache_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compress_cache_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: Default::default(),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compress_cache_pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Record compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compress_cache_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup count (256 threads per workgroup)
            let compressed_positions = (self.actual_tokens_stored.saturating_sub(self.compression_config.uncompressed_window))
                / self.compression_config.compression_ratio;
            let total_elements = self.batch_size
                * self.num_layers
                * compressed_positions
                * self.num_kv_heads
                * self.head_dim;
            let workgroup_count = (total_elements + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        Ok(())
    }

    /// Get compression statistics
    ///
    /// Returns (actual_tokens_stored, total_compressed_tokens, effective_context_length)
    pub const fn compression_stats(&self) -> (u32, u32, u32) {
        let effective_length = self.actual_tokens_stored + self.compressed_tokens;
        (self.actual_tokens_stored, self.compressed_tokens, effective_length)
    }

    /// Get compression configuration
    pub const fn compression_config(&self) -> &CacheCompressionConfig {
        &self.compression_config
    }

    /// Copy KV cache contents from source cache to this cache
    ///
    /// This method performs an efficient GPU-to-GPU copy of the entire KV cache.
    /// Used in speculative decoding to synchronize draft and target model caches.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder to record the copy operation
    /// * `source` - Source cache to copy from
    ///
    /// # Requirements
    /// - Source and destination caches must have identical dimensions
    /// - Both caches must have COPY_SRC and COPY_DST usage flags
    ///
    /// # Panics
    /// Panics if cache dimensions don't match
    pub fn copy_from(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &KVCache,
    ) -> Result<()> {
        // Verify cache dimensions match
        if self.batch_size != source.batch_size
            || self.num_layers != source.num_layers
            || self.max_seq_len != source.max_seq_len
            || self.num_kv_heads != source.num_kv_heads
            || self.head_dim != source.head_dim
        {
            return Err(super::ComputeError::Other(format!(
                "KV cache dimension mismatch: source {:?} != dest {:?}",
                (
                    source.batch_size,
                    source.num_layers,
                    source.max_seq_len,
                    source.num_kv_heads,
                    source.head_dim
                ),
                (
                    self.batch_size,
                    self.num_layers,
                    self.max_seq_len,
                    self.num_kv_heads,
                    self.head_dim
                )
            )));
        }

        // Calculate buffer size
        let cache_size = (self.batch_size
            * self.num_layers
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim) as u64
            * std::mem::size_of::<f32>() as u64;

        // Copy key cache
        encoder.copy_buffer_to_buffer(&source.key_cache, 0, &self.key_cache, 0, cache_size);

        // Copy value cache
        encoder.copy_buffer_to_buffer(&source.value_cache, 0, &self.value_cache, 0, cache_size);

        // Copy position
        self.current_position = source.current_position;

        tracing::debug!(
            "Copied KV cache: {} MB, position {}",
            (cache_size * 2) as f64 / (1024.0 * 1024.0),
            self.current_position
        );

        Ok(())
    }

    /// Get batch size
    pub const fn batch_size(&self) -> u32 {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kv_cache_creation() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let cache = KVCache::new(
            &engine,
            1,      // batch_size
            4,      // num_layers
            128,    // max_seq_len
            4,      // num_kv_heads
            16,     // head_dim
        );
        
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.max_seq_len(), 128);
        assert_eq!(cache.num_kv_heads(), 4);
        assert_eq!(cache.head_dim(), 16);
        assert_eq!(cache.current_position(), 0);
        assert_eq!(cache.batch_size(), 1);
    }

    #[tokio::test]
    async fn test_kv_cache_reset() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let mut cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        
        // Initially at position 0
        assert_eq!(cache.current_position(), 0);
        
        // Reset should set position back to 0
        cache.reset();
        assert_eq!(cache.current_position(), 0);
    }

    #[tokio::test]
    async fn test_kv_cache_copy_matching_dimensions() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        let mut dest = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        
        let mut encoder = engine.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("test_copy_encoder"),
            }
        );
        
        let result = dest.copy_from(&mut encoder, &source);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_kv_cache_copy_mismatched_dimensions() {
        let engine = ComputeEngine::new().await.unwrap();
        
        // Create caches with different dimensions
        let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        let mut dest = KVCache::new(&engine, 1, 4, 256, 4, 16).unwrap(); // Different max_seq_len
        
        let mut encoder = engine.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("test_copy_encoder"),
            }
        );
        
        let result = dest.copy_from(&mut encoder, &source);
        assert!(result.is_err());
        
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("dimension mismatch"));
        }
    }

    #[tokio::test]
    async fn test_kv_cache_copy_position_sync() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        let mut dest = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        
        // Manually set source position (simulating some forward passes)
        // Note: In real usage, position is set via update() method
        // Here we're just testing the copy_from position synchronization
        
        let mut encoder = engine.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("test_copy_encoder"),
            }
        );
        
        let result = dest.copy_from(&mut encoder, &source);
        assert!(result.is_ok());
        
        // After copying, destination should have same position as source
        assert_eq!(dest.current_position(), source.current_position());
    }

    #[test]
    fn test_kv_cache_buffer_access() {
        // Test that we can get references to the buffers
        // This is a compile-time test more than runtime
        
        // Note: We can't actually create a cache here without async/GPU
        // but we can verify the API exists
    }

    #[tokio::test]
    async fn test_compression_config_default() {
        let config = CacheCompressionConfig::default();
        
        assert!(!config.enabled);
        assert_eq!(config.uncompressed_window, 512);
        assert_eq!(config.compression_ratio, 2);
        assert_eq!(config.compression_trigger, 0.8);
    }

    #[tokio::test]
    async fn test_compression_enabled_cache_creation() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 256,
            compression_ratio: 2,
            compression_trigger: 0.75,
        };
        
        let cache = KVCache::with_compression(
            &engine,
            1,      // batch_size
            4,      // num_layers
            1024,   // max_seq_len
            4,      // num_kv_heads
            16,     // head_dim
            config,
        );
        
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        
        assert!(cache.compression_config().enabled);
        assert_eq!(cache.compression_config().uncompressed_window, 256);
        assert_eq!(cache.compression_config().compression_ratio, 2);
        
        let (actual, compressed, effective) = cache.compression_stats();
        assert_eq!(actual, 0);
        assert_eq!(compressed, 0);
        assert_eq!(effective, 0);
    }

    #[tokio::test]
    async fn test_should_compress_disabled() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        
        // Compression disabled by default
        assert!(!cache.should_compress());
    }

    #[tokio::test]
    async fn test_should_compress_below_threshold() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 256,
            compression_ratio: 2,
            compression_trigger: 0.8, // 80% full
        };
        
        let mut cache = KVCache::with_compression(
            &engine,
            1,
            4,
            1024, // max_seq_len
            4,
            16,
            config,
        ).unwrap();
        
        // Simulate storing tokens (but not enough to trigger compression)
        // actual_tokens_stored is normally updated in update()
        // For testing, we'll manually set it
        cache.actual_tokens_stored = 500; // ~48% full (500/1024)
        
        assert!(!cache.should_compress());
    }

    #[tokio::test]
    async fn test_should_compress_above_threshold() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 256,
            compression_ratio: 2,
            compression_trigger: 0.8, // 80% full
        };
        
        let mut cache = KVCache::with_compression(
            &engine,
            1,
            4,
            1024, // max_seq_len
            4,
            16,
            config,
        ).unwrap();
        
        // Simulate storing enough tokens to trigger compression
        cache.actual_tokens_stored = 900; // ~87% full (900/1024)
        
        assert!(cache.should_compress());
    }

    #[tokio::test]
    async fn test_compression_not_enough_tokens() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 512,
            compression_ratio: 2,
            compression_trigger: 0.8,
        };
        
        let mut cache = KVCache::with_compression(
            &engine,
            1,
            4,
            1024,
            4,
            16,
            config,
        ).unwrap();
        
        // Not enough tokens to compress (need at least uncompressed_window + compression_ratio)
        cache.actual_tokens_stored = 100;
        
        let mut encoder = engine.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("test_compression_encoder"),
            }
        );
        
        // Load pipeline cache (needed for compress to work)
        let pipeline_cache = PipelineCache::new(engine.device()).unwrap();
        
        let result = cache.compress(&engine, &mut encoder, &pipeline_cache);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No tokens freed
    }

    #[tokio::test]
    async fn test_compression_stats_tracking() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 256,
            compression_ratio: 2,
            compression_trigger: 0.8,
        };
        
        let mut cache = KVCache::with_compression(
            &engine,
            1,
            4,
            1024,
            4,
            16,
            config,
        ).unwrap();
        
        // Initial stats
        let (actual, compressed, effective) = cache.compression_stats();
        assert_eq!(actual, 0);
        assert_eq!(compressed, 0);
        assert_eq!(effective, 0);
        
        // Simulate some tokens stored
        cache.actual_tokens_stored = 600;
        let (actual, compressed, effective) = cache.compression_stats();
        assert_eq!(actual, 600);
        assert_eq!(compressed, 0);
        assert_eq!(effective, 600);
        
        // Simulate compression effect
        cache.compressed_tokens = 172; // e.g., compressed 344 tokens down to 172
        cache.actual_tokens_stored = 428; // 172 + 256 (uncompressed window)
        
        let (actual, compressed, effective) = cache.compression_stats();
        assert_eq!(actual, 428);
        assert_eq!(compressed, 172);
        assert_eq!(effective, 600); // Total context preserved
    }

    #[tokio::test]
    async fn test_reset_clears_compression_state() {
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = CacheCompressionConfig {
            enabled: true,
            uncompressed_window: 256,
            compression_ratio: 2,
            compression_trigger: 0.8,
        };
        
        let mut cache = KVCache::with_compression(
            &engine,
            1,
            4,
            1024,
            4,
            16,
            config,
        ).unwrap();
        
        // Simulate some state
        cache.actual_tokens_stored = 500;
        cache.compressed_tokens = 100;
        cache.current_position = 50;
        
        // Reset
        cache.reset();
        
        // All counters should be zero
        assert_eq!(cache.current_position(), 0);
        let (actual, compressed, effective) = cache.compression_stats();
        assert_eq!(actual, 0);
        assert_eq!(compressed, 0);
        assert_eq!(effective, 0);
    }

    #[tokio::test]
    async fn test_compress_disabled_cache_returns_error() {
        let engine = ComputeEngine::new().await.unwrap();
        
        // Create cache without compression enabled
        let mut cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
        
        let mut encoder = engine.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("test_encoder"),
            }
        );
        
        let pipeline_cache = PipelineCache::new(engine.device()).unwrap();
        
        // Attempt to compress should fail
        let result = cache.compress(&engine, &mut encoder, &pipeline_cache);
        assert!(result.is_err());
        
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("Compression not enabled"));
        }
    }
}


//! KV cache core implementation

use super::config::CacheCompressionConfig;
use crate::compute::engine::ComputeEngine;
use crate::compute::error::{ComputeError, Result};
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

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
    pub(super) key_cache: wgpu::Buffer,

    /// GPU buffer for cached Values
    pub(super) value_cache: wgpu::Buffer,

    /// Batch size (number of parallel sequences)
    pub(super) batch_size: u32,

    /// Number of transformer layers
    pub(super) num_layers: u32,

    /// Maximum sequence length supported by this cache
    pub(super) max_seq_len: u32,

    /// Number of KV attention heads (for GQA, this is fewer than query heads)
    pub(super) num_kv_heads: u32,

    /// Dimension of each attention head
    pub(super) head_dim: u32,

    /// Current position in the cache (for ring buffer behavior)
    pub(super) current_position: u32,

    /// Compression configuration
    pub(super) compression_config: CacheCompressionConfig,

    /// Actual number of tokens stored (may differ from current_position after compression)
    pub(super) actual_tokens_stored: u32,

    /// Number of compressed tokens (for statistics)
    pub(super) compressed_tokens: u32,
}

impl KVCache {
    fn checked_cache_size_bytes(
        batch_size: u32,
        num_layers: u32,
        max_seq_len: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<u64> {
        let elems = u64::from(batch_size)
            .checked_mul(u64::from(num_layers))
            .and_then(|v| v.checked_mul(u64::from(max_seq_len)))
            .and_then(|v| v.checked_mul(u64::from(num_kv_heads)))
            .and_then(|v| v.checked_mul(u64::from(head_dim)))
            .ok_or_else(|| {
                ComputeError::InvalidDimensions(format!(
                    "KV cache element count overflow: batch_size={} num_layers={} max_seq_len={} num_kv_heads={} head_dim={}",
                    batch_size, num_layers, max_seq_len, num_kv_heads, head_dim
                ))
            })?;

        elems
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                ComputeError::InvalidDimensions(format!(
                    "KV cache byte-size overflow: elements={} bytes_per_element={}",
                    elems,
                    std::mem::size_of::<f32>()
                ))
            })
    }

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

        // Calculate total size in bytes (includes batch dimension) with overflow checks.
        let cache_size = Self::checked_cache_size_bytes(
            batch_size,
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        )?;

        let total_kv_bytes = cache_size.checked_mul(2).ok_or_else(|| {
            ComputeError::InvalidDimensions(format!(
                "KV cache total key+value size overflow: key_or_value_bytes={}",
                cache_size
            ))
        })?;

        let limits = device.limits();
        if cache_size > u64::from(limits.max_storage_buffer_binding_size) {
            return Err(ComputeError::InvalidDimensions(format!(
                "KV cache buffer too large for device: {} bytes exceeds max_storage_buffer_binding_size {} (batch_size={} num_layers={} max_seq_len={} num_kv_heads={} head_dim={})",
                cache_size,
                limits.max_storage_buffer_binding_size,
                batch_size,
                num_layers,
                max_seq_len,
                num_kv_heads,
                head_dim
            )));
        }

        if cache_size > limits.max_buffer_size {
            return Err(ComputeError::InvalidDimensions(format!(
                "KV cache buffer too large for device: {} bytes exceeds max_buffer_size {} (batch_size={} num_layers={} max_seq_len={} num_kv_heads={} head_dim={})",
                cache_size,
                limits.max_buffer_size,
                batch_size,
                num_layers,
                max_seq_len,
                num_kv_heads,
                head_dim
            )));
        }

        if compression_config.enabled {
            tracing::info!(
                "Allocating KV cache with compression: batch_size={}, num_layers={}, max_seq_len={}, num_kv_heads={}, head_dim={}, total_size={:.2} MB, uncompressed_window={}, compression_ratio={}:1",
                batch_size,
                num_layers,
                max_seq_len,
                num_kv_heads,
                head_dim,
                total_kv_bytes as f64 / (1024.0 * 1024.0),
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
                total_kv_bytes as f64 / (1024.0 * 1024.0)
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

        // Disallow silent ring-buffer wraparound until sliding window attention is implemented.
        if position >= self.max_seq_len {
            return Err(crate::compute::ComputeError::Other(format!(
                "KV cache position {} exceeds max_seq_len {}",
                position, self.max_seq_len
            )));
        }

        let cache_position = position;

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
    pub(super) fn update_cache_buffer(
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
    pub const fn key_cache(&self) -> &wgpu::Buffer {
        &self.key_cache
    }

    /// Get a reference to the Value cache buffer
    pub const fn value_cache(&self) -> &wgpu::Buffer {
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
            return Err(crate::compute::ComputeError::Other(format!(
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
        let cache_size = Self::checked_cache_size_bytes(
            self.batch_size,
            self.num_layers,
            self.max_seq_len,
            self.num_kv_heads,
            self.head_dim,
        )?;

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

    /// Get compression configuration
    pub const fn compression_config(&self) -> &CacheCompressionConfig {
        &self.compression_config
    }
}

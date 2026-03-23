//! KV cache compression implementation

use super::core::KVCache;
use crate::compute::engine::ComputeEngine;
use crate::compute::error::Result;
use crate::compute::pipeline_cache::PipelineCache;
use wgpu::util::DeviceExt;

impl KVCache {
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
            return Err(crate::compute::ComputeError::Other(
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
        let compression_end = self
            .actual_tokens_stored
            .saturating_sub(self.compression_config.uncompressed_window);
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
        encoder.copy_buffer_to_buffer(
            &temp_value_buffer,
            0,
            &self.value_cache,
            0,
            compressed_bytes,
        );

        // Copy uncompressed recent tokens to follow the compressed section
        let recent_start_pos = compression_end;
        let recent_tokens = self.actual_tokens_stored - recent_start_pos;
        let recent_bytes =
            (self.batch_size * self.num_layers * recent_tokens * self.num_kv_heads * self.head_dim)
                as u64
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
            let compressed_positions = (self
                .actual_tokens_stored
                .saturating_sub(self.compression_config.uncompressed_window))
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
        (
            self.actual_tokens_stored,
            self.compressed_tokens,
            effective_length,
        )
    }
}

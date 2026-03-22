//! Token embedding operations

use super::Model;
use crate::compute::Result;

impl Model {
    /// Embed multiple tokens in parallel (batched embedding lookup)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `output_buffer` - Output buffer to write embeddings [batch_size, hidden_dim]
    /// * `token_ids` - Array of token IDs to embed [batch_size]
    ///
    /// # Returns
    /// Ok(()) on success
    pub(super) fn embed_tokens(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_buffer: &wgpu::Buffer,
        token_ids: &[u32],
    ) -> Result<()> {
        // Validate batch size
        if token_ids.len() != self.config.batch_size as usize {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Token IDs length ({}) does not match batch_size ({})",
                token_ids.len(),
                self.config.batch_size
            )));
        }

        // Validate all token IDs are in range
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.config.vocab_size {
                return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                    "Token ID {} at batch index {} is out of range [0, {})",
                    token_id, i, self.config.vocab_size
                )));
            }
        }

        // Create shader
        let shader = self
            .engine
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("embed_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../compute/shaders/embed.wgsl").into(),
                ),
            });

        // Create uniform buffer for params: [batch_size, hidden_dim]
        let params_buffer = self.engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed_params"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.engine.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::cast_slice(&[self.config.batch_size, self.config.hidden_dim]),
        );

        // Create token IDs buffer
        let token_ids_buffer = self.engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed_token_ids"),
            size: (self.config.batch_size * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.engine
            .queue()
            .write_buffer(&token_ids_buffer, 0, bytemuck::cast_slice(token_ids));

        // Create bind group layout
        let bind_group_layout =
            self.engine
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("embed_bind_group_layout"),
                    entries: &[
                        // Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Token IDs
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Embedding table
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self
            .engine
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: token_ids_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.token_embedding_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create pipeline
        let pipeline_layout =
            self.engine
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("embed_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: Default::default(),
                });

        let pipeline =
            self.engine
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("embed_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Record compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: process batch_size * hidden_dim elements
            // 256 threads per workgroup
            let total_elements = self.config.batch_size * self.config.hidden_dim;
            let workgroups = (total_elements + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        Ok(())
    }

    /// Embed a single token (convenience wrapper for batch_size=1)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `output_buffer` - Output buffer to write embedding [hidden_dim]
    /// * `token_id` - Token ID to embed
    ///
    /// # Returns
    /// Ok(()) on success
    pub(super) fn embed_token(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_buffer: &wgpu::Buffer,
        token_id: u32,
    ) -> Result<()> {
        // For batch_size=1, use the batched version with a single-element array
        if self.config.batch_size == 1 {
            self.embed_tokens(encoder, output_buffer, &[token_id])
        } else {
            // For batch_size > 1, this is a programming error
            Err(crate::compute::ComputeError::Other(format!(
                "embed_token() called with batch_size={}, use embed_tokens() instead",
                self.config.batch_size
            )))
        }
    }
}

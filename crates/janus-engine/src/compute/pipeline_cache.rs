//! GPU Pipeline Cache for Static Computation Graph
//!
//! Pre-creates and caches all shader modules, compute pipelines, and bind group
//! layouts used during inference. This eliminates the overhead of recreating
//! GPU resources on every forward pass.

/// Cache for all GPU pipelines used in the static computation graph
///
/// This struct holds pre-created shader modules, compute pipelines, and bind
/// group layouts for all operations used during model inference. By creating
/// these once and reusing them, we eliminate pipeline creation overhead which
/// can provide 10-20x performance improvement.
pub struct PipelineCache {
    // === Shader Modules ===
    pub embed_shader: wgpu::ShaderModule,
    pub rmsnorm_shader: wgpu::ShaderModule,
    pub gemm_shader: wgpu::ShaderModule,
    pub silu_shader: wgpu::ShaderModule,
    pub add_tensors_shader: wgpu::ShaderModule,
    pub elementwise_mul_shader: wgpu::ShaderModule,
    pub rope_shader: wgpu::ShaderModule,
    pub attention_shader: wgpu::ShaderModule,
    pub softmax_shader: wgpu::ShaderModule,
    pub update_cache_shader: wgpu::ShaderModule,
    pub compress_cache_shader: wgpu::ShaderModule,
    pub matmul_shader: wgpu::ShaderModule,
    pub gemm_q4_k_shader: wgpu::ShaderModule,
    pub gemm_q5_k_shader: wgpu::ShaderModule,
    pub gemm_q8_0_shader: wgpu::ShaderModule,

    // === Bind Group Layouts ===
    pub rmsnorm_layout: wgpu::BindGroupLayout,
    pub silu_layout: wgpu::BindGroupLayout,
    pub gemm_layout: wgpu::BindGroupLayout,
    pub matmul_layout: wgpu::BindGroupLayout,
    pub add_tensors_layout: wgpu::BindGroupLayout,
    pub elementwise_mul_layout: wgpu::BindGroupLayout,
    pub rope_layout: wgpu::BindGroupLayout,
    pub attention_qk_layout: wgpu::BindGroupLayout,
    pub attention_softmax_layout: wgpu::BindGroupLayout,
    pub attention_apply_layout: wgpu::BindGroupLayout,
    pub gemm_q4_k_layout: wgpu::BindGroupLayout,
    pub gemm_q5_k_layout: wgpu::BindGroupLayout,
    pub gemm_q8_0_layout: wgpu::BindGroupLayout,

    // === Compute Pipelines ===
    pub rmsnorm_pipeline: wgpu::ComputePipeline,
    pub silu_pipeline: wgpu::ComputePipeline,
    pub gemm_pipeline: wgpu::ComputePipeline,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub add_tensors_pipeline: wgpu::ComputePipeline,
    pub elementwise_mul_pipeline: wgpu::ComputePipeline,
    pub rope_pipeline: wgpu::ComputePipeline,
    pub attention_qk_pipeline: wgpu::ComputePipeline,
    pub attention_softmax_pipeline: wgpu::ComputePipeline,
    pub attention_apply_pipeline: wgpu::ComputePipeline,
    pub gemm_q4_k_pipeline: wgpu::ComputePipeline,
    pub gemm_q5_k_pipeline: wgpu::ComputePipeline,
    pub gemm_q8_0_pipeline: wgpu::ComputePipeline,
}

impl PipelineCache {
    /// Create a new pipeline cache with all shaders, layouts, and pipelines pre-compiled
    ///
    /// # Arguments
    /// * `device` - The GPU device to create pipelines on
    ///
    /// # Returns
    /// A new PipelineCache with all GPU resources pre-created
    pub fn new(device: &wgpu::Device) -> Self {
        println!("🔨 Creating comprehensive pipeline cache...");

        // ============================================
        // STEP 1: Load and compile all shader modules
        // ============================================

        let embed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/embed.wgsl").into()),
        });

        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("activations_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activations.wgsl").into()),
        });

        let gemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm.wgsl").into()),
        });

        let silu_shader = rmsnorm_shader.clone();

        let add_tensors_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add_tensors_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add_tensors.wgsl").into()),
        });

        let elementwise_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_mul_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/elementwise_mul.wgsl").into()),
        });

        let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rope.wgsl").into()),
        });

        let attention_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attention.wgsl").into()),
        });

        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/softmax.wgsl").into()),
        });

        let update_cache_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_cache_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/update_cache.wgsl").into()),
        });

        let compress_cache_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compress_cache_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compress_cache.wgsl").into()),
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let gemm_q4_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_q4_k_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm_q4_k.wgsl").into()),
        });

        let gemm_q5_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_q5_k_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm_q5_k.wgsl").into()),
        });

        let gemm_q8_0_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_q8_0_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm_q8_0.wgsl").into()),
        });

        println!("   ✅ {} shader modules compiled", 14);

        // ============================================
        // STEP 2: Create all bind group layouts
        // ============================================

        // RMSNorm layout: input, output, weights, uniforms
        let rmsnorm_layout = Self::create_rmsnorm_layout(device);

        // SiLU layout: input, output, uniforms
        let silu_layout = Self::create_silu_layout(device);

        // GEMM layout: matrix_a, matrix_b, output, uniforms
        let gemm_layout = Self::create_gemm_layout(device);

        // Matmul layout: matrix, vector, output, uniforms
        let matmul_layout = Self::create_matmul_layout(device);

        // Add tensors layout: a, b, output, uniforms
        let add_tensors_layout = Self::create_add_tensors_layout(device);

        // Elementwise multiply layout: a, b, output, uniforms
        let elementwise_mul_layout = Self::create_elementwise_mul_layout(device);

        // RoPE layout: input, output, uniforms
        let rope_layout = Self::create_rope_layout(device);

        // Attention QK layout
        let attention_qk_layout = Self::create_attention_qk_layout(device);

        // Attention softmax layout
        let attention_softmax_layout = Self::create_attention_softmax_layout(device);

        // Attention apply layout
        let attention_apply_layout = Self::create_attention_apply_layout(device);

        // Quantized GEMM layouts
        let gemm_q4_k_layout = Self::create_gemm_q4_k_layout(device);
        let gemm_q5_k_layout = Self::create_gemm_q5_k_layout(device);
        let gemm_q8_0_layout = Self::create_gemm_q8_0_layout(device);

        println!("   ✅ {} bind group layouts created", 13);

        // ============================================
        // STEP 3: Create all compute pipelines
        // ============================================

        let rmsnorm_pipeline = Self::create_pipeline(
            device,
            "rmsnorm_pipeline",
            &rmsnorm_layout,
            &rmsnorm_shader,
            "rmsnorm",
        );

        let silu_pipeline =
            Self::create_pipeline(device, "silu_pipeline", &silu_layout, &silu_shader, "silu");

        let gemm_pipeline =
            Self::create_pipeline(device, "gemm_pipeline", &gemm_layout, &gemm_shader, "gemm");

        let matmul_pipeline = Self::create_pipeline(
            device,
            "matmul_pipeline",
            &matmul_layout,
            &matmul_shader,
            "matmul",
        );

        let add_tensors_pipeline = Self::create_pipeline(
            device,
            "add_tensors_pipeline",
            &add_tensors_layout,
            &add_tensors_shader,
            "add_tensors",
        );

        let elementwise_mul_pipeline = Self::create_pipeline(
            device,
            "elementwise_mul_pipeline",
            &elementwise_mul_layout,
            &elementwise_mul_shader,
            "elementwise_mul",
        );

        let rope_pipeline =
            Self::create_pipeline(device, "rope_pipeline", &rope_layout, &rope_shader, "rope");

        let attention_qk_pipeline = Self::create_pipeline(
            device,
            "attention_qk_pipeline",
            &attention_qk_layout,
            &attention_shader,
            "compute_qk",
        );

        let attention_softmax_pipeline = Self::create_pipeline(
            device,
            "attention_softmax_pipeline",
            &attention_softmax_layout,
            &softmax_shader,
            "softmax",
        );

        let attention_apply_pipeline = Self::create_pipeline(
            device,
            "attention_apply_pipeline",
            &attention_apply_layout,
            &attention_shader,
            "apply_attention",
        );

        let gemm_q4_k_pipeline = Self::create_pipeline(
            device,
            "gemm_q4_k_pipeline",
            &gemm_q4_k_layout,
            &gemm_q4_k_shader,
            "main",
        );

        let gemm_q5_k_pipeline = Self::create_pipeline(
            device,
            "gemm_q5_k_pipeline",
            &gemm_q5_k_layout,
            &gemm_q5_k_shader,
            "main",
        );

        let gemm_q8_0_pipeline = Self::create_pipeline(
            device,
            "gemm_q8_0_pipeline",
            &gemm_q8_0_layout,
            &gemm_q8_0_shader,
            "main",
        );

        println!("   ✅ {} compute pipelines created", 13);
        println!("✅ Pipeline cache ready!");

        Self {
            embed_shader,
            rmsnorm_shader,
            gemm_shader,
            silu_shader,
            add_tensors_shader,
            elementwise_mul_shader,
            rope_shader,
            attention_shader,
            softmax_shader,
            update_cache_shader,
            compress_cache_shader,
            matmul_shader,
            gemm_q4_k_shader,
            gemm_q5_k_shader,
            gemm_q8_0_shader,
            rmsnorm_layout,
            silu_layout,
            gemm_layout,
            matmul_layout,
            add_tensors_layout,
            elementwise_mul_layout,
            rope_layout,
            attention_qk_layout,
            attention_softmax_layout,
            attention_apply_layout,
            gemm_q4_k_layout,
            gemm_q5_k_layout,
            gemm_q8_0_layout,
            rmsnorm_pipeline,
            silu_pipeline,
            gemm_pipeline,
            matmul_pipeline,
            add_tensors_pipeline,
            elementwise_mul_pipeline,
            rope_pipeline,
            attention_qk_pipeline,
            attention_softmax_pipeline,
            attention_apply_pipeline,
            gemm_q4_k_pipeline,
            gemm_q5_k_pipeline,
            gemm_q8_0_pipeline,
        }
    }

    // Helper to create a compute pipeline
    fn create_pipeline(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", label)),
            bind_group_layouts: &[Some(layout)],
            immediate_size: Default::default(),
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    // ============================================
    // Bind Group Layout Creators
    // ============================================

    fn create_rmsnorm_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rmsnorm_bind_group_layout"),
            entries: &[
                // Input (storage, read-only)
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
                // Output (storage, read-write)
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
                // Weights (storage, read-only)
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
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_silu_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("silu_bind_group_layout"),
            entries: &[
                // Input
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
                // Output
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
        })
    }

    fn create_gemm_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_bind_group_layout"),
            entries: &[
                // Matrix A
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
                // Matrix B
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_matmul_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bind_group_layout"),
            entries: &[
                // Matrix
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
                // Vector
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_add_tensors_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("add_tensors_bind_group_layout"),
            entries: &[
                // Input A
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
                // Input B
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_elementwise_mul_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise_mul_bind_group_layout"),
            entries: &[
                // Input A
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
                // Input B
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_rope_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_bind_group_layout"),
            entries: &[
                // Input
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
                // Output
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
        })
    }

    fn create_attention_qk_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention_qk_layout"),
            entries: &[
                // Q buffer
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
                // K cache
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
                // Scores output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_attention_softmax_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention_softmax_layout"),
            entries: &[
                // Scores input
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
                // Probs output
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
        })
    }

    fn create_attention_apply_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention_apply_layout"),
            entries: &[
                // Probs
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
                // V cache
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_gemm_q4_k_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_q4_k_layout"),
            entries: &[
                // Quantized matrix
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
                // Input vector
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
                // Output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_gemm_q5_k_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        // Same as Q4K
        Self::create_gemm_q4_k_layout(device)
    }

    fn create_gemm_q8_0_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        // Same as Q4K
        Self::create_gemm_q4_k_layout(device)
    }
}

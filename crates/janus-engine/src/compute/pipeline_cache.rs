//! GPU Pipeline Cache for Static Computation Graph
//!
//! Pre-creates and caches all shader modules and compute pipelines used during
//! inference. This eliminates the overhead of recreating pipelines on every
//! forward pass.

/// Cache for all GPU pipelines used in the static computation graph
///
/// This struct holds pre-created shader modules and compute pipelines
/// for all operations used during model inference. By creating these
/// once and reusing them, we eliminate shader compilation overhead.
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
}

impl PipelineCache {
    /// Create a new pipeline cache with all shaders pre-compiled
    ///
    /// # Arguments
    /// * `device` - The GPU device to create pipelines on
    ///
    /// # Returns
    /// A new PipelineCache with all shaders loaded and compiled
    pub fn new(device: &wgpu::Device) -> Self {
        tracing::info!("Creating pipeline cache...");

        // Load and compile all shader modules
        let embed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/embed.wgsl").into()),
        });

        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activations.wgsl").into()),
        });

        let gemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm.wgsl").into()),
        });

        let silu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silu_shader_cached"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/activations.wgsl").into()),
        });

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

        tracing::info!("Pipeline cache created with {} shader modules", 10);

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
        }
    }
}

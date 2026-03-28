use crate::formats::TensorDType;

#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub ggml_type: TensorDType,
}

impl std::ops::Deref for GpuTensor {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

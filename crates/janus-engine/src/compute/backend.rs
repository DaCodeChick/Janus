use super::buffer::{Buffer, BufferUsage};
use super::error::{ComputeError, Result};
use super::kernels::Kernels;
use wgpu;

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub limits: wgpu::Limits,
}

/// Main compute backend
pub struct ComputeBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Kernels,
    device_info: DeviceInfo,
}

impl ComputeBackend {
    /// Initialize the compute backend
    pub async fn new() -> Result<Self> {
        Self::new_with_power_preference(wgpu::PowerPreference::HighPerformance).await
    }

    /// Initialize with specific power preference
    pub async fn new_with_power_preference(power_pref: wgpu::PowerPreference) -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| ComputeError::DeviceRequestFailed(format!("Adapter request failed: {}", e)))?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("janus_compute_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits.clone(),
                    memory_hints: Default::default(),
                    experimental_features: Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .map_err(|e| ComputeError::DeviceRequestFailed(e.to_string()))?;

        let device_info = DeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            limits,
        };

        let kernels = Kernels::new(&device)?;

        Ok(Self {
            device,
            queue,
            kernels,
            device_info,
        })
    }

    /// Get device information
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Create a new buffer
    pub fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Buffer {
        Buffer::new(&self.device, size, usage, label)
    }

    /// Create a buffer with initial data
    pub fn create_buffer_with_data(&self, data: &[u8], usage: BufferUsage, label: &str) -> Buffer {
        Buffer::new_with_data(&self.device, data, usage, label)
    }

    /// Get reference to device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get reference to kernels
    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// Submit a command buffer
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

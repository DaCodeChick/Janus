use super::error::{ComputeError, Result};
use wgpu;

/// Buffer usage flags
#[derive(Debug, Clone, Copy)]
pub struct BufferUsage {
    pub storage: bool,
    pub uniform: bool,
    pub copy_src: bool,
    pub copy_dst: bool,
    pub map_read: bool,
    pub map_write: bool,
}

impl BufferUsage {
    pub fn storage_read_write() -> Self {
        Self {
            storage: true,
            uniform: false,
            copy_src: true,
            copy_dst: true,
            map_read: false,
            map_write: false,
        }
    }

    pub fn uniform() -> Self {
        Self {
            storage: false,
            uniform: true,
            copy_src: false,
            copy_dst: true,
            map_read: false,
            map_write: false,
        }
    }

    pub fn staging_read() -> Self {
        Self {
            storage: false,
            uniform: false,
            copy_src: false,
            copy_dst: true,
            map_read: true,
            map_write: false,
        }
    }

    pub fn staging_write() -> Self {
        Self {
            storage: false,
            uniform: false,
            copy_src: true,
            copy_dst: false,
            map_read: false,
            map_write: true,
        }
    }

    fn to_wgpu_usage(&self) -> wgpu::BufferUsages {
        let mut usage = wgpu::BufferUsages::empty();
        if self.storage {
            usage |= wgpu::BufferUsages::STORAGE;
        }
        if self.uniform {
            usage |= wgpu::BufferUsages::UNIFORM;
        }
        if self.copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        if self.copy_dst {
            usage |= wgpu::BufferUsages::COPY_DST;
        }
        if self.map_read {
            usage |= wgpu::BufferUsages::MAP_READ;
        }
        if self.map_write {
            usage |= wgpu::BufferUsages::MAP_WRITE;
        }
        usage
    }
}

/// GPU buffer wrapper
pub struct Buffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: u64,
}

impl Buffer {
    /// Create a new buffer
    pub(crate) fn new(device: &wgpu::Device, size: u64, usage: BufferUsage, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage.to_wgpu_usage(),
            mapped_at_creation: false,
        });

        Self { buffer, size }
    }

    /// Create a buffer with initial data
    pub(crate) fn new_with_data(
        device: &wgpu::Device,
        data: &[u8],
        usage: BufferUsage,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: usage.to_wgpu_usage(),
            mapped_at_creation: true,
        });

        buffer.slice(..).get_mapped_range_mut().copy_from_slice(data);
        buffer.unmap();

        Self {
            buffer,
            size: data.len() as u64,
        }
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Write data to buffer
    pub fn write(&self, queue: &wgpu::Queue, offset: u64, data: &[u8]) -> Result<()> {
        if offset + data.len() as u64 > self.size {
            return Err(ComputeError::BufferCreationFailed(
                "Write would exceed buffer size".to_string(),
            ));
        }
        queue.write_buffer(&self.buffer, offset, data);
        Ok(())
    }

    /// Read data from buffer (async)
    pub async fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
        // Create staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size);
        queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        receiver
            .await
            .map_err(|_| ComputeError::BufferMappingFailed)?
            .map_err(|_| ComputeError::BufferMappingFailed)?;

        let data = buffer_slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(data)
    }
}

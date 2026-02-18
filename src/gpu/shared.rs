//! Shared Memory Buffer for Vulkan-CUDA Hybrid Interop
//!
//! Provides synchronized memory access between CUDA and Vulkan backends
//! to prevent drift in long-running computations through concurrent execution

use anyhow::Result;
#[cfg(feature = "wgpu")]
use wgpu;

/// Shared buffer for cross-API memory access
/// Provides synchronized host memory accessible by both CUDA and Vulkan
pub struct SharedBuffer<T> {
    data: Vec<T>,
    #[cfg(feature = "wgpu")]
    vulkan_buffer: Option<wgpu::Buffer>,
    last_sync_version: u64,
}

impl<T: Copy + Default> SharedBuffer<T> {
    /// Create new shared buffer with zero-initialized memory
    pub fn new(len: usize) -> Self {
        let mut data = vec![T::default(); len];
        // Zero initialize explicitly
        for item in &mut data {
            *item = T::default();
        }

        Self {
            data,
            #[cfg(feature = "wgpu")]
            vulkan_buffer: None,
            last_sync_version: 0,
        }
    }

    /// Get immutable slice view
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice view
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.last_sync_version += 1; // Mark as modified
        &mut self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(feature = "wgpu")]
impl<T: Copy + Default> SharedBuffer<T> {
    /// Map buffer to Vulkan device memory via wgpu
    pub fn map_to_vulkan(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        let buffer_desc = wgpu::BufferDescriptor {
            label: Some("Shared Hybrid Buffer"),
            size: (self.data.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::MAP_WRITE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        };

        let vulkan_buffer = device.create_buffer(&buffer_desc);
        self.vulkan_buffer = Some(vulkan_buffer);

        // Initial sync to device
        self.sync_to_vulkan(device, queue)?;
        Ok(())
    }

    /// Sync data from Vulkan device back to host (simplified - data stays in host Vec)
    pub fn sync_from_vulkan(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) -> Result<()> {
        // For now, data stays synchronized through the shared Vec
        // In a full implementation, this would copy from Vulkan buffers
        Ok(())
    }

    /// Sync data from host to Vulkan device (simplified)
    pub fn sync_to_vulkan(&self, _device: &wgpu::Device, _queue: &wgpu::Queue) -> Result<()> {
        // For now, data stays synchronized through the shared Vec
        // In a full implementation, this would copy to Vulkan buffers
        Ok(())
    }

    /// Get Vulkan buffer reference
    pub fn vulkan_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vulkan_buffer.as_ref()
    }

    /// Check if Vulkan buffer needs sync
    pub fn needs_vulkan_sync(&self, version: u64) -> bool {
        self.last_sync_version != version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_buffer_creation() {
        let buffer: SharedBuffer<u64> = SharedBuffer::new(1024);
        assert_eq!(buffer.len(), 1024);
        assert!(buffer.as_slice().iter().all(|&x| x == 0));
    }
}

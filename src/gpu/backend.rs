//! Hybrid GPU dispatch abstraction
//!
//! Modular backend system for Vulkan/CUDA acceleration

#![allow(unsafe_code)] // Required for CUDA kernel launches and Vulkan buffer operations

#[cfg(feature = "rustacuda")]
use rustacuda::device::Device as CudaDevice;
#[cfg(feature = "rustacuda")]
use rustacuda::CudaFlags;
#[cfg(feature = "wgpu")]
use wgpu;

// Public re-exports
pub use super::backends::*;

/// Create a new hybrid backend with optimal GPU configuration
pub async fn create_backend() -> anyhow::Result<super::backends::HybridBackend> {
    super::backends::HybridBackend::new().await
}

/// Runtime GPU backend detection for optimal selection
pub fn detect_gpu_backend() -> String {
    // Try CUDA first (highest performance)
    #[cfg(feature = "rustacuda")]
    {
        if let Ok(()) = unsafe { rustacuda::init(CudaFlags::empty()) } {
            if let Ok(_) = CudaDevice::get_device(0) {
                return "cuda".to_string();
            }
        }
    }

    // Try Vulkan second
    #[cfg(feature = "wgpu")]
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        if !instance.enumerate_adapters(wgpu::Backends::PRIMARY).is_empty() {
            return "vulkan".to_string();
        }
    }

    // Fallback to CPU
    "cpu".to_string()
}
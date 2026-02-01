//! Hybrid GPU dispatch abstraction
//!
//! Modular backend system for Vulkan/CUDA acceleration

#![allow(unsafe_code)] // Required for CUDA kernel launches and Vulkan buffer operations

// Public re-exports
pub use backends::*;

// Private module declarations
mod backends;

/// Runtime GPU backend detection for optimal selection
pub fn detect_gpu_backend() -> String {
    // Try CUDA first (highest performance)
    #[cfg(feature = "rustacuda")]
    {
        if let Ok(()) = unsafe { rustacuda::init(rustacuda::CudaFlags::empty()) } {
            if let Ok(_) = rustacuda::device::get_device(0) {
                return "cuda".to_string();
            }
        }
    }

    // Try Vulkan second
    #[cfg(feature = "wgpu")]
    {
        if let Ok(instance) = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        }) {
            if !instance.enumerate_adapters(wgpu::Backends::PRIMARY).is_empty() {
                return "vulkan".to_string();
            }
        }
    }

    // Fallback to CPU
    "cpu".to_string()
}
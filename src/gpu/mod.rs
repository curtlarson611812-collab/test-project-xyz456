//! GPU acceleration module for SpeedBitCrack V3
//!
//! Hybrid Vulkan/CUDA backend for bulk compute operations

pub mod backend;
pub mod backends;
pub mod cuda;
pub mod hybrid_manager;
pub mod memory;
pub mod shared;
#[cfg(test)]
pub mod tests;
#[cfg(feature = "vulkano")]
pub mod vulkan;

// Re-export main backend interfaces
pub use backend::detect_gpu_backend;
pub use backends::*;
pub use hybrid_manager::HybridGpuManager;

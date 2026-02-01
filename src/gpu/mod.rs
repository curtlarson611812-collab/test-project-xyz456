//! GPU acceleration module for SpeedBitCrack V3
//!
//! Hybrid Vulkan/CUDA backend for bulk compute operations

pub mod backend;
pub mod backends;
#[cfg(feature = "vulkano")]
pub mod vulkan;
pub mod cuda;

// Re-export main backend interfaces
pub use backend::{detect_gpu_backend};
pub use backends::*;
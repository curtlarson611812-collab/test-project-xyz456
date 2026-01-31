//! GPU acceleration module for SpeedBitCrack V3
//!
//! Hybrid Vulkan/CUDA backend for bulk compute operations

pub mod backend;
pub mod vulkan;
pub mod cuda;

// Re-export main backend trait
pub use backend::{GpuBackend, create_backend};
//! Backend modules for GPU acceleration

pub mod backend_trait;
pub mod cpu_backend;
pub mod cuda_backend;
pub mod hybrid_backend;
pub mod vulkan_backend;

// Re-export everything
pub use backend_trait::*;
pub use cpu_backend::*;
pub use cuda_backend::*;
pub use hybrid_backend::*;
pub use vulkan_backend::*;

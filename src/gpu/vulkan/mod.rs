//! Vulkan/wgpu backend implementation
//!
//! wgpu device/queue/pipeline creation, bind group layouts, push constants

pub mod pipeline;

// Shaders will be embedded
pub const KANGAROO_SHADER: &str = include_str!("shaders/kangaroo.wgsl");
pub const JUMP_TABLE_SHADER: &str = include_str!("shaders/jump_table.wgsl");
pub const DP_CHECK_SHADER: &str = include_str!("shaders/dp_check.wgsl");
pub const UTILS_SHADER: &str = include_str!("shaders/utils.wgsl");
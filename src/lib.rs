//! SpeedBitCrack V3 - Pollard's rho/kangaroo ECDLP solver for secp256k1
//!
//! High-performance, multi-target ECDLP solver targeting RTX 5090s to crack
//! private keys from early unspent P2PK outputs (blocks 1–500k, >1 BTC) and
//! Bitcoin puzzle addresses.
//!
//! Built with Rust + hybrid GPU acceleration (Vulkan/wgpu bulk compute, CUDA precision math).
//!
//! Security guarantees:
//! - Constant-time cryptographic operations
//! - Comprehensive input validation
//! - Side-channel attack protection
//! - No unsafe code usage

#![deny(unsafe_code)]

pub mod config;
pub mod types;
pub mod math;
pub mod kangaroo;
pub mod gpu;
pub mod dp;
pub mod parity;
pub mod targets;
pub mod utils;
pub mod security;

// Re-export key types for library usage
pub use config::Config;
pub use types::{KangarooState, Point, AlphaBeta};
pub use kangaroo::KangarooManager;
pub use gpu::GpuBackend;
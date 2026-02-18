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

#[allow(non_snake_case)]
pub mod SmallOddPrime_Precise_code;
pub mod checkpoint;
pub mod cli;
pub mod config;
pub mod cuda_parity_test;
pub mod dp;
pub mod gpu;
pub mod kangaroo;
pub mod math;
pub mod parity;
pub mod performance_monitor;
pub mod puzzles;
pub mod security;
pub mod simple_test;
pub mod targets;
pub mod test_basic;
pub mod types;
pub mod utils;
// pub mod test_orchestrator; // Temporarily disabled

// Re-export key types for library usage
pub use config::Config;
pub use gpu::GpuBackend;
pub use kangaroo::KangarooManager;
pub use types::{AlphaBeta, KangarooState, Point};

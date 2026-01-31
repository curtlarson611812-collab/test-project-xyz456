//! Parity verification module
//!
//! 10M-step CPU vs GPU bit-for-bit verification harness

pub mod checker;

// Re-export main type
pub use checker::ParityChecker;
//! Parity verification modules
//!
//! Comprehensive CPU vs GPU bit-for-bit verification harness

pub mod checker;
pub mod framework;

// Re-export main type
pub use checker::ParityChecker;
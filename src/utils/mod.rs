//! Utility functions and helpers
//!
//! Logging, hashing, and other utilities

pub mod logging;
pub mod hash;

// Re-export commonly used utilities
pub use logging::setup_logging;
pub use hash::fast_hash;
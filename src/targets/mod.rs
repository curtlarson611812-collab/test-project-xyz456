//! Target loading and management
//!
//! Load & parse valuable_p2pk_publickey.txt + puzzles.txt, validate pubkeys

pub mod loader;

// Re-export main type
pub use loader::TargetLoader;
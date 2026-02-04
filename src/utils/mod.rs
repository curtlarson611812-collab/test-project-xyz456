//! Utility functions and helpers
//!
//! Logging, hashing, pubkey loading, and output utilities

pub mod logging;
pub mod hash;
pub mod pubkey_loader;
pub mod output;

// Re-export commonly used utilities
pub use logging::setup_logging;
pub use hash::fast_hash;
pub use pubkey_loader::{
    load_pubkeys_from_file,
    load_all_puzzles_pubkeys,
    load_test_puzzle_keys,
    load_unsolved_puzzle_keys,
    load_valuable_p2pk_keys
};
//! Utility functions and helpers
//!
//! Logging, hashing, pubkey loading, and output utilities

pub mod bias;
pub mod hash;
pub mod logging;
pub mod output;
pub mod pubkey_loader;

// Re-export commonly used utilities
pub use bias::{
    analyze_comprehensive_bias, analyze_comprehensive_bias_with_global, compute_global_stats,
    is_high_bias_target, BiasAnalysis, BiasWeights, GlobalBiasStats, PUZZLE_135_BIAS,
    PUZZLE_145_BIAS, PUZZLE_145_GOLD_BIAS, PUZZLE_145_MOD27_BIAS, PUZZLE_145_MOD3_BIAS,
    PUZZLE_145_MOD81_BIAS, PUZZLE_145_MOD9_BIAS, PUZZLE_145_POP_BIAS,
};
pub use hash::fast_hash;
pub use logging::setup_logging;
pub use pubkey_loader::{
    load_all_puzzles_pubkeys, load_pubkeys_from_file, load_test_puzzle_keys,
    load_unsolved_puzzle_keys, load_valuable_p2pk_keys,
};

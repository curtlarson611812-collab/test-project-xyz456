//! Search Configuration for Multi-Target Kangaroo Operations
//!
//! Defines customizable parameters for different types of pubkey lists:
//! - Valuable P2PK (unbounded, large batches)
//! - Test puzzles (bounded, small batches, quick validation)
//! - Unsolved puzzles (bounded, prime-biased jumps, long runs)

use crate::math::bigint::BigInt256;
use anyhow::Result;

/// Configuration for kangaroo search parameters
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of kangaroos per target pubkey
    pub batch_per_target: usize,
    /// Maximum steps before halting/resetting (prevents infinite loops)
    pub max_steps: u64,
    /// Small odd primes for jump distance selection (enables magic-9 bias)
    pub jump_primes: Vec<u64>,
    /// Distinguished point trailing zero bits (higher = fewer DPs but larger search)
    pub dp_bits: u32,
    /// Whether search is bounded (puzzles) or unbounded (P2PK)
    pub is_bounded: bool,
    /// Start of search range (for bounded searches)
    pub range_start: BigInt256,
    /// End of search range (for bounded searches)
    pub range_end: BigInt256,
    /// Target name for logging and identification
    pub name: String,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            batch_per_target: 512,  // Moderate batch size for valuable P2PK
            max_steps: u64::MAX,     // Run indefinitely until solution
            jump_primes: vec![3, 5, 7, 11, 13, 17, 19, 23], // Small odd primes
            dp_bits: 20,             // ~1M average steps per DP
            is_bounded: false,       // Unbounded for general P2PK
            range_start: BigInt256::zero(),
            range_end: BigInt256::max_value(),
            name: "default".to_string(),
        }
    }
}

impl SearchConfig {
    /// Configuration optimized for valuable P2PK targets (large scale, unbounded)
    pub fn for_valuable_p2pk() -> Self {
        Self {
            batch_per_target: 1024, // Large batches for throughput
            max_steps: u64::MAX,
            jump_primes: vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            dp_bits: 24,            // Fewer DPs for efficiency
            is_bounded: false,
            range_start: BigInt256::zero(),
            range_end: BigInt256::max_value(),
            name: "valuable_p2pk".to_string(),
        }
    }

    /// Configuration optimized for test/solved puzzles (quick validation)
    pub fn for_test_puzzles() -> Self {
        Self {
            batch_per_target: 64,   // Small batches for quick testing
            max_steps: 100_000,     // Limited steps for fast validation
            jump_primes: vec![3, 5, 7, 11, 13], // Minimal primes
            dp_bits: 16,            // More frequent DPs for testing
            is_bounded: true,       // Test keys have known small ranges
            range_start: BigInt256::one(),
            range_end: BigInt256::from_u64(1u64 << 32), // Up to 32-bit for tests
            name: "test_puzzles".to_string(),
        }
    }

    /// Configuration optimized for unsolved puzzles (precision, long runs)
    pub fn for_unsolved_puzzles() -> Self {
        Self {
            batch_per_target: 512,  // Moderate batches for balance
            max_steps: u64::MAX,
            jump_primes: vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73],
            dp_bits: 22,            // Balanced DP frequency
            is_bounded: true,       // Puzzles have defined ranges
            range_start: BigInt256::one().shl(65), // 2^65 for typical high puzzles
            range_end: BigInt256::one().shl(66).sub(&BigInt256::one()), // 2^66 - 1
            name: "unsolved_puzzles".to_string(),
        }
    }

    /// Create config for specific puzzle range
    pub fn for_puzzle_range(start_bit: u32, end_bit: u32) -> Self {
        let mut config = Self::for_unsolved_puzzles();
        config.range_start = BigInt256::one().shl(start_bit);
        config.range_end = BigInt256::one().shl(end_bit).sub(&BigInt256::one());
        config.name = format!("puzzle_{}_{}", start_bit, end_bit);
        config
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.batch_per_target == 0 {
            return Err("batch_per_target must be > 0".to_string());
        }
        if self.jump_primes.is_empty() {
            return Err("jump_primes cannot be empty".to_string());
        }
        if self.dp_bits == 0 || self.dp_bits > 32 {
            return Err("dp_bits must be between 1 and 32".to_string());
        }
        if self.is_bounded && self.range_start >= self.range_end {
            return Err("range_start must be < range_end for bounded searches".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SearchConfig::default();
        assert_eq!(config.batch_per_target, 512);
        assert!(!config.is_bounded);
        assert_eq!(config.name, "default");
    }

    #[test]
    fn test_valuable_config() {
        let config = SearchConfig::for_valuable_p2pk();
        assert_eq!(config.batch_per_target, 1024);
        assert!(!config.is_bounded);
        assert_eq!(config.name, "valuable_p2pk");
    }

    #[test]
    fn test_test_config() {
        let config = SearchConfig::for_test_puzzles();
        assert_eq!(config.batch_per_target, 64);
        assert_eq!(config.max_steps, 100_000);
        assert!(config.is_bounded);
        assert_eq!(config.name, "test_puzzles");
    }

    #[test]
    fn test_unsolved_config() {
        let config = SearchConfig::for_unsolved_puzzles();
        assert_eq!(config.batch_per_target, 512);
        assert!(config.is_bounded);
        assert_eq!(config.name, "unsolved_puzzles");
    }

    #[test]
    fn test_puzzle_range_config() {
        let config = SearchConfig::for_puzzle_range(64, 65);
        assert_eq!(config.range_start, BigInt256::one().shl(64));
        assert_eq!(config.range_end, BigInt256::one().shl(65).sub(&BigInt256::one()));
        assert_eq!(config.name, "puzzle_64_65");
    }

    #[test]
    fn test_config_validation() {
        let mut config = SearchConfig::default();
        assert!(config.validate().is_ok());

        config.batch_per_target = 0;
        assert!(config.validate().is_err());

        config = SearchConfig::default();
        config.jump_primes.clear();
        assert!(config.validate().is_err());
    }
}
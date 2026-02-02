//! Search Configuration for Multi-Target Kangaroo Operations
//!
//! Defines customizable parameters for different types of pubkey lists:
//! - Valuable P2PK (unbounded, large batches)
//! - Test puzzles (bounded, small batches, quick validation)
//! - Unsolved puzzles (bounded, prime-biased jumps, long runs)

use crate::math::bigint::BigInt256;
use anyhow::Result;
use std::collections::HashMap;

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
    /// Per-puzzle ranges: puzzle_id -> (start, end) for bounded puzzle searches
    pub per_puzzle_ranges: Option<HashMap<u32, (BigInt256, BigInt256)>>,
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
            range_end: BigInt256::from_u64(1u64 << 40), // Large but reasonable default
            per_puzzle_ranges: None,
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
            range_end: BigInt256::from_u64(1u64 << 40), // Large but reasonable default
            per_puzzle_ranges: None,
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
            per_puzzle_ranges: None,
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
            range_start: BigInt256::one() << 65usize, // 2^65 for typical high puzzles
            range_end: (BigInt256::one() << 66usize) - BigInt256::one(), // 2^66 - 1
            per_puzzle_ranges: None,
            name: "unsolved_puzzles".to_string(),
        }
    }

    /// Create config for specific puzzle range
    pub fn for_puzzle_range(start_bit: u32, end_bit: u32) -> Self {
        let mut config = Self::for_unsolved_puzzles();
        config.range_start = BigInt256::one() << start_bit as usize;
        config.range_end = (BigInt256::one() << end_bit as usize) - BigInt256::one();
        config.name = format!("puzzle_{}_{}", start_bit, end_bit);
        config
    }

    /// Enable per-puzzle ranges configuration
    pub fn with_per_puzzle_ranges(mut self) -> Self {
        self.per_puzzle_ranges = Some(HashMap::new());
        self
    }

    /// Add a specific puzzle range
    pub fn add_puzzle_range(&mut self, puzzle_id: u32, bit_depth: u32) {
        if let Some(ranges) = &mut self.per_puzzle_ranges {
            let start = BigInt256::one() << (bit_depth - 1) as usize;
            let end = (BigInt256::one() << bit_depth as usize) - BigInt256::one();
            ranges.insert(puzzle_id, (start, end));
        }
    }

    /// Load default unsolved puzzle ranges (hardcoded from public data)
    pub fn load_default_unsolved_ranges(&mut self) {
        self.per_puzzle_ranges = Some(HashMap::new());
        if let Some(ranges) = &mut self.per_puzzle_ranges {
            // Add ranges for unsolved puzzles #66 to #160
            for puzzle_id in 66..=160 {
                let start = BigInt256::one() << (puzzle_id - 1) as usize;
                let end = (BigInt256::one() << puzzle_id as usize) - BigInt256::one();
                ranges.insert(puzzle_id, (start, end));
            }
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.batch_per_target == 0 {
            return Err(anyhow::anyhow!("batch_per_target must be > 0"));
        }
        if self.jump_primes.is_empty() {
            return Err(anyhow::anyhow!("jump_primes cannot be empty"));
        }
        if self.dp_bits == 0 || self.dp_bits > 32 {
            return Err(anyhow::anyhow!("dp_bits must be between 1 and 32"));
        }
        if self.is_bounded && self.range_start >= self.range_end {
            return Err(anyhow::anyhow!("range_start must be < range_end for bounded searches"));
        }
        // Validate per-puzzle ranges if present
        if let Some(ranges) = &self.per_puzzle_ranges {
            for (puzzle_id, (start, end)) in ranges {
                if start >= end {
                    return Err(anyhow::anyhow!("puzzle {}: range_start must be < range_end", puzzle_id));
                }
                if *puzzle_id < 64 || *puzzle_id > 160 {
                    return Err(anyhow::anyhow!("puzzle {}: id must be between 64 and 160", puzzle_id));
                }
            }
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
        assert_eq!(config.range_start, BigInt256::one() << 64usize);
        assert_eq!(config.range_end, (BigInt256::one() << 65usize) - BigInt256::one());
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
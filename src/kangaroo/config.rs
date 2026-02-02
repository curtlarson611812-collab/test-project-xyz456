//! Kangaroo Search Configuration for Multi-Target Optimization
//!
//! Configurable parameters for different target lists (valuable P2PK, test puzzles, unsolved puzzles)
//! Allows tailored search strategies per list type for optimal performance

use crate::math::bigint::BigInt256;

#[derive(Clone)]
pub struct SearchConfig {
    /// Number of kangaroos per target (batch size)
    pub batch_per_target: usize,
    /// Maximum steps before halting/resetting
    pub max_steps: u64,
    /// Small odd primes for jump distance selection (magic-9 bias)
    pub jump_primes: Vec<u64>,
    /// Distinguished point trailing zero bits
    pub dp_bits: u32,
    /// Whether search is bounded (puzzles) or unbounded (P2PK)
    pub is_bounded: bool,
    /// Lower bound for bounded searches (2^{n-1} for puzzles)
    pub range_start: BigInt256,
    /// Upper bound for bounded searches (2^n - 1 for puzzles)
    pub range_end: BigInt256,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            batch_per_target: 512,
            max_steps: u64::MAX,
            jump_primes: vec![3, 5, 7, 11, 13, 17, 19, 23], // Small odds for magic-9 bias
            dp_bits: 20, // ~1M average steps per DP
            is_bounded: false,
            range_start: BigInt256::zero(),
            range_end: BigInt256::max_value(),
        }
    }
}

impl SearchConfig {
    /// Create config optimized for valuable P2PK targets (unbounded, large scale)
    pub fn for_valuable_p2pk() -> Self {
        Self {
            batch_per_target: 1024, // Large batches for GPU efficiency
            max_steps: u64::MAX,    // Run indefinitely
            jump_primes: vec![3,5,7,11,13,17,19,23,29,31,37,41,43,47], // Extended primes
            dp_bits: 24,            // More bits for large search space
            is_bounded: false,
            range_start: BigInt256::zero(),
            range_end: BigInt256::max_value(),
        }
    }

    /// Create config optimized for test puzzle validation (small, fast)
    pub fn for_test_puzzles() -> Self {
        Self {
            batch_per_target: 64,      // Small batches for quick validation
            max_steps: 100_000,        // Limited steps for testing
            jump_primes: vec![3,5,7],  // Minimal primes
            dp_bits: 16,               // Fewer bits for faster DP detection
            is_bounded: true,
            range_start: BigInt256::one(),
            range_end: BigInt256::from_u64(1u64 << 32), // Up to 32-bit for test puzzles
        }
    }

    /// Create config optimized for unsolved puzzles (bounded, prime-biased)
    pub fn for_unsolved_puzzles() -> Self {
        Self {
            batch_per_target: 512,
            max_steps: u64::MAX,
            jump_primes: vec![3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73], // Extended for magic-9
            dp_bits: 20,
            is_bounded: true,
            range_start: BigInt256::one() << 65usize, // Default 2^65 for higher puzzles
            range_end: (BigInt256::one() << 66usize) - BigInt256::one(), // 2^66 - 1
        }
    }

    /// Create config for specific puzzle bit range
    pub fn for_puzzle_bits(bits: u32) -> Self {
        let mut config = Self::for_unsolved_puzzles();
        config.range_start = BigInt256::one() << (bits - 1) as usize;
        config.range_end = (BigInt256::one() << bits as usize) - BigInt256::one();
        config
    }
}
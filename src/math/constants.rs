//! Mathematical constants for SpeedBitCrack V3
//!
//! Contains cryptographic constants and prime arrays for kangaroo optimization.

// Concise Block: Verbatim Preset Small Odd Primes (>128, odd, low Hamming)
// From ./SmallOddPrime_Precise_code.rs — locked, no adjustments.
pub const PRIME_MULTIPLIERS: [u64; 32] = [
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
];

// Test: assert_eq!(PRIME_MULTIPLIERS.len(), 32); // Cycle %32 for unique starts
// Deep note: Low Hamming wt (e.g., 179=0b10110011, wt=5) for fast scalar mul in GPU.
//! Mathematical constants for SpeedBitCrack V3
//!
//! Contains cryptographic constants and prime arrays for kangaroo optimization.

use crate::math::bigint::BigInt256;
use crate::types::Point;
use std::sync::LazyLock;

// Concise Block: Verbatim Preset Small Odd Primes (>128, odd, low Hamming)
// From ./SmallOddPrime_Precise_code.rs — locked, no adjustments.
pub const PRIME_MULTIPLIERS: [u64; 32] = [
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
];

// Secp256k1 curve constants - lazy initialized to avoid const function limitations
pub static CURVE_ORDER: LazyLock<BigInt256> = LazyLock::new(|| {
    BigInt256::from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
});

pub static GENERATOR: LazyLock<Point> = LazyLock::new(|| {
    Point {
        x: BigInt256::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798").limbs,
        y: BigInt256::from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8").limbs,
        z: BigInt256::from_u64(1).limbs,
    }
});

// DP and jump table constants
pub const DP_BITS: u32 = 24;

// Mock jump table - expand to proper EC operations in production
pub static JUMP_TABLE: LazyLock<[BigInt256; 8]> = LazyLock::new(|| [
    BigInt256::from_u64(1),   // G
    BigInt256::from_u64(2),   // 2G
    BigInt256::from_u64(3),   // 3G
    BigInt256::from_u64(4),   // 4G
    BigInt256::from_u64(5),   // 5G
    BigInt256::from_u64(6),   // 6G
    BigInt256::from_u64(7),   // 7G
    BigInt256::from_u64(8),   // 8G
]);

// Test: assert_eq!(PRIME_MULTIPLIERS.len(), 32); // Cycle %32 for unique starts
// Deep note: Low Hamming wt (e.g., 179=0b10110011, wt=5) for fast scalar mul in GPU.
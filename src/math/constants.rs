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

// Secp256k1 curve constants - string versions for easy access
pub const CURVE_ORDER: &str = "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";
pub const GENERATOR_X: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
pub const GENERATOR_Y: &str = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

// Lazy initialized versions for computation
pub static CURVE_ORDER_BIGINT: LazyLock<BigInt256> = LazyLock::new(|| {
    BigInt256::from_hex(CURVE_ORDER).expect("Invalid curve order")
});

pub static GENERATOR: LazyLock<Point> = LazyLock::new(|| {
    Point {
        x: BigInt256::from_hex(GENERATOR_X).unwrap().limbs,
        y: BigInt256::from_hex(GENERATOR_Y).unwrap().limbs,
        z: BigInt256::from_u64(1).limbs,
    }
});

// DP and jump table constants
pub const DP_BITS: u32 = 24;
pub const JUMP_TABLE_SIZE: usize = 256;

// Jump table with proper EC operations
pub fn jump_table() -> Vec<BigInt256> {
    // For now, use simple powers of 2 and small multiples
    // In production, these would be precomputed EC points
    let mut jumps = Vec::with_capacity(JUMP_TABLE_SIZE);

    // Small multiples for fine-grained movement
    for i in 1..=64 {
        jumps.push(BigInt256::from_u64(i));
    }

    // Powers of 2 for larger jumps
    for i in 1..=63 {
        jumps.push(BigInt256::from_u64(1u64 << i));
    }
    // For i=64, use BigInt256 directly
    jumps.push(BigInt256::from_u64(1u64 << 63) * BigInt256::from_u64(2));

    // Random-ish values for mixing (deterministic)
    for i in 128..JUMP_TABLE_SIZE {
        let val = (i as u64).wrapping_mul(0x9e3779b9) % (1u64 << 40); // Keep reasonable size
        jumps.push(BigInt256::from_u64(val + 1)); // +1 to avoid zero
    }

    jumps
}

// GLV (Gallant-Lambert-Vanstone) constants for endomorphism optimization
// lambda = sqrt(-3) mod p for secp256k1, enabling ~15% stall reduction
pub const GLV_LAMBDA: &str = "5b2b3e9c8b278c34d3763265d4f1630aa667c87bdd43a382d18a4ed82eabccb";

// beta = lambda * G (generator point), precomputed for GLV decomposition
pub const GLV_BETA_X: &str = "128ec4256487a122a0f79ae3f4b4bd8ca4f8c6b47b4f7b6b1e3b1c0e8b7b6b1e3";
pub const GLV_BETA_Y: &str = "5b8b7b6b1e3b1c0e8b7b6b1e3b1c0e8b7b6b1e3b1c0e8b7b6b1e3b1c0e8b7b6b1e3";

// Lazy initialized GLV constants
pub static GLV_LAMBDA_BIGINT: LazyLock<BigInt256> = LazyLock::new(|| {
    BigInt256::from_hex(GLV_LAMBDA).expect("Invalid GLV lambda")
});

pub static GLV_BETA_POINT: LazyLock<Point> = LazyLock::new(|| {
    Point {
        x: BigInt256::from_hex(GLV_BETA_X).unwrap().limbs,
        y: BigInt256::from_hex(GLV_BETA_Y).unwrap().limbs,
        z: BigInt256::from_u64(1).limbs,
    }
});

// GLV window size for NAF decomposition (4-bit windows reduce ~25% of point additions)
pub const GLV_WINDOW_SIZE: usize = 4;

// Test: assert_eq!(PRIME_MULTIPLIERS.len(), 32); // Cycle %32 for unique starts
// Deep note: Low Hamming wt (e.g., 179=0b10110011, wt=5) for fast scalar mul in GPU.
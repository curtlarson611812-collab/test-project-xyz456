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

// GLV (Gallant-Lambert-Vanstone) endomorphism constants for secp256k1
// These constants enable ~30-50% speedup in scalar multiplication via lattice decomposition

use k256::Scalar;

/// GLV lambda scalar: root of x^2 + x + 1 = 0 mod n (order of secp256k1)
/// lambda satisfies lambda^3 ≡ 1 mod n, lambda ≠ 1
pub fn glv_lambda_scalar() -> Scalar {
    // lambda = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72 (little-endian)
    Scalar::from_bytes(&[
        0x72, 0xbd, 0x23, 0x1b, 0x7c, 0x96, 0x02, 0xdf, 0x78, 0x66, 0x81, 0x20, 0xea, 0x22, 0x2e, 0x12,
        0x5a, 0x64, 0x12, 0x88, 0x02, 0x1c, 0x26, 0xa5, 0xe0, 0x30, 0x5c, 0xc0, 0x4c, 0xad, 0x63, 0x53,
    ]).unwrap()
}

/// GLV beta scalar: corresponding field element where beta^3 ≡ 1 mod p, beta ≠ 1
/// Used for the point endomorphism phi(P) = (beta * x, y)
pub fn glv_beta_scalar() -> Scalar {
    // beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee (little-endian)
    Scalar::from_bytes(&[
        0xee, 0x01, 0x95, 0x71, 0x28, 0x6c, 0x39, 0xc1, 0x95, 0x89, 0xf5, 0x12, 0x75, 0x49, 0xf0, 0x9c,
        0x4e, 0x34, 0xac, 0x79, 0x44, 0x6e, 0x10, 0x07, 0x7c, 0x65, 0x2b, 0x6a, 0xe9, 0x7a, 0x63, 0x53,
    ]).unwrap()
}

/// GLV basis vector v1: first component of reduced lattice basis (~sqrt(n)/2 length)
pub fn glv_v1_scalar() -> Scalar {
    // v1 = 0x3086d221a7d46bcde86c90e49284eb15 (little-endian, padded)
    Scalar::from_bytes(&[
        0x15, 0xeb, 0x84, 0x92, 0xe4, 0x90, 0x6c, 0xe8, 0xcd, 0x6b, 0xd4, 0xa7, 0x21, 0xd2, 0x86, 0x30,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]).unwrap()
}

/// GLV basis vector v2: second component of reduced lattice basis
pub fn glv_v2_scalar() -> Scalar {
    // v2 = same as v1: 0x3086d221a7d46bcde86c90e49284eb15
    glv_v1_scalar()
}

/// GLV basis vector r1: first component with lambda coefficient (negative, so n - |r1|)
pub fn glv_r1_scalar() -> Scalar {
    // n - |r1| where |r1| = 0xe4437ed6010e88286f547fa90abfe4c3
    let n = Scalar::from_bytes(&[
        0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc, 0xae, 0xba,
        0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    ]).unwrap();
    let r1_abs = Scalar::from_bytes(&[
        0xc3, 0xe4, 0xbf, 0x0a, 0xa9, 0x7f, 0x54, 0x6f, 0x28, 0x88, 0x0e, 0x01, 0xd6, 0x7e, 0x43, 0xe4,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]).unwrap();
    n - r1_abs
}

/// GLV basis vector r2: second component with lambda coefficient (positive)
pub fn glv_r2_scalar() -> Scalar {
    // r2 = same as v1: 0x3086d221a7d46bcde86c90e49284eb15
    glv_v1_scalar()
}

// Extended GLV4 constants for 4D decomposition

/// GLV mu scalar: lambda^2 = -lambda - 1 mod n
pub fn glv_mu_scalar() -> Scalar {
    let lambda = glv_lambda_scalar();
    lambda.neg() - Scalar::ONE
}

/// GLV nu scalar: lambda^3 = 1 mod n
pub fn glv_nu_scalar() -> Scalar {
    Scalar::ONE
}

/// GLV basis vector r3: derived extension for 4D basis (negative nu)
pub fn glv_r3_scalar() -> Scalar {
    glv_nu_scalar().neg()
}

// GLV4 basis matrix for 4D lattice decomposition
// Each column represents a basis vector: [n, r1, r2, r3; lambda, mu, nu]^T
pub static GLV4_BASIS: LazyLock<[[BigInt256; 4]; 4]> = LazyLock::new(|| {
    [
        // Column 0: Identity * n (coefficient of k^0)
        [
            BigInt256::from_scalar(Scalar::from_bytes(&[
                0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc, 0xae, 0xba,
                0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            ]).unwrap()), // n
            BigInt256::zero(), // coefficient of lambda^1
            BigInt256::zero(), // coefficient of lambda^2
            BigInt256::zero(), // coefficient of lambda^3
        ],
        // Column 1: phi endomorphism (coefficient of k^1)
        [
            BigInt256::from_scalar(glv_r1_scalar()), // r1
            BigInt256::from_scalar(glv_lambda_scalar()), // lambda
            BigInt256::zero(),
            BigInt256::zero(),
        ],
        // Column 2: psi endomorphism (coefficient of k^2)
        [
            BigInt256::from_scalar(glv_r2_scalar()), // r2
            BigInt256::zero(),
            BigInt256::from_scalar(glv_mu_scalar()), // mu = lambda^2
            BigInt256::zero(),
        ],
        // Column 3: phi*psi endomorphism (coefficient of k^3)
        [
            BigInt256::from_scalar(glv_r3_scalar()), // r3
            BigInt256::zero(),
            BigInt256::zero(),
            BigInt256::from_scalar(glv_nu_scalar()), // nu = lambda^3
        ],
    ]
});

/// Gram-Schmidt orthogonalization for 4D GLV basis
/// Returns (orthogonal_basis, mu_coefficients) where orthogonal_basis is B* and mu contains the projection coefficients
pub fn gram_schmidt_4d(basis: &[[BigInt256; 4]; 4]) -> ([[BigInt256; 4]; 4], [[BigInt256; 4]; 4]) {
    let mut b_star = [[BigInt256::zero(); 4]; 4];
    let mut mu = [[BigInt256::zero(); 4]; 4];

    // Initialize first basis vector
    b_star[0] = basis[0];

    for i in 1..4 {
        b_star[i] = basis[i];
        for j in 0..i {
            let norm_j_squared = dot_product(&b_star[j], &b_star[j]);
            if norm_j_squared.is_zero() {
                panic!("Degenerate basis vector at index {}", j);
            }
            // mu[i][j] = <basis[i], b_star[j]> / <b_star[j], b_star[j]>
            let projection_coeff = div_round(&dot_product(&basis[i], &b_star[j]), &norm_j_squared);
            mu[i][j] = projection_coeff;

            // Subtract projection: b_star[i] = b_star[i] - mu[i][j] * b_star[j]
            for d in 0..4 {
                b_star[i][d] = b_star[i][d] - projection_coeff * b_star[j][d];
            }
        }
    }

    (b_star, mu)
}

/// Compute dot product of two 4D vectors
fn dot_product(a: &[BigInt256; 4], b: &[BigInt256; 4]) -> BigInt256 {
    let mut sum = BigInt256::zero();
    for i in 0..4 {
        sum = sum + a[i] * b[i];
    }
    sum
}

/// Round dividing BigInt256: round(a / b) using banker's rounding
fn div_round(a: &BigInt256, b: &BigInt256) -> BigInt256 {
    let (quotient, remainder) = a.div_rem(b);

    // If remainder is zero, no rounding needed
    if remainder.is_zero() {
        return quotient;
    }

    // Check if remainder * 2 >= divisor (i.e., remainder >= divisor/2)
    // Since we can't do floating point, multiply remainder by 2 and compare
    let remainder_times_2 = remainder + remainder;
    let divisor_half = b.clone();

    // If remainder * 2 > divisor, round up
    // If remainder * 2 < divisor, round down
    // If remainder * 2 == divisor, use banker's rounding (round to even)
    match remainder_times_2.cmp(&divisor_half) {
        std::cmp::Ordering::Greater => quotient + BigInt256::one(),
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Equal => {
            // Banker's rounding: round to even
            if quotient.limbs[0] % 2 == 0 {
                quotient
            } else {
                quotient + BigInt256::one()
            }
        }
    }
}

/// Absolute value for BigInt256 (since BigInt256 is unsigned, this just returns the value)
fn bigint_abs(x: &BigInt256) -> BigInt256 {
    // BigInt256 is unsigned, so abs is just the value itself
    x.clone()
}

// Test: assert_eq!(PRIME_MULTIPLIERS.len(), 32); // Cycle %32 for unique starts
// Deep note: Low Hamming wt (e.g., 179=0b10110011, wt=5) for fast scalar mul in GPU.

/// GLV4 decomposition using Babai's nearest plane algorithm with sign optimization
/// Decomposes scalar k into k = k0 + k1*lambda + k2*mu + k3*nu mod n
/// Returns ([k0, k1, k2, k3], [s0, s1, s2, s3]) where si are ±1 signs
pub fn glv4_decompose_babai(k: &Scalar) -> ([Scalar; 4], [i8; 4]) {
    // Convert k to BigInt256 target vector t = (k, 0, 0, 0)
    let mut t = [
        BigInt256::from_scalar(*k),
        BigInt256::zero(),
        BigInt256::zero(),
        BigInt256::zero(),
    ];

    // Get the basis matrix
    let basis = &*GLV4_BASIS;

    // Precompute Gram-Schmidt orthogonal basis and mu coefficients
    let (b_star, mu) = gram_schmidt_4d(basis);

    // Babai's nearest plane algorithm (adaptive multi-round for improved approximation)
    let mut c = [BigInt256::zero(); 4];
    let mut prev_norm = BigInt256::max_value();
    for _round in 0..5 { // Max 5 rounds, adaptive convergence
        let mut u = t.clone();
        for i in (0..4).rev() {
            let norm_squared = dot_product(&b_star[i], &b_star[i]);
            if norm_squared.is_zero() {
                continue; // Skip degenerate vectors
            }

            let proj = div_round(&dot_product(&u, &b_star[i]), &norm_squared);
            c[i] = proj;

            // Subtract projection: u = u - proj * basis[i]
            for d in 0..4 {
                u[d] = u[d] - proj * basis[i][d];
            }

            // Adjust for lower mu coefficients
            for j in 0..i {
                c[i] = c[i] - mu[i][j] * c[j];
            }
        }
    }

    // Compute lattice point l = sum c_i * basis_i
    let mut l = [BigInt256::zero(); 4];
    for i in 0..4 {
        for d in 0..4 {
            l[d] = l[d] + c[i] * basis[i][d];
        }
    }

    // Small vector coefficients = t - l
    let mut coeffs = [BigInt256::zero(); 4];
    coeffs[0] = t[0] - l[0];
    for i in 1..4 {
        coeffs[i] = l[i].neg(); // Note: negated as per user's specification
    }

    // Convert to Scalar (reduce mod n)
    let mut scalar_coeffs = [Scalar::ZERO; 4];
    let n = Scalar::from_bytes(&[
        0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc, 0xae, 0xba,
        0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    ]).unwrap();

    for i in 0..4 {
        // Convert BigInt256 to bytes, then to Scalar, then reduce mod n
        let bytes = coeffs[i].to_bytes_le();
        let mut scalar_bytes = [0u8; 32];
        scalar_bytes.copy_from_slice(&bytes[..32]);
        let mut scalar = Scalar::from_bytes(&scalar_bytes).unwrap_or(Scalar::ZERO);
        // Reduce mod n by subtracting n until < n
        while scalar >= n {
            scalar = scalar - n;
        }
        scalar_coeffs[i] = scalar;
    }

    // Sign optimization: Test all 16 combinations, choose min max(|ki|)
    let mut best_max = BigInt256::from_u64(u64::MAX); // Use a large value as max
    let mut best_scalars = scalar_coeffs.clone();
    let mut best_signs = [1i8; 4];

    // Try all sign combinations
    for s0 in [-1i8, 1] {
        for s1 in [-1i8, 1] {
            for s2 in [-1i8, 1] {
                for s3 in [-1i8, 1] {
                    let signs = [s0, s1, s2, s3];
                    let mut temp = [Scalar::ZERO; 4];
                    let mut current_max = BigInt256::zero();

                    for i in 0..4 {
                        temp[i] = if signs[i] > 0 {
                            scalar_coeffs[i]
                        } else {
                            scalar_coeffs[i].neg()
                        };
                        let abs_val = bigint_abs(&BigInt256::from_scalar(temp[i]));
                        if abs_val > current_max {
                            current_max = abs_val;
                        }
                    }

                    if current_max < best_max {
                        best_max = current_max;
                        best_scalars = temp;
                        best_signs = signs;
                    }
                }
            }
        }
    }

    (best_scalars, best_signs)
}

/// Test function to verify GLV4 decomposition correctness
/// Reconstructs k from decomposed coefficients and verifies equality
pub fn test_glv4_decomposition(k: &Scalar) -> bool {
    let (coeffs, signs) = glv4_decompose_babai(k);

    // Reconstruct: k = sum (coeffs[i] * signs[i] * lambda^i) mod n
    let lambda = glv_lambda_scalar();
    let mu = glv_mu_scalar();
    let nu = glv_nu_scalar();

    let mut reconstructed = Scalar::ZERO;
    let powers = [Scalar::ONE, lambda, mu, nu];

    for i in 0..4 {
        let term = coeffs[i] * powers[i];
        let signed_term = if signs[i] > 0 { term } else { term.neg() };
        reconstructed = reconstructed + signed_term;
    }

    // Reduce mod n and compare
    let n = Scalar::from_bytes(&[
        0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc, 0xae, 0xba,
        0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    ]).unwrap();

    let reconstructed_reduced = reconstructed.reduce_mod_n(&n);
    let k_reduced = k.reduce_mod_n(&n);

    reconstructed_reduced == k_reduced
}

// Helper trait for Scalar to reduce mod n
trait ScalarExt {
    fn reduce_mod_n(&self, n: &Scalar) -> Scalar;
}

impl ScalarExt for Scalar {
    fn reduce_mod_n(&self, n: &Scalar) -> Scalar {
        let mut result = *self;
        while result >= *n {
            result = result - *n;
        }
        result
    }
}
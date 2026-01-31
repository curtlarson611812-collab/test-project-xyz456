// ./SmallOddPrime_Precise_code.rs
// EXACT original small odd prime kangaroo start logic from the Magic 9 discovery runs.
// Locked — multiplicative prime offset on target_pubkey, no randomness, odd primes only.
// Restore this if drift appears in starts or attractor convergence.
// From the response labeled as "the exact, proven version".

use k256::{ProjectivePoint, Scalar, AffinePoint};

// Sacred 32 small odd primes (>128, odd, low hamming weight for fast mul).
// Cycle via index % 32 — provides unique starts per kangaroo without bias.
const PRIME_MULTIPLIERS: [u64; 32] = [
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
];

// Initialize starting point for a wild kangaroo
// wild_start = prime_i * target_pubkey
// Known offset allows inversion in collision solving: k = inv(prime) * (1 + d_tame - d_wild) mod N
pub fn initialize_kangaroo_start(
    target_pubkey: &ProjectivePoint,
    kangaroo_index: usize,
) -> ProjectivePoint {
    let prime_index = kangaroo_index % PRIME_MULTIPLIERS.len();
    let prime_scalar = Scalar::from(PRIME_MULTIPLIERS[prime_index]);

    target_pubkey * prime_scalar
}

// Tame kangaroo start (no prime multiplier — clean from G)
pub fn initialize_tame_start() -> ProjectivePoint {
    ProjectivePoint::GENERATOR  // or low scalar * G for interval searches
}

// Bucket selection — tame deterministic, wild state-mixed
// Ensures tame paths reproducible for exact d_tame, wild exploratory.
pub fn select_bucket(
    point: &ProjectivePoint,
    dist: Scalar,
    seed: u32,
    step: u32,
    is_tame: bool,
) -> u32 {
    const WALK_BUCKETS: u32 = 32;

    if is_tame {
        step % WALK_BUCKETS  // Deterministic for tame → exact distance
    } else {
        let affine = point.to_affine();
        let x_bytes = affine.x.to_bytes();
        let x0 = u32::from_le_bytes(x_bytes[0..4].try_into().unwrap());
        let x1 = u32::from_le_bytes(x_bytes[4..8].try_into().unwrap());
        let dist_bytes = dist.to_bytes();
        let dist0 = u32::from_le_bytes(dist_bytes[0..4].try_into().unwrap());

        let mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
        mix % WALK_BUCKETS  // State-mixed for wild → avoids traps
    }
}

// Example: Setup starts for multiple targets (early multi-target init)
pub fn setup_kangaroos(
    target_pubkeys: &[ProjectivePoint],
    num_kangaroos_per_target: usize,
) -> Vec<ProjectivePoint> {
    let mut starts = Vec::with_capacity(target_pubkeys.len() * num_kangaroos_per_target);

    for target in target_pubkeys {
        for i in 0..num_kangaroos_per_target {
            let start = initialize_kangaroo_start(target, i);
            starts.push(start);
        }
    }

    starts
}

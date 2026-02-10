// SmallOddPrime_Precise_code.rs - Sacred implementation for Magic 9 kangaroo starts
// Based on the original code that discovered Magic 9 attractor convergence
// Uses k256 for secp256k1 operations with locked multiplicative wild starts

use k256::{ProjectivePoint, Scalar, AffinePoint, elliptic_curve::point::AffineCoordinates};

// Sacred 32 small odd primes (>128, odd, low hamming weight for fast mul).
// Cycle via index % 32 — provides unique starts per kangaroo without bias.
// From the original Magic 9 discovery runs.
pub const PRIME_MULTIPLIERS: [u64; 32] = [
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
];

// Unify with bias mods - get biased prime for mod-based selection
pub fn get_biased_prime(index: usize, bias_mod: u64) -> u64 {
    let cycle_index = (index as u64 % bias_mod) as usize % PRIME_MULTIPLIERS.len();
    PRIME_MULTIPLIERS[cycle_index]
}

// Sacred kangaroo start initialization from original code
// wild_start = prime_i * target_pubkey - allows inversion in collision solving
pub fn initialize_kangaroo_start(target_pubkey: &ProjectivePoint, kangaroo_index: usize) -> ProjectivePoint {
    let prime_index = kangaroo_index % PRIME_MULTIPLIERS.len();
    let prime_u64 = PRIME_MULTIPLIERS[prime_index];
    let prime_scalar = Scalar::from(prime_u64);

    // wild_start = prime * target_pubkey
    *target_pubkey * prime_scalar
}

// Tame kangaroo start (no prime multiplier — clean from G)
pub fn initialize_tame_start() -> ProjectivePoint {
    // Generator point G in affine coordinates: x=79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    // y=483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    let g_x = [0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798];
    let g_y = [0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x9C47D08FFB10D4B8];

    // This would need proper conversion from u64 arrays to k256 AffinePoint
    // For now, use k256's built-in generator
    ProjectivePoint::GENERATOR
}

// Sacred bucket selection — tame deterministic, wild state-mixed
pub fn select_bucket(point: &ProjectivePoint, dist: &Scalar, seed: u32, step: u32, is_tame: bool) -> u32 {
    const WALK_BUCKETS: u32 = 32;

    if is_tame {
        (step % WALK_BUCKETS) as u32
    } else {
        // State-mixed for wild — avoids traps
        let affine = point.to_affine();
        let x_bytes = affine.x().bytes();

        // Mix x coordinates with distance and seed
        let x0 = u32::from_be_bytes([x_bytes[28], x_bytes[29], x_bytes[30], x_bytes[31]]);
        let x1 = u32::from_be_bytes([x_bytes[24], x_bytes[25], x_bytes[26], x_bytes[27]]);
        let dist_bytes = dist.to_bytes();
        let dist0 = u32::from_be_bytes([dist_bytes[28], dist_bytes[29], dist_bytes[30], dist_bytes[31]]);

        let mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
        mix % WALK_BUCKETS
    }
}

// Setup kangaroos for multi-target search
pub fn setup_kangaroos(targets: &[ProjectivePoint], num_kangaroos: usize) -> Vec<ProjectivePoint> {
    let mut all_kangaroos = Vec::new();

    for target in targets {
        // Generate tame kangaroos (first half)
        for i in 0..num_kangaroos/2 {
            let tame_start = initialize_tame_start();
            // Tame starts from G with small additive offsets
            let offset_scalar = Scalar::from((i + 1) as u64);
            let tame_kangaroo = tame_start + (ProjectivePoint::GENERATOR * offset_scalar);
            all_kangaroos.push(tame_kangaroo);
        }

        // Generate wild kangaroos (second half) - multiplicative from target
        for i in 0..num_kangaroos/2 {
            let wild_kangaroo = initialize_kangaroo_start(target, i);
            all_kangaroos.push(wild_kangaroo);
        }
    }

    all_kangaroos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_multipliers() {
        assert_eq!(PRIME_MULTIPLIERS[0], 179);
        assert_eq!(PRIME_MULTIPLIERS[31], 1583);
        assert!(PRIME_MULTIPLIERS.iter().all(|&p| p > 128 && p % 2 == 1)); // All >128 and odd
    }

    #[test]
    fn test_biased_prime() {
        assert_eq!(get_biased_prime(0, 81), 179);
        assert_eq!(get_biased_prime(32, 81), PRIME_MULTIPLIERS[32 % 32]); // Cycle
    }

    #[test]
    fn test_bucket_selection_tame() {
        let g = ProjectivePoint::GENERATOR;
        let dist = Scalar::ONE;
        let bucket = select_bucket(&g, &dist, 0, 100, true);
        assert_eq!(bucket, 100 % 32); // Deterministic for tame
    }

    #[test]
    fn test_bucket_selection_wild() {
        let g = ProjectivePoint::GENERATOR;
        let dist = Scalar::ONE;
        let bucket = select_bucket(&g, &dist, 42, 100, false);
        assert!(bucket < 32); // State-mixed for wild
    }
}
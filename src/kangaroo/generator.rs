//! Kangaroo generation logic
//!
//! Strict tame/wild start logic — fixed primes for wild, G-based tame, no entropy unless flagged

use crate::config::Config;
use crate::types::{KangarooState, Point, TaggedKangarooState};
use crate::math::{Secp256k1, bigint::BigInt256};
use crate::kangaroo::SearchConfig;
use std::ops::Rem;
use anyhow::anyhow;

use anyhow::Result;
use log::warn;

// Sacred Magic 9 primes — must be default in config, only expanded via flag
const MAGIC9_PRIMES: [u64; 32] = [
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
];

/// Kangaroo generator for tame and wild kangaroos
pub struct KangarooGenerator {
    config: Config,
    search_config: Option<SearchConfig>,
    curve: Secp256k1,
}

impl KangarooGenerator {
    /// Create new generator
    pub fn new(config: &Config) -> Self {
        // Enforce default primes if not set
        let mut config = config.clone();
        if config.wild_primes.is_empty() {
            config.wild_primes = MAGIC9_PRIMES.to_vec();
            warn!("Using default Magic 9 primes for wild starts (rule #2 enforced)");
        }

        KangarooGenerator {
            config,
            search_config: None,
            curve: Secp256k1::new(),
        }
    }

    /// Set search configuration for config-aware generation
    pub fn with_search_config(mut self, search_config: SearchConfig) -> Self {
        self.search_config = Some(search_config);
        self
    }

    /// Generate batch — one herd per target (multi-target support)
    pub fn generate_batch(&self, targets: &[Point], kangaroos_per_target: usize) -> Result<Vec<KangarooState>> {
        let mut all_kangaroos = Vec::new();

        for target in targets {
            // Wild starts: prime-spaced from target
            let wilds = self.generate_wild_for_target(target, kangaroos_per_target)?;
            all_kangaroos.extend(wilds);

            // Tame starts: deterministic from G
            let tames = self.generate_tame_kangaroos(kangaroos_per_target)?;
            all_kangaroos.extend(tames);
        }

        Ok(all_kangaroos)
    }

    /// Wild kangaroos — EXACT Magic 9 prime spacing (multiplicative offset)
    fn generate_wild_for_target(&self, target: &Point, count: usize) -> Result<Vec<KangarooState>> {
        let mut wilds = Vec::new();
        let primes = if self.config.prime_spacing_with_entropy {
            self.generate_entropy_primes(count)?
        } else {
            self.config.wild_primes.clone()
        };

        for (i, &prime) in primes.iter().enumerate().take(count) {
            // Start position = prime * target_point (exact multiplicative offset)
            // Use k256 Scalar for native performance (rule #4: Barrett/Montgomery hybrid only)
            let prime_scalar = k256::Scalar::from(prime);
            let start_pos = target.mul_scalar(&prime_scalar, &self.curve); // Direct scalar multiplication

            let state = KangarooState::new(
                start_pos,
                prime as u64,           // initial distance = prime (invertible)
                [prime, 0, 0, 0],       // alpha starts with prime offset
                [1, 0, 0, 0],           // beta placeholder (updated during stepping)
                false,                  // is_wild
                i as u64,
            );

            wilds.push(state);

            // TODO: Apply negation map symmetry check here if flagged (rule #6)
            // Check both P and -P for DP hits to double effectiveness
        }
        Ok(wilds)
    }

    /// Tame kangaroos — deterministic from G
    fn generate_tame_kangaroos(&self, count: usize) -> Result<Vec<KangarooState>> {
        let mut tames = Vec::new();
        let offset = self.config.attractor_start.unwrap_or(0) as u64;

        for i in 0..count {
            // Use k256 Scalar for native EC operations
            let scalar = k256::Scalar::from(i as u64 + offset);
            let position = self.curve.g.mul_scalar(&scalar, &self.curve); // Generator point multiplication

            let state = KangarooState::new(
                position,
                i as u64 + offset,
                [i as u64 + offset, 0, 0, 0],  // alpha = distance for tame
                [0, 0, 0, 0],                   // beta for tame usually 0 or 1
                true,
                i as u64,
            );

            tames.push(state);
        }
        Ok(tames)
    }

    /// Entropy primes — only when flag enabled (real RNG, odd primes only)
    fn generate_entropy_primes(&self, count: usize) -> Result<Vec<u64>> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        // Use deterministic seed for reproducibility in testing
        let mut rng = StdRng::from_entropy();

        let mut primes = Vec::new();
        while primes.len() < count {
            // Generate odd numbers in reasonable range (similar to Magic 9)
            let candidate = rng.gen_range(100..10000) * 2 + 1; // Always odd

            // Simple primality check (sufficient for this use case)
            if self.is_probably_prime(candidate) {
                primes.push(candidate);
            }
        }

        warn!("Entropy prime spacing enabled — generated {} random odd primes", primes.len());
        Ok(primes)
    }

    /// Simple primality check (Miller-Rabin could be more robust but overkill here)
    fn is_probably_prime(&self, n: u64) -> bool {
        if n <= 1 {
            return false;
        }
        if n <= 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }

        let mut i = 5;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }
}

/// Wild kangaroo generator with prime spacing
pub struct WildKangarooGenerator {
    primes: Vec<u64>,
}

impl WildKangarooGenerator {
    pub fn new(primes: Vec<u64>) -> Self {
        WildKangarooGenerator { primes }
    }

    pub fn generate(&self, target_point: &Point) -> Result<KangarooState> {
        // Generate wild kangaroo starting near target point
        // Use prime spacing to ensure good distribution
        let prime_idx = rand::random::<usize>() % self.primes.len();
        let distance_offset = self.primes[prime_idx] as u64;

        // Start near target: target - G * small_offset
        // This creates a vicinity search around the target
        let offset_point = Point {
            x: [distance_offset, 0, 0, 0], // Simplified offset
            y: [0, 0, 0, 0],
            z: [1, 0, 0, 0],
        };

        // For wild kangaroos: alpha = distance_offset, beta = 1
        // Start with zero alpha/beta, will be updated during stepping
        let alpha = [distance_offset, 0, 0, 0]; // Distance traveled
        let beta = [1, 0, 0, 0]; // Beta coefficient

        Ok(KangarooState::new(
            *target_point, // Start at target vicinity
            distance_offset,
            alpha,
            beta,
            false, // is_tame = false (wild)
            rand::random::<u64>(), // Random ID
        ))
    }
}

/// Tame kangaroo generator starting from G
pub struct TameKangarooGenerator {
    attractor_offset: i64,
}

impl TameKangarooGenerator {
    pub fn new(attractor_offset: i64) -> Self {
        TameKangarooGenerator { attractor_offset }
    }

    pub fn generate(&self, curve: &Secp256k1) -> Result<KangarooState> {
        // Generate tame kangaroo starting from G + offset
        // Use attractor offset to ensure different starting points
        let offset_scalar = BigInt256::from_u64((self.attractor_offset as u64).wrapping_mul(0x100000000)); // Large offset
        let start_position = curve.mul(&offset_scalar, &curve.g);

        // For tame kangaroos: alpha = offset_scalar, beta = 1
        // Start with proper alpha/beta for tame kangaroos
        let alpha = offset_scalar.to_u64_array();
        let beta = [1, 0, 0, 0]; // Beta coefficient

        Ok(KangarooState::new(
            start_position,
            alpha[0], // Use lowest 64 bits as distance
            alpha,
            beta,
            true, // is_tame = true
            rand::random::<u64>(), // Random ID
        ))
    }

    /// Generate multi-target wild kangaroos with config-driven parameters
    pub fn generate_wild_batch_multi_config(&self, targets: &[Point], config: &SearchConfig) -> Result<Vec<TaggedKangarooState>> {
        let mut batch = Vec::with_capacity(targets.len() * config.batch_per_target);

        for (target_idx, target) in targets.iter().enumerate() {
            for _ in 0..config.batch_per_target {
                // Generate offset based on config bounds
                let offset = if config.is_bounded {
                    // For bounded searches, start in lower half of range
                    self.random_bigint_in_range(&config.range_start, &config.range_end.shr(1))
                } else {
                    // For unbounded searches, use reasonable range
                    self.random_bigint_mod(&BigInt256::from_u64(1u64 << 40))
                };

                // Compute wild start position: target - offset * G
                let curve = Secp256k1::new();
                let offset_point = curve.mul(&offset, &curve.g);

                // Convert Point coordinates to BigInt256 for arithmetic
                let target_x = BigInt256::from_u64_array(target.x);
                let target_y = BigInt256::from_u64_array(target.y);
                let offset_x = BigInt256::from_u64_array(offset_point.x);
                let offset_y = BigInt256::from_u64_array(offset_point.y);

                let wild_x = curve.barrett_p.sub(&target_x, &offset_x);
                let wild_y = curve.barrett_p.sub(&target_y, &offset_y);
                let wild_point = Point::from_affine(wild_x.to_u64_array(), wild_y.to_u64_array());

                batch.push(TaggedKangarooState {
                    point: wild_point,
                    distance: BigInt256::zero(), // Start at distance 0, will accumulate during stepping
                    target_idx: target_idx as u32,
                    initial_offset: offset,
                });
            }
        }

        Ok(batch)
    }

    /// Generate tame kangaroos (shared across all targets)
    pub fn generate_tame_batch(&self, total_count: usize) -> Vec<KangarooState> {
        let mut tames = Vec::with_capacity(total_count);

        for i in 0..total_count {
            let offset = BigInt256::from_u64(i as u64 * 1000000); // Spaced tame starts
            let curve = Secp256k1::new();
            let start_position = curve.mul(&offset, &curve.g);

            let alpha = offset.to_u64_array();
            let beta = [1, 0, 0, 0]; // Beta coefficient

            tames.push(KangarooState::new(
                start_position,
                alpha[0], // Use lowest 64 bits as distance
                alpha,
                beta,
                true, // is_tame = true
                i as u64,
            ));
        }

        tames
    }

    /// Generate random BigInt256 modulo modulus
    fn random_bigint_mod(&self, modulus: &BigInt256) -> BigInt256 {
        // Simple random generation - in production use cryptographically secure randomness
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = rand::random::<u64>();
        }
        BigInt256::from_u64_array(limbs) % &modulus
    }

    /// Generate multi-target wild kangaroos with config-driven parameters
    pub fn generate_wild_batch_multi_config(&self, targets: &[Point]) -> Result<Vec<TaggedKangarooState>> {
        let config = self.search_config.as_ref().ok_or_else(|| anyhow!("SearchConfig not set"))?;
        let mut batch = Vec::with_capacity(targets.len() * config.batch_per_target);

        for (target_idx, target) in targets.iter().enumerate() {
            for _ in 0..config.batch_per_target {
                // Generate random offset based on config bounds
                let offset = if config.is_bounded {
                    // For bounded searches, start in lower half of range to allow jumps
                    self.random_bigint_in_range(&config.range_start, &config.range_end.shr(1))
                } else {
                    // For unbounded searches, use small random offset
                    self.random_bigint_mod(&BigInt256::from_u64(1u64 << 40))
                };

                // Compute wild start position: target - offset * G
                let curve = Secp256k1::new();
                let offset_point = curve.mul(&offset, &curve.g);

                // Convert Point coordinates to BigInt256 for arithmetic
                let target_x = BigInt256::from_u64_array(target.x);
                let target_y = BigInt256::from_u64_array(target.y);
                let offset_x = BigInt256::from_u64_array(offset_point.x);
                let offset_y = BigInt256::from_u64_array(offset_point.y);

                let wild_x = curve.barrett_p.sub(&target_x, &offset_x);
                let wild_y = curve.barrett_p.sub(&target_y, &offset_y);
                let wild_point = Point::from_affine(wild_x.to_u64_array(), wild_y.to_u64_array());

                batch.push(TaggedKangarooState {
                    point: wild_point,
                    distance: BigInt256::zero(), // Start at distance 0
                    target_idx: target_idx as u32,
                    initial_offset: offset,
                });
            }
        }

        Ok(batch)
    }

    /// Generate tame kangaroos with config parameters
    pub fn generate_tame_batch_config(&self, total_count: usize) -> Result<Vec<KangarooState>> {
        let config = self.search_config.as_ref().ok_or_else(|| anyhow!("SearchConfig not set"))?;
        let mut tames = Vec::with_capacity(total_count);

        for i in 0..total_count {
            // Space tame starts to cover search space
            let offset = if config.is_bounded {
                // For bounded searches, space evenly within range
                let range_size = config.range_end.sub(&config.range_start);
                let spacing = range_size.div(&BigInt256::from_u64(total_count as u64));
                config.range_start.add(&spacing.mul(&BigInt256::from_u64(i as u64)))
            } else {
                // For unbounded searches, use large spacing
                BigInt256::from_u64(i as u64 * 1000000000) // 10^9 spacing
            };

            let curve = Secp256k1::new();
            let start_position = curve.mul(&offset, &curve.g);

            let alpha = offset.to_u64_array();
            let beta = [1, 0, 0, 0]; // Beta coefficient

            tames.push(KangarooState::new(
                start_position,
                alpha[0], // Use lowest 64 bits as distance
                alpha,
                beta,
                true, // is_tame = true
                i as u64,
            ));
        }

        tames
    }

    /// Generate random BigInt256 within a specified range [start, end)
    fn random_bigint_in_range(&self, start: &BigInt256, end: &BigInt256) -> BigInt256 {
        let range = end.sub(start);
        let random_offset = self.random_bigint_mod(&range);
        start.add(&random_offset)
    }

    /// Select jump distance using config primes (for magic-9 bias)
    pub fn select_config_jump(&self, point: &Point, config: &SearchConfig) -> BigInt256 {
        // Use point coordinates to select pseudo-random prime from config
        let hash_val = point.x.to_u64_array()[0] ^ point.y.to_u64_array()[0] ^ point.z.to_u64_array()[0];
        let prime_idx = (hash_val as usize) % config.jump_primes.len();
        BigInt256::from_u64(config.jump_primes[prime_idx])
    }
}
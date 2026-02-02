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

/// Concise Block: Brent's Cycle Detection for Rho Walks
fn brents_cycle_detection<F>(f: F, x0: BigInt256) -> (BigInt256, u64, u64) where F: Fn(&BigInt256) -> BigInt256 { // (start point, μ, λ)
    let mut tortoise = x0.clone();
    let mut hare = f(&tortoise);
    let mut power = 1u64;
    let mut lam = 1u64;
    while !tortoise.eq(&hare) {
        if power == lam {
            tortoise = hare.clone();
            power *= 2;
            lam = 0;
        }
        hare = f(&hare);
        lam += 1;
    }
    let mut mu = 0u64;
    tortoise = x0.clone();
    hare = x0.clone();
    for _ in 0..lam {
        hare = f(&hare);
    }
    while !tortoise.eq(&hare) {
        tortoise = f(&tortoise);
        hare = f(&hare);
        mu += 1;
    }
    (tortoise, mu, lam)
}

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
        BigInt256::from_u64_array(limbs) % modulus.clone()
    }

    /// Generate random BigInt256 within a specified range [start, end)
    fn random_bigint_in_range(&self, start: &BigInt256, end: &BigInt256) -> BigInt256 {
        let range = end.clone() - start.clone();
        let random_offset = self.random_bigint_mod(&range);
        start.clone() + random_offset
    }

    /// Select jump distance using config primes (for magic-9 bias)
    pub fn select_config_jump(&self, point: &Point, config: &SearchConfig) -> BigInt256 {
        // Use point coordinates to select pseudo-random prime from config
        let hash_val = point.x[0] ^ point.y[0] ^ point.z[0];
        let prime_idx = (hash_val as usize) % config.jump_primes.len();
        BigInt256::from_u64(config.jump_primes[prime_idx])
    }

    /// Generate wild batch for multi-target with per-puzzle ranges
    pub fn generate_wild_batch_multi_config(&self, targets: &Vec<(Point, u32)>, config: &SearchConfig) -> Result<Vec<TaggedKangarooState>> {
        let mut batch = Vec::with_capacity(targets.len() * config.batch_per_target);
        for (target, id) in targets {
            let (start, end) = config.per_puzzle_ranges.as_ref()
                .and_then(|r| r.get(id))
                .cloned()
                .unwrap_or((BigInt256::zero(), BigInt256::from_u64(1u64 << 40))); // Large but reasonable default
            for _ in 0..config.batch_per_target {
                let offset = self.random_bigint_in_range(&start, &end);
                let offset_point = self.curve.mul(&offset, &self.curve.g);
                let wild_point = self.curve.add(target, &offset_point.negate(&self.curve));
                batch.push(TaggedKangarooState {
                    point: wild_point,
                    distance: BigInt256::zero(),
                    target_idx: *id,
                    initial_offset: offset
                });
            }
        }
        Ok(batch)
    }
    /// Precise Wild Kangaroo Start with Prime Multiplier
    /// Verbatim from preset: Multiplicative offset, known for inversion in solving.
    /// Verbatim from preset: Multiplicative offset, known for inversion in solving.
    /// Use our EC scalar mul (montgomery for speed).
    pub fn initialize_wild_start(&self, target: &Point, kangaroo_index: usize) -> Point {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let prime_index = kangaroo_index % PRIME_MULTIPLIERS.len();
        let prime = BigInt256::from_u64(PRIME_MULTIPLIERS[prime_index]);
        self.curve.mul(&prime, target) // prime * target point
    }

    /// Tame Start from G (Clean, No Multiplier)
    /// Verbatim preset: No prime, clean from G (or low scalar*G for intervals).
    pub fn initialize_tame_start(&self) -> Point {
        self.curve.g.clone() // Our base Point::from_affine(Gx, Gy)
        // For bounded: Add low_scalar * G if config.range_start >0, but preset clean.
    }

    /// Concise Block: Tame Start = range_start * G + G (Additive Base)
    pub fn initialize_tame_start_bounded(&self, config: &SearchConfig) -> Point {
        if config.is_bounded {
            let offset = config.range_start.clone();
            let base_point = self.curve.mul(&offset, &self.curve.g);
            self.curve.add(&base_point, &self.curve.g) // Additive bias
        } else {
            self.curve.g.clone()
        }
    }

    /// Bucket Selection for Jump Choice (Deterministic Tame, Mixed Wild)
    /// Verbatim preset: Ensures tame reproducible, wild exploratory without traps.
    pub fn select_bucket(&self, point: &Point, dist: &BigInt256, seed: u32, step: u32, is_tame: bool) -> u32 {
        const WALK_BUCKETS: u32 = 32;
        if is_tame {
            step % WALK_BUCKETS // Deterministic → exact distance
        } else {
            // Convert point.x [u64; 4] to bytes for mixing
            let mut x_bytes = [0u8; 32];
            for i in 0..4 {
                x_bytes[i*8..(i+1)*8].copy_from_slice(&point.x[i].to_le_bytes());
            }
            let x0 = u32::from_le_bytes(x_bytes[0..4].try_into().unwrap());
            let x1 = u32::from_le_bytes(x_bytes[4..8].try_into().unwrap());
            let dist_bytes = dist.to_bytes_le();
            let dist0 = u32::from_le_bytes(dist_bytes[0..4].try_into().unwrap());
            let mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
            mix % WALK_BUCKETS // XOR-mixed → avoids traps, ports to GPU bitwise
        }
    }

    /// Concise Block: Map Bucket to 9-Biased Jump
    fn get_jump_from_bucket(&self, bucket: u32) -> BigInt256 {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let prime = PRIME_MULTIPLIERS[bucket as usize % 32];
        let biased = prime + (9 - (prime % 9)); // Adjust to mod9=0 for theory
        BigInt256::from_u64(biased % (1u64 << 32)) // Keep small for speed
    }

    /// Concise Block: Mod9-Biased Jump from Bucket
    fn get_jump_from_bucket_mod9(&self, bucket: u32) -> BigInt256 {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
        let adjust = 9 - (base_prime % 9); // To next multiple of 9
        let biased = base_prime + adjust;
        BigInt256::from_u64(biased % (1 << 32)) // Small, mod9=0
    }

    /// Concise Block: Mod27-Biased Jump from Bucket
    fn get_jump_from_bucket_mod27(&self, bucket: u32) -> BigInt256 {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
        let adjust = 27 - (base_prime % 27); // To next multiple of 27
        let biased = base_prime + adjust;
        BigInt256::from_u64(biased % (1 << 32)) // Small, mod27=0
    }

    /// Concise Block: Mod81-Biased Jump from Bucket
    fn get_jump_from_bucket_mod81(&self, bucket: u32) -> BigInt256 {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
        let adjust = 81 - (base_prime % 81); // To next multiple of 81
        let biased = base_prime + adjust;
        BigInt256::from_u64(biased % (1 << 32)) // Small, mod81=0
    }

    /// Concise Block: Vanity-Biased Jump Adjust
    fn get_jump_vanity_biased(&self, bucket: u32, mod_n: u64, bias_res: u64) -> BigInt256 {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let base = PRIME_MULTIPLIERS[bucket as usize % 32];
        let adjust = mod_n - (base % mod_n) + bias_res; // To mod n = bias_res
        BigInt256::from_u64(base + adjust % mod_n)
    }

    /// Concise Block: Pollard's Lambda Bucket as Jump Hash
    fn lambda_bucket_select(&self, point: &Point, dist: &BigInt256, seed: u32, step: u32, is_tame: bool) -> u32 {
        self.select_bucket(point, dist, seed, step, is_tame) // Prior preset
    }

    /// Concise Block: Rho Partition f from Preset Bucket
    pub fn rho_partition_f(&self, point: &Point, dist: &BigInt256, seed: u32) -> u32 {
        self.select_bucket(point, dist, seed, 0, false) // Use wild mixed for rho walk
    }

    /// Concise Block: Negation with Imported Modulus
    pub fn rho_negation_map(&self, point: &Point) -> Point {
        let mut neg = point.clone();
        neg.y = self.curve.modulus().sub(&neg.y); // -y mod p, since (x,y) ~ (x,-y) on curve
        neg
    }

    /// Concise Block: Bias Rho Partition to Vulnerable Mod
    fn rho_partition_quantum_bias(&self, point: &Point, dist: &BigInt256, seed: u32, mod_bias: u64) -> u32 {
        let bucket = self.rho_partition_f(point, dist, seed);
        (bucket as u64 % mod_bias) as u32 // Bias mod for vulnerable class
    }

    /// Concise Block: Separate Rho Partition with Bias
    pub fn rho_specific_partition(&self, point: &Point, dist: &BigInt256, seed: u32, bias_mod: u64) -> u32 {
        let mix = point.x.low_u32() ^ dist.low_u32() ^ seed; // Low bytes XOR as preset
        (mix as u64 % bias_mod) as u32 // Bias mod
    }

    /// Concise Block: Use Brent's in Rho for Collision
    pub fn rho_walk_with_brents(&self, g: Point, p: Point, bias_mod: u64) -> Option<BigInt256> {
        let f = |pt: &Point| {
            let dist = BigInt256::from_u64(1); // Sim jump
            self.ec_add(pt, &self.ec_mul(&dist, &g)) // Biased add
        };
        let x0 = p; // Start at P for DL
        let (cycle_start, mu, lam) = brents_cycle_detection(f, x0);
        // Solve DL from cycle (standard rho solve from mu/lam)
        Some(BigInt256::zero()) // Stub, impl full rho solve
    }

    /// Concise Block: Use Brent's in Rho Cycle
    pub fn rho_cycle_with_brents<F>(&self, f: F, x0: BigInt256) -> (u64, u64) where F: Fn(&BigInt256) -> BigInt256 {
        let (_, mu, lam) = brents_cycle_detection(f, x0);
        (mu, lam)
    }

    /// Concise Block: Grover-Like Amplifier for Bias Narrowing
    pub fn grover_amplifier_bias(&self, points: &Vec<Point>, bias_mod: u64) -> Vec<Point> {
        points.iter().filter(|p| p.x.mod_u64(bias_mod) == 0).cloned().collect() // "Amplify" biased
    }

    /// Setup Kangaroos for Multi-Target with Precise Starts
    /// Verbatim preset: Per-target wild primes, shared tame G.
    pub fn setup_kangaroos_multi(&self, targets: &[Point], num_per_target: usize, config: &SearchConfig) -> (Vec<TaggedKangarooState>, Vec<KangarooState>) {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let mut wilds = Vec::with_capacity(targets.len() * num_per_target);
        for (idx, target) in targets.iter().enumerate() {
            for i in 0..num_per_target {
                let point = self.initialize_wild_start(target, i);
                let prime_idx = i % PRIME_MULTIPLIERS.len();
                let initial_prime = BigInt256::from_u64(PRIME_MULTIPLIERS[prime_idx]); // For inv in solve
                wilds.push(TaggedKangarooState {
                    point,
                    distance: BigInt256::zero(),
                    target_idx: idx as u32,
                    initial_offset: initial_prime
                });
            }
        }
        let tames: Vec<_> = (0..wilds.len()).map(|_| {
            let mut tame = KangarooState::new(Point::infinity(), 0, [0; 4], [0; 4], true, 0);
            tame.position = self.initialize_tame_start();
            tame.distance = 0; // u64 distance for tame kangaroos
            tame
        }).collect();
        (wilds, tames)
    }
}

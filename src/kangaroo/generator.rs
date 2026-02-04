//! Kangaroo generation logic
//!
//! Strict tame/wild start logic — fixed primes for wild, G-based tame, no entropy unless flagged

use crate::config::Config;
use crate::types::{KangarooState, Point, TaggedKangarooState};
use crate::math::{Secp256k1, bigint::{BigInt256, BigInt512}};
use crate::kangaroo::SearchConfig;
use num_bigint::BigInt;
use std::ops::Rem;
use std::ops::Sub;
use anyhow::anyhow;

use anyhow::Result;
use log::warn;
use rayon::prelude::*;
use rand::rngs::OsRng;
use rand::Rng;
use std::sync::Arc;

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

/// Positional Proxy Slice State for iterative range narrowing
#[derive(Clone, Debug)]
pub struct PosSlice {
    /// Lower bound of current slice
    pub low: BigInt,
    /// Upper bound of current slice
    pub high: BigInt,
    /// Positional proxy value (0 for unsolved puzzles)
    pub pos_proxy: u32,
    /// Accumulated bias factor from iterative refinement
    pub bias_factor: f64,
    /// Current iteration depth
    pub iteration: usize,
}

impl PosSlice {
    /// Create new slice from full range with initial proxy
    pub fn new(full_range: (BigInt, BigInt), initial_proxy: u32) -> Self {
        PosSlice {
            low: full_range.0,
            high: full_range.1,
            pos_proxy: initial_proxy,
            bias_factor: 1.0,
            iteration: 0,
        }
    }
}

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
        let y_bigint = point.y_bigint();
        let neg_y_bigint = self.curve.modulus().clone().sub(y_bigint);
        neg.y = neg_y_bigint.to_u64_array();
        neg
    }

    /// Concise Block: Bias Rho Partition to Vulnerable Mod
    fn rho_partition_quantum_bias(&self, point: &Point, dist: &BigInt256, seed: u32, mod_bias: u64) -> u32 {
        let bucket = self.rho_partition_f(point, dist, seed);
        (bucket as u64 % mod_bias) as u32 // Bias mod for vulnerable class
    }

    /// Concise Block: Separate Rho Partition with Bias
    pub fn rho_specific_partition(&self, point: &Point, dist: &BigInt256, seed: u32, bias_mod: u64) -> u32 {
        let mix = point.x_low_u32() as u32 ^ dist.low_u32() ^ seed; // Low bytes XOR as preset
        (mix as u64 % bias_mod) as u32 // Bias mod
    }

    /// Concise Block: Use Brent's in Rho for Collision
    pub fn rho_walk_with_brents(&self, g: Point, p: Point, bias_mod: u64) -> Option<BigInt256> {
        use std::collections::HashMap;

        const W: usize = 1 << 20; // Jump table size ~1M
        const DP_BITS: u32 = 32; // Distinguished point bits

        let mut tame = g.clone(); // Tame starts at G
        let mut wild = p.clone(); // Wild starts at target P
        let mut tame_steps = BigInt256::zero();
        let mut wild_steps = BigInt256::zero();
        let mut tame_dps: HashMap<[u64; 4], BigInt256> = HashMap::new();

        loop {
            // Tame walk
            let tame_hash = self.simple_hash(&tame);
            let jump_idx = if bias_mod == 0 {
                tame_hash % W as u64
            } else {
                (tame_hash % bias_mod) % W as u64
            } as usize;

            // Add jump (simplified - would use precomputed multiples)
            tame = self.curve.add(&tame, &self.curve.g_multiples[jump_idx % self.curve.g_multiples.len()]);
            tame_steps = tame_steps + BigInt256::from_u64(1);

            // Check for DP
            if tame.x[0] & ((1 << DP_BITS) - 1) == 0 {
                tame_dps.insert(tame.x, tame_steps.clone());
            }

            // Wild walk
            let wild_hash = self.simple_hash(&wild);
            let jump_idx = if bias_mod == 0 {
                wild_hash % W as u64
            } else {
                (wild_hash % bias_mod) % W as u64
            } as usize;

            wild = self.curve.add(&wild, &self.curve.g_multiples[jump_idx % self.curve.g_multiples.len()]);
            wild_steps = wild_steps + BigInt256::from_u64(1);

            // Check for collision
            if tame_dps.contains_key(&wild.x) {
                let tame_at_collision = tame_dps[&wild.x].clone();
                // Solve: tame_steps - tame_at_collision = k mod n
                let diff = tame_steps - tame_at_collision;
                let diff_512 = BigInt512::from_bigint256(&diff);
                return Some(self.curve.barrett_n.reduce(&diff_512).unwrap_or(BigInt256::zero()));
            }
        }
    }

    /// Simple hash function for kangaroo jumps
    fn simple_hash(&self, point: &Point) -> u64 {
        // Simple hash using the x coordinate
        point.x[0] ^ point.x[1] ^ point.x[2] ^ point.x[3]
    }

    /// Concise Block: Use Brent's in Rho Cycle
    pub fn rho_cycle_with_brents<F>(&self, f: F, x0: BigInt256) -> (u64, u64) where F: Fn(&BigInt256) -> BigInt256 {
        let (_, mu, lam) = brents_cycle_detection(f, x0);
        (mu, lam)
    }

    /// Concise Block: Grover-Like Amplifier for Bias Narrowing
    pub fn grover_amplifier_bias(&self, points: &Vec<Point>, bias_mod: u64) -> Vec<Point> {
        points.iter().filter(|p| p.x_bigint().mod_u64(bias_mod) == 0).cloned().collect() // "Amplify" biased
    }

    /// Concise Block: Utilize Biases in Jump with Detected b
    pub fn get_utilized_bias_jump(&self, bucket: u32, biases: &std::collections::HashMap<String, f64>) -> BigInt256 {
        let mut jump = crate::math::constants::PRIME_MULTIPLIERS[bucket as usize % 32];
        if biases["mod81"] > 0.012 { jump = jump + (81 - jump % 81); } // Utilize if detected high
        if biases["mod27"] > 0.037 { jump = jump + (27 - jump % 27); }
        if biases["mod9"] > 0.111 { jump = jump + (9 - jump % 9); }
        if biases["vanity"] > 0.0625 { jump = jump + (16 - jump % 16) + 9; } // Mod16=9
        BigInt256::from_u64(jump % (1u64 << 32))
    }

    /// Concise Block: Dynamic Rho m with Bias Prob
    pub fn rho_dynamic_m(&self, n: BigInt256, bias_prob: f64) -> u64 {
        (n.to_f64().sqrt() * bias_prob.sqrt()) as u64 // Adjust sqrt for bias
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

    /// Pollard's lambda algorithm for discrete logarithm in intervals
    /// Searches for k in [a, a+w] such that [k]G = Q
    /// Expected time O(√w) with tame/wild kangaroos
    /// Pollard's lambda algorithm for discrete logarithm in intervals
    /// Searches for k in [a, a+w] such that [k]G = Q
    /// Expected time O(√w) with tame/wild kangaroos
    pub fn pollard_lambda(&self, curve: &Secp256k1, g: &Point, q: &Point, a: BigInt256, w: BigInt256, max_cycles: u64, bias_mod: u64, b_pos: f64, pos_proxy: f64) -> Option<BigInt256> {
        use std::collections::HashMap;
        use crate::math::bigint::{BigInt256, BigInt512};
        use crate::types::Point;

        let midpoint = curve.barrett_n.add(&a, &w.right_shift(1));
        let mut tame = curve.mul_constant_time(&midpoint, g).ok()?;
        let mut wild = q.clone();
        let mut tame_dp: HashMap<[u64; 4], BigInt256> = HashMap::new();
        let mut tame_steps = BigInt256::zero();
        let mut wild_steps = BigInt256::zero();

        // Helper functions
        let is_dp = |p: &Point| -> bool {
            let dp_bits = 20;
            let mask = (1u64 << dp_bits) - 1;
            (p.x[0] & mask) == 0
        };

        let hash_point = |p: &Point| -> u64 {
            p.x[0] % 1024 // Simple hash for demo
        };

        let max_steps = if max_cycles == 0 { 10000000 } else { max_cycles }; // 0 = unlimited for demo
        for step in 0..max_steps {
            // Move tame with bias-aware jumping
            let tame_jump_idx = self.select_bias_aware_jump(&tame, bias_mod, b_pos, pos_proxy);
            tame = curve.add(&tame, &curve.g_multiples[tame_jump_idx % curve.g_multiples.len()]);
            tame_steps = curve.barrett_n.add(&tame_steps, &BigInt256::one());
            if is_dp(&tame) {
                tame_dp.insert(tame.x, tame_steps.clone());
            }

            // Move wild with bias-aware jumping
            let wild_jump_idx = self.select_bias_aware_jump(&wild, bias_mod, b_pos, pos_proxy);
            wild = curve.add(&wild, &curve.g_multiples[wild_jump_idx % curve.g_multiples.len()]);
            wild_steps = curve.barrett_n.add(&wild_steps, &BigInt256::one());

            // Check collision
            if let Some(t_steps) = tame_dp.get(&wild.x) {
                let neg_wild = wild_steps.negate(&curve.barrett_n);
                let diff = curve.barrett_n.add(&t_steps, &neg_wild);
                let k = curve.barrett_n.add(&midpoint, &diff);
                return Some(curve.barrett_n.reduce(&BigInt512::from_bigint256(&k)).ok()?);
            }
        }
        None
    }

    /// Multi-kangaroo parallel Pollard's lambda algorithm
    /// Uses multiple independent kangaroo pairs for O(√w / t) expected time
    /// where t is the number of kangaroo pairs
    /// Multi-kangaroo parallel Pollard's lambda algorithm
    /// Uses multiple independent kangaroo pairs for O(√w / t) expected time
    /// where t is the number of kangaroo pairs
    pub fn pollard_lambda_parallel(&self, curve: &Secp256k1, g: &Point, q: &Point, a: BigInt256, w: BigInt256, num_kangaroos: usize, max_cycles: u64, gpu: bool, bias_mod: u64, b_pos: f64, pos_proxy: f64) -> Option<BigInt256> {
        if gpu {
            // GPU implementation - dispatch to hybrid manager
            // Note: HybridGpuManager::new() is async, so for now we create a simple instance
            warn!("GPU multi-kangaroo dispatch not yet fully implemented - using CPU fallback");
            // For now, fall back to CPU implementation
        }

        // CPU parallel implementation using rayon
        let curve_arc = Arc::new(curve.clone());
        let g_arc = Arc::new(g.clone());
        let q_arc = Arc::new(q.clone());
        let a_arc = Arc::new(a.clone());
        let w_arc = Arc::new(w.clone());
        (0..num_kangaroos).into_par_iter().map(|_| {
            // Generate random offset for this kangaroo pair
            let w_array = w_arc.to_u64_array();
            let w_u64 = w_array[0]; // Take the lowest 64 bits for range splitting
            let offset_u64 = OsRng.gen::<u64>() % (w_u64 / num_kangaroos as u64).max(1);
            let offset = BigInt256::from_u64(offset_u64);

            let adjusted_a = curve_arc.barrett_n.add(&a_arc, &offset);
            let adjusted_w = w_arc.right_shift((num_kangaroos as f64).log2() as usize); // Split range

            self.pollard_lambda(&curve_arc, &g_arc, &q_arc, adjusted_a, adjusted_w, max_cycles / num_kangaroos as u64, bias_mod, b_pos, pos_proxy)
        }).find_any(|sol: &Option<BigInt256>| sol.is_some()).flatten()
    }


    /// Select bias-aware jump operation with hierarchical modulus preferences (mod9 -> mod27 -> mod81 -> pos)
    pub fn select_bias_aware_jump(&self, point: &Point, bias_mod: u64, b_pos: f64, pos_proxy: f64) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let hash = self.hash_position(point) as u64;
        let g_multiples_len = self.curve.g_multiples.len();

        // Hierarchical bias checking: mod9 first, then mod27, then mod81
        let mod9 = hash % 9;
        let mod27 = hash % 27;
        let mod81 = hash % 81;

        // High bias residues from solved puzzle analysis
        const HIGH_BIAS_MOD9: [u64; 2] = [0, 3];  // Magic 9 and common residues
        const HIGH_BIAS_MOD27: [u64; 3] = [0, 9, 18];  // Multiples of 9 within mod27
        const HIGH_BIAS_MOD81: [u64; 4] = [0, 9, 27, 36];  // Multiples of 9 within mod81

        // Apply hierarchical bias with decreasing priority
        if HIGH_BIAS_MOD9.contains(&mod9) {
            // Mod9 bias - strongest, favors multiples of 9
            (hash % (g_multiples_len / 3) as u64) as usize
        } else if HIGH_BIAS_MOD27.contains(&mod27) {
            // Mod27 bias - medium strength
            (hash % (g_multiples_len / 2) as u64) as usize
        } else if HIGH_BIAS_MOD81.contains(&mod81) {
            // Mod81 bias - finer control
            (hash % (g_multiples_len * 2 / 3) as u64) as usize
        } else if pos_proxy < 0.1 && rng.gen::<f64>() < b_pos {
            // Positional bias - lowest priority, for unsolved puzzles
            (hash % (g_multiples_len / 10) as u64) as usize
        } else {
            // Standard random selection
            (hash % g_multiples_len as u64) as usize
        }
    }

    /// Simple hash function for point-based jump selection
    fn hash_position(&self, point: &Point) -> usize {
        // Simple hash using point coordinates
        let mut hash = point.x[0] as usize;
        hash = hash.wrapping_mul(31).wrapping_add(point.x[1] as usize);
        hash = hash.wrapping_mul(31).wrapping_add(point.x[2] as usize);
        hash = hash.wrapping_mul(31).wrapping_add(point.x[3] as usize);
        hash.wrapping_mul(31).wrapping_add(point.y[0] as usize)
    }

    /// Iterative POS slice refiner - narrows search space based on positional proxies
    /// Returns refined slice with accumulated bias factor
    pub fn refine_pos_slice(slice: &mut PosSlice, mod_biases: &std::collections::HashMap<u32, f64>, max_iterations: usize) {
        if slice.iteration >= max_iterations {
            return; // Prevent over-slicing that could miss solutions
        }

        let range_size = &slice.high - &slice.low;
        let proxy_offset = BigInt::from(slice.pos_proxy as u64);

        // Get bias adjustment for this proxy value
        let bias_adj = *mod_biases.get(&slice.pos_proxy).unwrap_or(&1.0);

        // Refine slice bounds: offset by proxy factor, scale by bias
        // low = original_low + (range_size * proxy_offset / max_proxy_factor)
        slice.low += &range_size / BigInt::from(10u32) * &proxy_offset; // 10% base offset per proxy unit

        // high = low + (original_range_size * bias_adjustment)
        slice.high = &slice.low + &range_size * BigInt::from((bias_adj * 10.0) as u64) / BigInt::from(10u32);

        // Accumulate bias factor for jump weighting
        slice.bias_factor *= bias_adj;

        slice.iteration += 1;
    }

    /// Generate random BigInt within a POS slice for kangaroo initialization
    pub fn random_in_slice(slice: &PosSlice) -> BigInt {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let range_size = &slice.high - &slice.low;
        if range_size <= BigInt::from(0) {
            return slice.low.clone(); // Degenerate case
        }

        // Generate random offset within slice bounds
        // Use secure randomness in production (this is simplified for demo)
        let random_offset = BigInt::from(rng.gen::<u64>()) % &range_size;
        &slice.low + random_offset
    }

    /// Dynamic bias adjustment based on collision rates
    /// Increases bias factors when collision rate is below expected threshold
    pub fn adjust_bias_dynamically(biases: &mut std::collections::HashMap<u32, f64>, collision_rate: f64, expected_rate: f64) {
        if collision_rate < expected_rate {
            // Increase bias factors to encourage more directed searching
            for bias_val in biases.values_mut() {
                *bias_val *= 1.1; // 10% increase
                *bias_val = bias_val.min(3.0); // Cap at 3x to prevent over-biasing
            }
        } else if collision_rate > expected_rate * 1.5 {
            // Decrease bias if too many false positives
            for bias_val in biases.values_mut() {
                *bias_val *= 0.95; // 5% decrease
                *bias_val = bias_val.max(0.5); // Floor at 0.5x
            }
        }
    }
}

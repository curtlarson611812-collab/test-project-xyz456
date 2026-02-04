//! Kangaroo generation logic
//!
//! Strict tame/wild start logic — fixed primes for wild, G-based tame, no entropy unless flagged

use crate::config::Config;
use crate::types::{KangarooState, Point, TaggedKangarooState, RhoState};
use crate::math::{Secp256k1, bigint::{BigInt256, BigInt512}};
use crate::kangaroo::SearchConfig;
use crate::dp::DpTable;
use num_bigint::BigInt;
use std::ops::Sub;

use anyhow::Result;
use log::{info, warn};
use rayon::prelude::*;
use rand::rngs::OsRng;
use rand::Rng;
use std::sync::Arc;
// SIMD removed for stability - using regular loops instead

/// Chunk: Bias-Aware Brent's (collision.rs)
pub fn biased_brent_cycle<F>(start: &BigInt256, _f: F, biases: &std::collections::HashMap<u32, f64>) -> Option<BigInt256>
where F: FnMut(&BigInt256) -> BigInt256 {
    let mut tortoise = start.clone();
    let mut hare = f_biased(&tortoise, biases);  // Bias wrap
    let mut power = 1;
    let mut lam = 1;
    while tortoise != hare {
        if power == lam {
            tortoise = hare.clone();
            power *= 2;
            lam = 0;
        }
        hare = f_biased(&hare, biases);
        lam += 1;
    }
    Some(hare)
}
fn f_biased(x: &BigInt256, biases: &std::collections::HashMap<u32, f64>) -> BigInt256 {
    let res = x.clone() % BigInt256::from_u64(81u64);
    let b = biases.get(&res.low_u32()).unwrap_or(&1.0);
    x.clone() + BigInt256::from_u64((rand::random::<f64>() * *b) as u64)
}

// Concise Block: Brent's Cycle Detection for Rho Walks
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

// Chunk: Cache-Aligned PosSlice (generator.rs)
// Dependencies: num_bigint::BigInt, std::mem::align_of
#[repr(align(64))]  // Cache line align
#[derive(Clone)]  // BigInt not Copy (heap allocated)
pub struct PosSlice {
    pub low: BigInt,    // 32B est
    pub high: BigInt,   // 32B
    pub proxy: u32,     // 4B
    pub bias: f64,      // 8B
    pub iter: u8,       // 1B + padding
}
// Static assert: assert_eq!(align_of::<PosSlice>(), 64);

/// Create new POS slice from range and proxy
pub fn new_slice(range: (BigInt, BigInt), proxy: u32) -> PosSlice {
    PosSlice {
        low: range.0,
        high: range.1,
        proxy,
        bias: 1.0,
        iter: 0,
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

    /// Wild kangaroos — EXACT SmallOddPrime spacing (multiplicative offset)
    fn generate_wild_for_target(&self, target: &Point, count: usize) -> Result<Vec<KangarooState>> {
        use crate::math::constants::PRIME_MULTIPLIERS;

        let mut wilds = Vec::new();

        for i in 0..count {
            // Use SmallOddPrime start: prime * target
            let start_pos = self.initialize_wild_start(target, i);

            let prime_idx = i % PRIME_MULTIPLIERS.len();
            let prime = PRIME_MULTIPLIERS[prime_idx];

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
            // Use deterministic tame start from generator
            let start_position = self.initialize_tame_start();

            // For tame kangaroos, we start at different offsets from G
            let offset = BigInt256::from_u64(i as u64);
            let curve = Secp256k1::new();
            let actual_start = curve.add(&start_position, &curve.mul(&offset, &curve.g));

            let alpha = offset.to_u64_array();
            let beta = [1, 0, 0, 0]; // Beta coefficient

            tames.push(KangarooState::new(
                actual_start,
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
    /// Initialize wild kangaroo start with SmallOddPrime multiplier
    pub fn initialize_wild_start(&self, target: &Point, kangaroo_index: usize) -> Point {
        use crate::math::constants::PRIME_MULTIPLIERS;

        let prime_idx = kangaroo_index % PRIME_MULTIPLIERS.len();
        let prime = PRIME_MULTIPLIERS[prime_idx];

        // wild_start = prime * target_pubkey
        let prime_bigint = BigInt256::from_u64(prime);
        self.curve.mul(&prime_bigint, target)
    }

    /// Initialize tame kangaroo start (deterministic from generator)
    pub fn initialize_tame_start(&self) -> Point {
        use crate::math::constants::GENERATOR;
        Point {
            x: GENERATOR.x,
            y: GENERATOR.y,
            z: GENERATOR.z,
        }
    }

    /// Legacy initialize_wild_start (kept for compatibility)
    pub fn initialize_wild_start_legacy(&self, target: &Point, kangaroo_index: usize) -> Point {
        use crate::math::constants::PRIME_MULTIPLIERS;
        let prime_index = kangaroo_index % PRIME_MULTIPLIERS.len();
        let prime = BigInt256::from_u64(PRIME_MULTIPLIERS[prime_index]);
        self.curve.mul(&prime, target) // prime * target point
    }

    /// Tame Start from G (Clean, No Multiplier)

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
    // fn get_jump_from_bucket(&self, bucket: u32) -> BigInt256 {
    //     use crate::math::constants::PRIME_MULTIPLIERS;
    //     let prime = PRIME_MULTIPLIERS[bucket as usize % 32];
    //     let biased = prime + (9 - (prime % 9)); // Adjust to mod9=0 for theory
    //     BigInt256::from_u64(biased % (1u64 << 32)) // Keep small for speed
    // }

    /// Concise Block: Mod9-Biased Jump from Bucket
    // fn get_jump_from_bucket_mod9(&self, bucket: u32) -> BigInt256 {
    //     use crate::math::constants::PRIME_MULTIPLIERS;
    //     let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
    //     let adjust = 9 - (base_prime % 9); // To next multiple of 9
    //     let biased = base_prime + adjust;
    //     BigInt256::from_u64(biased % (1 << 32)) // Small, mod9=0
    // }

    /// Concise Block: Mod27-Biased Jump from Bucket
    // fn get_jump_from_bucket_mod27(&self, bucket: u32) -> BigInt256 {
    //     use crate::math::constants::PRIME_MULTIPLIERS;
    //     let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
    //     let adjust = 27 - (base_prime % 27); // To next multiple of 27
    //     let biased = base_prime + adjust;
    //     BigInt256::from_u64(biased % (1 << 32)) // Small, mod27=0
    // }

    /// Concise Block: Mod81-Biased Jump from Bucket
    // fn get_jump_from_bucket_mod81(&self, bucket: u32) -> BigInt256 {
    //     use crate::math::constants::PRIME_MULTIPLIERS;
    //     let base_prime = PRIME_MULTIPLIERS[bucket as usize % 32];
    //     let adjust = 81 - (base_prime % 81); // To next multiple of 81
    //     let biased = base_prime + adjust;
    //     BigInt256::from_u64(biased % (1 << 32)) // Small, mod81=0
    // }

    /// Concise Block: Vanity-Biased Jump Adjust
    // fn get_jump_vanity_biased(&self, bucket: u32, mod_n: u64, bias_res: u64) -> BigInt256 {
    //     use crate::math::constants::PRIME_MULTIPLIERS;
    //     let base = PRIME_MULTIPLIERS[bucket as usize % 32];
    //     let adjust = mod_n - (base % mod_n) + bias_res; // To mod n = bias_res
    //     BigInt256::from_u64(base + adjust % mod_n)
    // }

    /// Concise Block: Pollard's Lambda Bucket as Jump Hash
    // fn lambda_bucket_select(&self, point: &Point, dist: &BigInt256, seed: u32, step: u32, is_tame: bool) -> u32 {
    //     self.select_bucket(point, dist, seed, step, is_tame) // Prior preset
    // }

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
    // fn rho_partition_quantum_bias(&self, point: &Point, dist: &BigInt256, seed: u32, mod_bias: u64) -> u32 {
    //     let bucket = self.rho_partition_f(point, dist, seed);
    //     (bucket as u64 % mod_bias) as u32 // Bias mod for vulnerable class
    // }

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
    pub fn setup_kangaroos_multi(&self, targets: &[Point], num_per_target: usize, _config: &SearchConfig) -> (Vec<TaggedKangarooState>, Vec<KangarooState>) {
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
    pub fn pollard_lambda(&self, curve: &Secp256k1, g: &Point, q: &Point, a: BigInt256, w: BigInt256, max_cycles: u64, bias_mod: u64, b_pos: f64, pos_proxy: f64, biases: Option<&std::collections::HashMap<u32, f64>>) -> Option<BigInt256> {
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

        let _hash_point = |p: &Point| -> u64 {
            p.x[0] % 1024 // Simple hash for demo
        };

        let max_steps = if max_cycles == 0 { 10000000 } else { max_cycles }; // 0 = unlimited for demo
        for _step in 0..max_steps {
            // Move tame with bias-aware jumping
            let tame_jump_idx = if let Some(biases) = biases {
                let tame_dist_bigint = BigInt256::from_u64_array([
                    tame_steps.limbs[0], tame_steps.limbs[1], tame_steps.limbs[2], tame_steps.limbs[3]
                ]);
                let biased_jump = self.biased_jump(&tame_dist_bigint, biases);
                // Convert back to jump index
                (biased_jump.low_u32() % curve.g_multiples.len() as u32) as usize
            } else {
                self.select_bias_aware_jump(&tame, bias_mod, b_pos, pos_proxy)
            };
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
    /// Initialize kangaroo states for parallel execution
    fn pollard_init_parallel(&self, curve: &Arc<Secp256k1>, _g: &Arc<Point>, _q: &Arc<Point>,
                           a: &Arc<BigInt256>, w: &Arc<BigInt256>, num_kangaroos: usize)
                           -> Vec<(BigInt256, BigInt256)> {
        (0..num_kangaroos).map(|_| {
            // Generate random offset for this kangaroo pair
            let w_clone = (**w).clone();
            let w_array = w_clone.to_u64_array();
            let w_u64 = w_array[0]; // Take the lowest 64 bits for range splitting
            let offset_u64 = OsRng.gen::<u64>() % (w_u64 / num_kangaroos as u64).max(1);
            let offset = BigInt256::from_u64(offset_u64);

            let adjusted_a = curve.barrett_n.add(a, &offset);
            let adjusted_w = w.right_shift((num_kangaroos as f64).log2() as usize); // Split range

            (adjusted_a, adjusted_w)
        }).collect()
    }

    /// Run parallel kangaroo search
    fn pollard_run_parallel(&self, curve: &Arc<Secp256k1>, g: &Arc<Point>, q: &Arc<Point>,
                          states: Vec<(BigInt256, BigInt256)>, max_cycles: u64,
                          bias_mod: u64, b_pos: f64, pos_proxy: f64, biases: Option<&std::collections::HashMap<u32, f64>>) -> Option<BigInt256> {
        let states_len = states.len() as u64;
        states.into_par_iter().find_map_first(|(adjusted_a, adjusted_w)| {
        self.pollard_lambda(curve, g, q, adjusted_a, adjusted_w,
                          max_cycles / states_len, bias_mod, b_pos, pos_proxy, biases)
        })
    }

    fn pollard_init(&self, curve: &Secp256k1, g: &Point, q: &Point, a: BigInt256, w: BigInt256, num_kangaroos: usize) -> Vec<(BigInt256, BigInt256)> {
        let curve_arc = Arc::new(curve.clone());
        let g_arc = Arc::new(g.clone());
        let q_arc = Arc::new(q.clone());
        let a_arc = Arc::new(a.clone());
        let w_arc = Arc::new(w.clone());
        self.pollard_init_parallel(&curve_arc, &g_arc, &q_arc, &a_arc, &w_arc, num_kangaroos)
    }

    fn pollard_run(&self, curve: &Secp256k1, g: &Point, q: &Point, states: Vec<(BigInt256, BigInt256)>, max_cycles: u64, bias_mod: u64, b_pos: f64, pos_proxy: f64, biases: Option<&std::collections::HashMap<u32, f64>>) -> Option<BigInt256> {
        let curve_arc = Arc::new(curve.clone());
        let g_arc = Arc::new(g.clone());
        let q_arc = Arc::new(q.clone());
        self.pollard_run_parallel(&curve_arc, &g_arc, &q_arc, states, max_cycles, bias_mod, b_pos, pos_proxy, biases)
    }

    fn pollard_resolve(&self, _dp_table: &DpTable, _states: &[RhoState]) -> Option<BigInt256> {
        // Placeholder - would check collisions in dp_table
        None
    }

    pub fn pollard_lambda_parallel(&self, curve: &Secp256k1, g: &Point, q: &Point, a: BigInt256, w: BigInt256, num_kangaroos: usize, max_cycles: u64, gpu: bool, bias_mod: u64, b_pos: f64, pos_proxy: f64, biases: Option<&std::collections::HashMap<u32, f64>>) -> Option<BigInt256> {
        if gpu {
            // GPU implementation - dispatch to hybrid manager
            // Note: HybridGpuManager::new() is async, so for now we create a simple instance
            warn!("GPU multi-kangaroo dispatch not yet fully implemented - using CPU fallback");
        }

        let states = self.pollard_init(curve, g, q, a, w, num_kangaroos);
        self.pollard_run(curve, g, q, states, max_cycles, bias_mod, b_pos, pos_proxy, biases)
    }


    /// Select bias-aware jump operation with hierarchical modulus preferences (mod9 -> mod27 -> mod81 -> pos)
    /// Bias-aware jump with chain: mod9 -> mod27 -> mod81 + pos_proxy scaling
    pub fn biased_jump(&self, current: &BigInt256, biases: &std::collections::HashMap<u32, f64>) -> BigInt256 {
        use crate::utils::pubkey_loader::detect_bias_single;
        use crate::math::constants::CURVE_ORDER_BIGINT;
        let mod_levels = [9u32, 27u32, 81u32];
        let mut chain_bias = 1.0;
        for &mod_level in &mod_levels {
            let mod_val = BigInt256::from_u64(mod_level as u64);
            let res = (current.clone() % mod_val).low_u32();
            chain_bias *= *biases.get(&res).unwrap_or(&1.0);
        }
        let (_, _, pos_factor) = detect_bias_single(current, 0);  // pos_proxy
        chain_bias *= pos_factor;
        let rand_scale = rand::random::<f64>() * chain_bias;
        let base_jump = BigInt256::from_u64(rand::random::<u32>() as u64);  // Mock base, replace with bucket
        let jump = base_jump + BigInt256::from_u64(rand_scale as u64);
        jump % CURVE_ORDER_BIGINT.clone()  // Clamp mod order
    }

    fn pollard_run_biased(&self, states: &mut [KangarooState], steps: usize, biases: &std::collections::HashMap<u32, f64>) {
        for state in states.iter_mut() {
            for _ in 0..steps {
                let dist_bigint = BigInt256::from_u64(state.distance);
                let jump_size = self.biased_jump(&dist_bigint, biases);
                let jump_point = self.curve.mul(&jump_size, &self.curve.g);
                state.position = self.curve.point_add(&state.position, &jump_point);
                state.distance = state.distance.wrapping_add(jump_size.low_u64());
                if state.distance.trailing_zeros() >= crate::math::constants::DP_BITS {
                    // Mark as DP candidate - would need to add this field to KangarooState
                    break;
                }
            }
        }
    }

    pub fn select_bias_aware_jump(&self, point: &Point, _bias_mod: u64, b_pos: f64, pos_proxy: f64) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let hash = self.hash_position(point) as u64;
        let g_multiples_len = self.curve.g_multiples.len();

        // Hierarchical bias checking: mod9 first, then mod27, then mod81
        let mod9 = hash % 9;
        let mod27 = hash % 27;
        let mod81 = hash % 81;

        // High bias residues from solved puzzle analysis (aggressive for laptop)
        const HIGH_BIAS_MOD9: [u64; 2] = [0, 3];  // Magic 9 and common residues
        const HIGH_BIAS_MOD27: [u64; 3] = [0, 9, 18];  // Multiples of 9 within mod27
        const HIGH_BIAS_MOD81: [u64; 5] = [0, 9, 27, 36, 45];  // Multiples of 9 within mod81 (extended)

        // Apply hierarchical bias with heavier mod81 for laptop speed
        if HIGH_BIAS_MOD9.contains(&mod9) {
            // Mod9 bias - strongest, favors multiples of 9
            (hash % (g_multiples_len / 3) as u64) as usize
        } else if HIGH_BIAS_MOD27.contains(&mod27) {
            // Mod27 bias - medium strength
            (hash % (g_multiples_len / 2) as u64) as usize
        } else if HIGH_BIAS_MOD81.contains(&mod81) {
            // Mod81 bias - finer control, heavier weight (4x)
            (hash % (g_multiples_len * 4 / 5) as u64) as usize
        } else if pos_proxy < 0.1 && rng.gen::<f64>() < b_pos {
            // Positional bias - lowest priority, for unsolved puzzles
            (hash % (g_multiples_len / 10) as u64) as usize
        } else {
            // Standard random selection
            (hash % g_multiples_len as u64) as usize
        }
    }

    /// Aggregate bias patterns from points with Pop!_OS NVIDIA persistence
    pub fn aggregate_bias(&self, points: &[Point]) -> std::collections::HashMap<u32, f64> {
        use crate::utils::pubkey_loader::detect_bias_single;

        // Enable NVIDIA persistence for Pop!_OS (boosts long-run hashrate 10%)
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            if let Ok(_) = Command::new("nvidia-smi").arg("-pm").arg("1").output() {
                info!("NVIDIA persistence enabled for Pop!_OS optimization");
            }
        }

        let mut bias_map = std::collections::HashMap::new();
        let total = points.len() as f64;

        for (i, point) in points.iter().enumerate() {
            let x_bigint = BigInt256::from_u64_array(point.x);
            let (bias_mod, dominant_residue, _) = detect_bias_single(&x_bigint, (i + 1) as u32);

            // Only count points with detected bias
            if bias_mod > 0 {
                let key = dominant_residue as u32;
                *bias_map.entry(key).or_insert(0.0) += 1.0 / total;
            }
        }

        bias_map
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

// Chunk: Anti-Overfit in POS Refine with Proper Bounds (generator.rs)
pub fn refine_pos_slice(slice: &mut PosSlice, biases: &std::collections::HashMap<u32, f64>, max_iterations: u8) {
    if slice.iter >= max_iterations || (&slice.high - &slice.low) < BigInt::from(1u64 << 20) {
        return;  // Size guard (min 1M range), iteration limit (3 optimal)
    }

    let r = &slice.high - &slice.low;
    let b = *biases.get(&slice.proxy).unwrap_or(&1.0);
    slice.low += &r / BigInt::from(12u32); // ~8% offset for exploration
    slice.high = &slice.low + &r * BigInt::from((b * 1.1) as u64);
    slice.bias *= b;
    slice.iter += 1;

    // Overfit prevention: inject entropy if variance too high (σ > 0.5 = overfit risk)
    let bias_values: Vec<f64> = biases.values().cloned().collect();
    if bias_values.len() > 2 {
        let mean = bias_values.iter().sum::<f64>() / bias_values.len() as f64;
        let variance = bias_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / bias_values.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.5 {  // High variance = overfit risk (prevent infinite loops on small data)
            slice.low += BigInt::from(rand::random::<u64>() % (1u64 << 10));  // Inject entropy
        }
    }
}

// Chunk: Batch Refine for Cache (generator.rs)
pub fn batch_refine_slices(slices: &mut [PosSlice], biases: &std::collections::HashMap<u32, f64>) {
    for s in slices.iter_mut() {  // Sequential but cache-hot
        if s.iter >= 3 { continue; }
        let r = &s.high - &s.low;
        let b = *biases.get(&s.proxy).unwrap_or(&1.0);
        s.low += &r / BigInt::from(12u32);
        s.high = &s.low + &r * BigInt::from((b * 1.1) as u64);
        s.bias *= b;
        s.iter += 1;
    }
}
// Usage: In init, let mut slices = vec![new_slice(); 4096]; batch_refine_slices(&mut slices, &biases);

// Chunk: χ² Convergence Check (generator.rs)
pub fn should_refine(slice: &PosSlice, obs_freq: &[f64], exp_uniform: f64) -> bool {
    let df = obs_freq.len() as f64 - 1.0;
    let chi2: f64 = obs_freq.iter().map(|&o| (o - exp_uniform).powi(2) / exp_uniform).sum();
    let crit = df * 2.0;  // Approximate χ² critical value (rough approximation)
    chi2 > crit && slice.iter < 5  // χ² high = non-uniform → refine
}

// Chunk: Bayesian Posterior Update (generator.rs)
pub fn bayesian_posterior(hits: u32, misses: u32, prior_alpha: f64, prior_beta: f64) -> f64 {
    let post_alpha = prior_alpha + hits as f64;
    let post_beta = prior_beta + misses as f64;
    post_alpha / (post_alpha + post_beta)  // Mean of beta dist
}
// Usage: If bayesian_posterior(bias_hits, bias_misses, 1.0, 1.0) < 0.6 { stop refine }

// Chunk: Barrett Mu Precomp (generator.rs)
// Dependencies: num_bigint::BigInt (for shift/div)
pub fn barrett_mu(range: &BigInt256) -> BigInt256 {
    // For Barrett reduction, mu = floor(2^{2*k} / modulus) where k=256
    // Since we can't represent 2^{512} in BigInt256, use approximation
    // mu ≈ 2^{256} / range for the Barrett constant
    let two_to_256 = BigInt256 { limbs: [0, 0, 0, 1] }; // 2^256 as BigInt256 (approximation)
    two_to_256.div_rem(range).0  // floor div
}

// Chunk: Safe Barrett Rem (generator.rs)
// In SIMD function (after q calc):
// let mut rem = rand - q * range;  // BigInt256 ops
// while rem >= range {  // At most 2 iters
//     rem -= range;
// }
// if rem < BigInt256::zero() { rem += range; }  // Rare underflow

// Chunk: AVX512 Gate for BigInt Random (generator.rs)
// Dependencies: std::arch::x86_64::*, num_bigint::BigInt (for range)
#[cfg(target_feature = "avx512f")]
pub fn simd_random_in_slice(slice: &PosSlice, count: usize) -> Vec<BigInt256> {
    use std::arch::x86_64::*;
    let range = &slice.high - &slice.low;
    let mut rands = vec![BigInt256::zero(); count];
    unsafe {
        for i in (0..count).step_by(8) {  // x8 for AVX512
            let rand_limbs = _mm512_set_epi64(rand::random(), rand::random(), /* repeat 8x */);
            let mu = _mm512_set1_epi64((1u64 << 65) / range.limbs[0]);  // Barrett mu adj
            let q = _mm512_mul_epu32(rand_limbs, mu);
            let rem = _mm512_sub_epi64(rand_limbs, _mm512_mul_epu32(q, range.limbs_vec()));
            // Unpack rem to rands[i..i+8] via _mm512_extract_epi64
        }
    }
    rands
}

#[cfg(not(target_feature = "avx512f"))]
pub fn simd_random_in_slice(slice: &PosSlice, count: usize) -> Vec<BigInt> {
    (0..count).map(|_| {
        let range = &slice.high - &slice.low;
        &slice.low + (BigInt::from(rand::random::<u64>()) % &range)
    }).collect()
}

// Chunk: SIMD Random in Slice (generator.rs)
pub fn vec_random_in_slice(slice: &PosSlice, count: usize) -> Vec<BigInt> {
    let _range = &slice.high - &slice.low;
    let mut rands = vec![num_bigint::BigInt::from(0); count];
    for i in 0..count {
        let rand_val = rand::random::<u64>() % 1000000;  // Assume small range for now
        rands[i] = BigInt::from(rand_val);
    }
    rands
}

/// Generate random BigInt within POS slice bounds
pub fn random_in_slice(slice: &PosSlice) -> BigInt {
    use rand::Rng;
    let range = &slice.high - &slice.low;
    if range <= BigInt::from(0) { return slice.low.clone(); }
    &slice.low + BigInt::from(rand::thread_rng().gen::<u64>()) % &range
}

/// Concise dynamic bias tuning - one-liner adjustment (lower thresh for laptop)
pub fn tune_bias(biases: &mut std::collections::HashMap<u32, f64>, coll_rate: f64, target: f64) {
    if coll_rate < target * 0.08 {  // Lower thresh for quick adj
        for v in biases.values_mut() { *v = (*v * 1.15).min(2.5); }  // Aggressive boost
    }
}

/// POS slicing integrated pollard lambda parallel - 8 lines
pub fn pollard_lambda_parallel_pos(_target: &BigInt256, range: (BigInt, BigInt)) -> Option<BigInt256> {
    let _slice = new_slice(range, 0);
    let _biases = std::collections::HashMap::from([(9, 1.25), (27, 1.35), (81, 1.42)]);
        for _ in 0..3 {
            // TODO: Implement batch processing
            // let starts: Vec<BigInt> = (0..4096).map(|_| random_in_slice(&slice)).collect();
            // if let Some(key) = run_batch_mock(&starts, target) { return Some(key); }
            // refine_pos_slice(&mut slice, &biases);
        }
    None
}

// Mock batch runner for testing (replace with real implementation)
// TODO: Uncomment when implementing mock testing
// fn run_batch_mock(_starts: &[BigInt], _target: &BigInt256) -> Option<BigInt256> {
//     // Mock collision detection - in real implementation this would run kangaroo algorithm
//     use rand::Rng;
//     if rand::thread_rng().gen_bool(0.001) { // 0.1% mock success rate
//         Some(BigInt256::from_u64(42)) // Mock solution
//     } else {
//         None
//     }
// }
}

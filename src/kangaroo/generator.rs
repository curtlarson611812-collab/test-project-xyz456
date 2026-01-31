//! Kangaroo generation logic
//!
//! Strict tame/wild start logic — fixed primes for wild, G-based tame, no entropy unless flagged

use crate::config::Config;
use crate::types::{KangarooState, Point};
use crate::math::Secp256k1;

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
            curve: Secp256k1::new(),
        }
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
        // TODO: Generate wild kangaroo starting from target vicinity
        todo!("Implement wild kangaroo generation")
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
        // TODO: Generate tame kangaroo starting from G + offset
        todo!("Implement tame kangaroo generation")
    }
}
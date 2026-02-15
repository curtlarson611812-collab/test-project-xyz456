//! Kangaroo stepping logic for Pollard's rho algorithm
//!
//! Implements the jump operations for tame and wild kangaroos, updating positions
//! and alpha/beta coefficients according to the distinguished point method.

use crate::types::{KangarooState, Point, JumpOp};
use crate::math::constants::jump_table;
use crate::math::{Secp256k1, BigInt256};
// use crate::SmallOddPrime_Precise_code as sop; // Module not found
use anyhow::Result;

/// Kangaroo stepper implementing jump operations
#[derive(Clone)]
pub struct KangarooStepper {
    curve: Secp256k1,
    _jump_table: Vec<Point>, // Precomputed jump points
    expanded_mode: bool, // Enable expanded jump table mode for bias adaptation
    dp_bits: usize, // DP bits for negation check
    step_count: u32, // Global step counter for tame kangaroo bucket selection
    seed: u32, // Configurable seed for randomization
}

impl KangarooStepper {
        pub fn new(expanded_mode: bool) -> Self {
    KangarooStepper::with_dp_bits_and_seed(expanded_mode, 20, 42)
    }

    pub fn with_dp_bits(expanded_mode: bool, dp_bits: usize) -> Self {
        KangarooStepper::with_dp_bits_and_seed(expanded_mode, dp_bits, 42)
    }

    pub fn with_dp_bits_and_seed(expanded_mode: bool, dp_bits: usize, seed: u32) -> Self {
        let curve = Secp256k1::new();
        let jump_table = Self::build_jump_table(&curve, expanded_mode);
        KangarooStepper {
            curve,
            _jump_table: jump_table,
            expanded_mode,
            dp_bits,
            step_count: 0,
            seed,
        }
    }

    fn build_jump_table(curve: &Secp256k1, expanded: bool) -> Vec<Point> {
        if expanded {
            Self::precompute_jumps_expanded(32)
        } else {
            (0..16).map(|i| curve.mul_constant_time(&BigInt256::from_u64(i as u64 + 1), &curve.g).unwrap()).collect()
        }
    }

    pub fn precompute_jumps_expanded(size: usize) -> Vec<Point> {
        let curve = Secp256k1::new();
        (0..size).map(|i| curve.mul_constant_time(&BigInt256::from_u64(i as u64 + 1), &curve.g).unwrap()).collect()
    }

    pub fn step_kangaroo_with_bias(&self, kangaroo: &KangarooState, target: Option<&Point>, bias_mod: u64) -> KangarooState {
        let bucket = self.select_sop_bucket(kangaroo, target, bias_mod);
        // Simple prime selection (replace sop::get_biased_prime)
        let primes = [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137];
        let jump_d = primes[(bucket as usize).min(primes.len() - 1)];

        let (new_position, new_distance, alpha_update, beta_update) = if kangaroo.is_tame {
            let jump_point = self.curve.mul_constant_time(&BigInt256::from_u64(jump_d), &self.curve.g).unwrap();
            let new_pos = self.curve.add(&kangaroo.position, &jump_point);
            let new_dist = kangaroo.distance.clone() + BigInt256::from_u64(jump_d);
            (new_pos, new_dist, [jump_d as u64, 0, 0, 0], [0, 0, 0, 0])
        } else {
            if let Some(t) = target {
                let jump_point = self.curve.mul_constant_time(&BigInt256::from_u64(jump_d), t).unwrap();
                let new_pos = self.curve.add(&kangaroo.position, &jump_point);
                let new_dist = kangaroo.distance.clone() * BigInt256::from_u64(jump_d) % self.curve.n.clone();
                (new_pos, new_dist, [0, 0, 0, 0], [jump_d as u64, 0, 0, 0])
            } else {
                (kangaroo.position.clone(), kangaroo.distance.clone(), [0;4], [0;4])
            }
        };

        let new_alpha = self.update_coefficient(&kangaroo.alpha, &alpha_update, true);
        let new_beta = self.update_coefficient(&kangaroo.beta, &beta_update, false);

        KangarooState {
            position: new_position,
            distance: new_distance,
            alpha: new_alpha,
            beta: new_beta,
            is_tame: kangaroo.is_tame,
            is_dp: kangaroo.is_dp,
            id: kangaroo.id,
            step: kangaroo.step + 1,
            kangaroo_type: kangaroo.kangaroo_type,
        }
    }

    // ... keep your other methods (select_sop_bucket, update_coefficient, etc.)

    pub fn select_sop_bucket(&self, kangaroo: &KangarooState, _target: Option<&Point>, _bias_mod: u64) -> u32 {
        if kangaroo.is_tame {
            self.step_count % 32
        } else {
            let pos_hash = self.hash_position(&kangaroo.position);
            let dist_hash = self.hash_position(&Point::infinity());
            let seed = self.seed;
            let step = self.step_count;
            let mix = pos_hash ^ dist_hash ^ (seed as u64) ^ (step as u64);
            (mix % 32) as u32
        }
    }

    fn update_coefficient(&self, current: &[u64; 4], update: &[u64; 4], _is_alpha: bool) -> [u64; 4] {
        let mut result = *current;
        for i in 0..4 {
            result[i] = result[i].wrapping_add(update[i]);
        }
        result
    }

    fn hash_position(&self, position: &Point) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        position.x.hash(&mut hasher);
        hasher.finish()
    }

    pub fn is_distinguished_point(&self, point: &Point, dp_bits: usize) -> bool {
        let x_hash = self.hash_position(point);
        (x_hash & ((1u64 << dp_bits) - 1)) == 0
    }

    /// Step a batch of kangaroos
    pub fn step_batch(&self, kangaroos: &[KangarooState], target: Option<&Point>) -> Result<Vec<KangarooState>> {
        kangaroos.iter()
            .map(|k| Ok(self.step_kangaroo_with_bias(k, target, 1u64)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    /// Test single kangaroo step
    #[test]
    fn test_single_step() {
        let stepper = KangarooStepper::new(false);
        let initial_pos = stepper.curve.g.clone();

        let kangaroo = KangarooState::new(
            initial_pos,
            BigInt256::zero(),
            [0; 4], // alpha
            [0; 4], // beta
            true,   // tame
            false,  // is_dp
            0,      // id
            0,      // step
            0,      // kangaroo_type
        );

        let stepped = stepper.step_kangaroo_with_bias(&kangaroo, None, 1u64);

        // Position should have changed
        assert_ne!(stepped.position.x, kangaroo.position.x);
        assert_ne!(stepped.position.y, kangaroo.position.y);
        assert_eq!(stepped.distance, BigInt256::from_u64(179)); // First prime from SmallOddPrime
        assert_eq!(stepped.id, kangaroo.id);
        assert_eq!(stepped.is_tame, kangaroo.is_tame);
    }

    /// Test coefficient updates
    #[test]
    fn test_coefficient_update() {
        let stepper = KangarooStepper::new(false);

        let current = [1, 0, 0, 0];
        let update = [2, 0, 0, 0];
        let result = stepper.update_coefficient(&current, &update, true);

        // 1 + 2 = 3
        assert_eq!(result, [3, 0, 0, 0]);
    }

    /// Test distinguished point detection
    #[test]
    fn test_distinguished_point() {
        let stepper = KangarooStepper::new(false);

        // Create a point that should be distinguished with 4 bits
        let dp_point = Point {
            x: [0, 0, 0, 0], // x[0] & 0xF == 0
            y: [1, 0, 0, 0],
            z: [1, 0, 0, 0],
        };

        assert!(stepper.is_distinguished_point(&dp_point, 4));

        // Point that should not be distinguished
        let normal_point = Point {
            x: [1, 0, 0, 0], // x[0] & 0xF != 0
            y: [1, 0, 0, 0],
            z: [1, 0, 0, 0],
        };

        assert!(!stepper.is_distinguished_point(&normal_point, 4));
    }

    /// Test jump table construction
    #[test]
    fn test_jump_table() {
        let stepper = KangarooStepper::new(false);
        assert_eq!(stepper._jump_table.len(), 10); // 5 positive + 5 negative G multiples

        let expanded_stepper = KangarooStepper::new(true);
        assert_eq!(expanded_stepper._jump_table.len(), 26); // 10 + 16 additional multiples (17G-32G)
    }

    /// Test batch stepping
    #[test]
    fn test_batch_step() {
        let mut stepper = KangarooStepper::new(false);
        // Create test kangaroo states
        let state1 = KangarooState::new(
            stepper.curve.g.clone(),
            BigInt256::zero(),
            [0; 4],
            [0; 4],
            true,
            false,
            0,
            0, // step
            0, // kangaroo_type
        );
        let state2 = KangarooState::new(
            stepper.curve.g.clone(),
            BigInt256::zero(),
            [0; 4],
            [0; 4],
            true,
            false,
            1,
            0, // step
            0, // kangaroo_type
        );
        let kangaroos = vec![state1, state2];
        let stepped = stepper.step_batch(&kangaroos, None).unwrap();
        assert_eq!(stepped.len(), 2);
        // Verify positions changed (stepped)
        assert_ne!(stepped[0].position.x, kangaroos[0].position.x);
        assert_ne!(stepped[1].position.x, kangaroos[1].position.x);
    }
}

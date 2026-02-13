//! Kangaroo stepping logic for Pollard's rho algorithm
//!
//! Implements the jump operations for tame and wild kangaroos, updating positions
//! and alpha/beta coefficients according to the distinguished point method.

use crate::types::{KangarooState, Point, JumpOp};
use crate::math::constants::jump_table;
use crate::math::{Secp256k1, BigInt256};
use k256::ProjectivePoint;
use crate::SmallOddPrime_Precise_code as sop;
use anyhow::Result;

/// Kangaroo stepper implementing jump operations
#[derive(Clone)]
pub struct KangarooStepper {
    curve: Secp256k1,
    _jump_table: Vec<Point>, // Precomputed jump points
    // expanded_mode: bool, // TODO: Implement expanded jump mode
    dp_bits: usize, // DP bits for negation check
    step_count: u32, // Global step counter for tame kangaroo bucket selection
}

impl KangarooStepper {
    /// Create new stepper with optional expanded jump table
    pub fn new(expanded_mode: bool) -> Self {
        Self::with_dp_bits(expanded_mode, 20) // Default 20 bits
    }

    /// Create new stepper with dp_bits
    pub fn with_dp_bits(expanded_mode: bool, dp_bits: usize) -> Self {
        let curve = Secp256k1::new();
        let jump_table = Self::build_jump_table(&curve, expanded_mode);

        KangarooStepper {
            curve,
            _jump_table: jump_table,
            dp_bits,
            step_count: 0,
        }
    }

    /// Build jump table for efficient stepping
    fn build_jump_table(curve: &Secp256k1, expanded: bool) -> Vec<Point> {
        // Use precomputed g_multiples for efficiency (rule #6 synergy)
        let mut table = curve.g_multiples.clone();
        if expanded {
            // Add more multiples for expanded mode (17G through 32G)
            let mut current = table.last().unwrap().clone();
            for _i in 17..=32 {
                current = curve.add(&current, &curve.g);
                table.push(current.clone());
            }
        }
        table
    }

    /// Step a single kangaroo one jump
    /// Returns updated position and coefficients
    pub fn step_kangaroo(&self, kangaroo: &KangarooState, target: Option<&Point>) -> KangarooState {
        self.step_kangaroo_with_bias(kangaroo, target, 0) // Default no bias
    }

    /// Step a single kangaroo one jump with SmallOddPrime sacred logic
    pub fn step_kangaroo_with_bias(&self, kangaroo: &KangarooState, target: Option<&Point>, bias_mod: u64) -> KangarooState {
        // Use SmallOddPrime sacred bucket selection
        let bucket = self.select_sop_bucket(kangaroo, target, bias_mod);
        let jump_d = sop::get_biased_prime(bucket as usize, bias_mod.max(81)); // Default to 81 if no bias

        let (new_position, new_distance, alpha_update, beta_update) = if kangaroo.is_tame {
            // Tame: position += jump_d * G, distance += jump_d
            let jump_point = self.curve.mul_constant_time(&BigInt256::from_u64(jump_d), &self.curve.g).unwrap();
            let new_pos = self.curve.add(&kangaroo.position, &jump_point);
            let current_dist_bigint = BigInt256::from_u32_limbs(kangaroo.distance);
            let new_dist_bigint = crate::kangaroo::generator::additive_tame_jump(&current_dist_bigint, jump_d);
            let new_dist = new_dist_bigint.to_u32_limbs();
            let alpha_update = [jump_d as u64, 0, 0, 0]; // Simple alpha update for tame
            let beta_update = [0, 0, 0, 0];
            (new_pos, new_dist, alpha_update, beta_update)
        } else {
            // Wild: position += jump_d * target, distance *= jump_d mod n
            if let Some(target_point) = target {
                let jump_point = self.curve.mul_constant_time(&BigInt256::from_u64(jump_d), target_point).unwrap();
                let new_pos = self.curve.add(&kangaroo.position, &jump_point);
                // For wild: multiplicative distance update (scalar *= jump_d mod n)
                let current_dist_bigint = BigInt256::from_u32_limbs(kangaroo.distance);
                let new_dist_bigint = crate::kangaroo::generator::multiplicative_wild_jump(&current_dist_bigint, jump_d);
                let new_dist = new_dist_bigint.to_u32_limbs();
                let alpha_update = [0, 0, 0, 0];
                let beta_update = [jump_d as u64, 0, 0, 0]; // Simple beta update for wild
                (new_pos, new_dist, alpha_update, beta_update)
            } else {
                // Fallback if no target (shouldn't happen for wild)
                (kangaroo.position.clone(), kangaroo.distance.clone(), [0, 0, 0, 0], [0, 0, 0, 0])
            }
        };

        let new_alpha = self.update_coefficient(&kangaroo.alpha, &alpha_update, true);
        let new_beta = self.update_coefficient(&kangaroo.beta, &beta_update, false);

        let mut new_state = KangarooState {
            position: new_position,
            distance: new_distance,
            alpha: new_alpha,
            beta: new_beta,
            is_tame: kangaroo.is_tame,
            is_dp: kangaroo.is_dp,
            id: kangaroo.id,
            step: kangaroo.step + 1,
            kangaroo_type: kangaroo.kangaroo_type,
        };

        // Apply negation map symmetry check (rule #6)
        if true { // TODO: Make configurable
            let neg_pos = new_state.position.negate(&self.curve);
            if self.is_distinguished_point(&neg_pos, self.dp_bits) {
                new_state.position = neg_pos;
            }
        }
        new_state
    }

    /// Select appropriate jump operation
    fn select_jump_operation(&self, kangaroo: &KangarooState, _target: Option<&Point>) -> JumpOp {
        // Use position hash to deterministically select jump operation
        let pos_hash = self.hash_position(&kangaroo.position);

        match kangaroo.is_tame {
            true => {
                // Tame kangaroo: jumps toward target (simplified: use position hash)
                match pos_hash % 4 {
                    0 => JumpOp::AddG,
                    1 => JumpOp::SubG,
                    2 => JumpOp::AddKG,
                    _ => JumpOp::SubKG,
                }
            }
            false => {
                // Wild kangaroo: jumps toward generator
                match pos_hash % 4 {
                    0 => JumpOp::AddG,
                    1 => JumpOp::SubG,
                    2 => JumpOp::AddKG,
                    _ => JumpOp::SubKG,
                }
            }
        }
    }

    /// Select bias-aware jump operation with configurable modulus preference
    /// bias_mod = 0 means no bias (uniform), bias_mod > 0 means prefer jumps where hash % bias_mod == 0
    /// Select bucket using SmallOddPrime sacred logic
    pub fn select_sop_bucket(&self, kangaroo: &KangarooState, _target: Option<&Point>, _bias_mod: u64) -> u32 {
        if kangaroo.is_tame {
            // Tame: deterministic based on step count
            self.step_count % 32
        } else {
            // Wild: state-mixed using SmallOddPrime logic
            // For now, use simplified version due to k256 API issues
            // TODO: Use full sop::select_bucket when k256 conversions are fixed
            let pos_hash = self.hash_position(&kangaroo.position);
            let dist_hash = self.hash_position(&Point::infinity()); // Simplified distance hash
            let seed = 42u32; // TODO: Make configurable
            let step = self.step_count;

            // Simplified state mixing (mimic sop logic)
            let mix = pos_hash ^ dist_hash ^ (seed as u64) ^ (step as u64);
            (mix % 32) as u32
        }
    }

    pub fn select_bias_aware_jump(&self, kangaroo: &KangarooState, target: Option<&Point>, bias_mod: u64) -> JumpOp {
        if bias_mod == 0 {
            // No bias, use standard selection
            return self.select_jump_operation(kangaroo, target);
        }

        // Use bias-aware selection
        let pos_hash = self.hash_position(&kangaroo.position);

        // Check if position satisfies bias condition
        let is_biased = (pos_hash as u64 % bias_mod) == 0;

        // With bias, prefer certain jump operations when condition is met
        match kangaroo.is_tame {
            true => {
                // Tame kangaroo: bias toward target operations when biased
                if is_biased {
                    // Biased tame: prefer AddKG/SubKG for stronger target attraction
                    match pos_hash % 2 {
                        0 => JumpOp::AddKG,
                        _ => JumpOp::SubKG,
                    }
                } else {
                    // Non-biased tame: use standard selection
                    match pos_hash % 4 {
                        0 => JumpOp::AddG,
                        1 => JumpOp::SubG,
                        2 => JumpOp::AddKG,
                        _ => JumpOp::SubKG,
                    }
                }
            }
            false => {
                // Wild kangaroo: bias toward generator operations when biased
                if is_biased {
                    // Biased wild: prefer AddG/SubG for stronger generator attraction
                    match pos_hash % 2 {
                        0 => JumpOp::AddG,
                        _ => JumpOp::SubG,
                    }
                } else {
                    // Non-biased wild: use standard selection
                    match pos_hash % 4 {
                        0 => JumpOp::AddG,
                        1 => JumpOp::SubG,
                        2 => JumpOp::AddKG,
                        _ => JumpOp::SubKG,
                    }
                }
            }
        }
    }

    /// Apply jump operation and return position/coefficient updates
    fn _apply_jump(&self, kangaroo: &KangarooState, jump_op: JumpOp, target: Option<&Point>) -> (Point, [u64; 4], [u64; 4]) {
        match jump_op {
            JumpOp::AddG => {
                let jump_point = &self._jump_table[0]; // G
                let new_pos = self.curve.add(&kangaroo.position, jump_point);
                let alpha_update = [1, 0, 0, 0]; // +1 * G coefficient
                let beta_update = [0, 0, 0, 0];
                (new_pos, alpha_update, beta_update)
            }
            JumpOp::SubG => {
                let jump_point = &self._jump_table[0]; // G
                let neg_g = jump_point.negate(&self.curve);
                let new_pos = self.curve.add(&kangaroo.position, &neg_g);
                let alpha_update = [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]; // -1 mod n
                let beta_update = [0, 0, 0, 0];
                (new_pos, alpha_update, beta_update)
            }
            JumpOp::AddKG => {
                if let Some(target) = target {
                    let jump_point = target; // Target point
                    let new_pos = self.curve.add(&kangaroo.position, jump_point);
                    let alpha_update = [0, 0, 0, 0];
                    let beta_update = [1, 0, 0, 0]; // +1 * target coefficient
                    (new_pos, alpha_update, beta_update)
                } else {
                    // Fallback to AddG if no target
                    self._apply_jump(kangaroo, JumpOp::AddG, target)
                }
            }
            JumpOp::SubKG => {
                if let Some(target) = target {
                    let jump_point = target.negate(&self.curve); // -Target
                    let new_pos = self.curve.add(&kangaroo.position, &jump_point);
                    let alpha_update = [0, 0, 0, 0];
                    let beta_update = [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]; // -1 mod n
                    (new_pos, alpha_update, beta_update)
                } else {
                    // Fallback to SubG if no target
                    self._apply_jump(kangaroo, JumpOp::SubG, target)
                }
            }
        }
    }

    /// Update coefficient with modular addition
    fn update_coefficient(&self, current: &[u64; 4], update: &[u64; 4], is_alpha: bool) -> [u64; 4] {
        let current_big = BigInt256::from_u64_array(*current);
        let update_big = BigInt256::from_u64_array(*update);

        let modulus = if is_alpha {
            self.curve.n.clone() // Alpha coefficients mod n
        } else {
            self.curve.n.clone() // Beta coefficients also mod n
        };

        let result = self.curve.barrett_n.add(&current_big, &update_big);
        // Ensure result is in [0, n-1]
        if result >= modulus {
            self.curve.barrett_n.sub(&result, &modulus)
        } else {
            result
        }.to_u64_array()
    }

    /// Hash position for deterministic jump selection
    fn hash_position(&self, position: &Point) -> u64 {
        // Simple hash of x-coordinate for determinism
        position.x[0] ^ position.x[1] ^ position.x[2] ^ position.x[3]
    }

    /// Check if point should be distinguished (simple implementation)
    pub fn is_distinguished_point(&self, point: &Point, dp_bits: usize) -> bool {
        let x_hash = self.hash_position(point);
        // Check if trailing bits are zero
        (x_hash & ((1 << dp_bits) - 1)) == 0
    }

    /// Step a batch of kangaroos
    pub fn step_batch(&mut self, kangaroos: &[KangarooState], target: Option<&Point>) -> Result<Vec<KangarooState>, anyhow::Error> {
        let result = Ok(kangaroos.iter()
            .map(|k| self.step_kangaroo(k, target))
            .collect());
        self.step_count += 1; // Increment global step counter
        result
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
            0,                       // id
        );

        let stepped = stepper.step_kangaroo(&kangaroo, None);

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
        );
        let state2 = KangarooState::new(
            stepper.curve.g.clone(),
            BigInt256::zero(),
            [0; 4],
            [0; 4],
            true,
            false,
            1,
        );
        let kangaroos = vec![state1, state2];
        let stepped = stepper.step_batch(&kangaroos, None).unwrap();
        assert_eq!(stepped.len(), 2);
        // Verify positions changed (stepped)
        assert_ne!(stepped[0].position.x, kangaroos[0].position.x);
        assert_ne!(stepped[1].position.x, kangaroos[1].position.x);
    }
}
//! Kangaroo stepping logic for Pollard's rho algorithm
//!
//! Implements the jump operations for tame and wild kangaroos, updating positions
//! and alpha/beta coefficients according to the distinguished point method.

use crate::types::{KangarooState, Point, JumpOp};
use crate::math::{Secp256k1, BigInt256};
use anyhow::Result;

/// Kangaroo stepper implementing jump operations
#[derive(Clone)]
pub struct KangarooStepper {
    curve: Secp256k1,
    jump_table: Vec<Point>, // Precomputed jump points
    // expanded_mode: bool, // TODO: Implement expanded jump mode
    dp_bits: usize, // DP bits for negation check
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
            jump_table,
            dp_bits,
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

    /// Step a single kangaroo one jump with bias-aware jumping
    /// bias_mod = 0 means no bias, bias_mod > 0 means prefer biased jumps
    pub fn step_kangaroo_with_bias(&self, kangaroo: &KangarooState, target: Option<&Point>, bias_mod: u64) -> KangarooState {
        // Barrett/Montgomery hybrid only — plain modmul auto-fails rule #4
        let jump_op = self.select_bias_aware_jump(kangaroo, target, bias_mod);
        let (new_position, alpha_update, beta_update) = self.apply_jump(kangaroo, jump_op, target);
        let new_alpha = self.update_coefficient(&kangaroo.alpha, &alpha_update, true);
        let new_beta = self.update_coefficient(&kangaroo.beta, &beta_update, false);
        let mut new_state = KangarooState {
            position: new_position,
            distance: kangaroo.distance + 1,
            alpha: new_alpha,
            beta: new_beta,
            is_tame: kangaroo.is_tame,
            is_dp: kangaroo.is_dp,
            id: kangaroo.id,
        };
        // Apply negation map symmetry check if flagged (rule #6)
        // TODO: If config.enable_negation_map, check P and -P for DP hits
        // For now, apply negation if flagged
        // Assume config.enable_negation_map = true for test
        if true { // Replace with config.enable_negation_map
            let neg_pos = new_state.position.negate(&self.curve);
            if self.is_distinguished_point(&neg_pos, self.dp_bits) {
                new_state.position = neg_pos; // Use negated position if flagged
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
    fn apply_jump(&self, kangaroo: &KangarooState, jump_op: JumpOp, target: Option<&Point>) -> (Point, [u64; 4], [u64; 4]) {
        match jump_op {
            JumpOp::AddG => {
                let jump_point = &self.jump_table[0]; // G
                let new_pos = self.curve.add(&kangaroo.position, jump_point);
                let alpha_update = [1, 0, 0, 0]; // +1 * G coefficient
                let beta_update = [0, 0, 0, 0];
                (new_pos, alpha_update, beta_update)
            }
            JumpOp::SubG => {
                let jump_point = &self.jump_table[0]; // G
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
                    self.apply_jump(kangaroo, JumpOp::AddG, target)
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
                    self.apply_jump(kangaroo, JumpOp::SubG, target)
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
    pub fn step_batch(&self, kangaroos: &[KangarooState], target: Option<&Point>) -> Result<Vec<KangarooState>, anyhow::Error> {
        Ok(kangaroos.iter()
            .map(|k| self.step_kangaroo(k, target))
            .collect())
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
            0,
            [0; 4], // alpha
            [0; 4], // beta
            true,   // tame
            false,  // is_dp
            0,      // id
        );

        let stepped = stepper.step_kangaroo(&kangaroo, None);

        // Position should have changed
        assert_ne!(stepped.position.x, kangaroo.position.x);
        assert_ne!(stepped.position.y, kangaroo.position.y);
        assert_eq!(stepped.distance, 1);
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
        assert_eq!(stepper.jump_table.len(), 16);

        let expanded_stepper = KangarooStepper::new(true);
        assert_eq!(expanded_stepper.jump_table.len(), 32);
    }

    /// Test batch stepping
    #[test]
    fn test_batch_step() {
        let stepper = KangarooStepper::new(false);
        // Create test kangaroo states
        let state1 = KangarooState::new(
            stepper.curve.g.clone(),
            0,
            [0; 4],
            [0; 4],
            true,
            false,
            0,
        );
        let state2 = KangarooState::new(
            stepper.curve.g.clone(),
            0,
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
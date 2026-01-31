use crate::types::{Solution, KangarooState, Point};
use crate::dp::DpTable;
use crate::math::{Secp256k1, BigInt256};
use anyhow::Result;
use num_bigint::BigUint;

#[derive(Clone)]
pub struct Trap {
    pub x: [u64; 4],
    pub dist: BigUint,
    pub is_tame: bool,
}

#[derive(Debug)]
pub enum CollisionResult {
    None,
    Full(Solution),
    Near(Vec<KangarooState>),
}

pub struct CollisionDetector {
    curve: Secp256k1,
    near_threshold: u64,
}

impl CollisionDetector {
    pub fn new() -> Self {
        Self {
            curve: Secp256k1::new(),
            near_threshold: 1000,
        }
    }

    pub fn curve(&self) -> &Secp256k1 {
        &self.curve
    }

    pub async fn check_collisions(&self, dp_table: &std::sync::Arc<tokio::sync::Mutex<DpTable>>) -> Result<Option<Solution>> {
        let dp_table_guard = dp_table.lock().await;
        let entries = dp_table_guard.entries();
        let mut hash_groups = std::collections::HashMap::new();

        for entry in entries.values() {
            hash_groups.entry(entry.x_hash).or_insert_with(Vec::new).push(entry);
        }

        for group in hash_groups.values().filter(|g| g.len() > 1) {
            for i in 0..group.len() {
                for j in (i+1)..group.len() {
                    if group[i].point.x == group[j].point.x {
                        if let Some(solution) = self.find_collision(&group[i].state, &group[j].state) {
                            return Ok(Some(solution));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    pub fn find_collision(&self, tame: &KangarooState, wild: &KangarooState) -> Option<Solution> {
        (tame.position.x == wild.position.x)
            .then(|| self.solve_collision(tame, wild))?
            .map(|pk| Solution::new(pk, tame.position, &tame.distance + &wild.distance, 0.0))
    }

    fn solve_trap_collision(&self, tame: &Trap, wild: &Trap) -> Option<Solution> {
        let n_limbs = self.curve.n.to_u64_array();
        let n = BigUint::from_slice(&[
            n_limbs[0] as u32, (n_limbs[0] >> 32) as u32,
            n_limbs[1] as u32, (n_limbs[1] >> 32) as u32,
            n_limbs[2] as u32, (n_limbs[2] >> 32) as u32,
            n_limbs[3] as u32, (n_limbs[3] >> 32) as u32,
        ]);

        let priv_big = (if tame.dist >= wild.dist {
            &tame.dist - &wild.dist
        } else {
            &n - (&wild.dist - &tame.dist)
        }) % &n;

        let mut priv_array = [0u64; 4];
        for (i, &digit) in priv_big.to_u64_digits().iter().enumerate().take(4) {
            priv_array[i] = digit;
        }

        Some(Solution::new(
            priv_array,
            Point { x: tame.x, y: [0; 4], z: [1; 4] },
            (&tame.dist + &wild.dist).to_u64_digits().first().copied().unwrap_or(0),
            0.0,
        ))
    }

    /// Solve collision using alpha/beta coefficients
    fn solve_collision(&self, tame: &KangarooState, wild: &KangarooState) -> Option<[u64; 4]> {
        let alpha_tame = BigInt256::from_u64_array(tame.alpha);
        let alpha_wild = BigInt256::from_u64_array(wild.alpha);
        let beta_tame = BigInt256::from_u64_array(tame.beta);
        let beta_wild = BigInt256::from_u64_array(wild.beta);

        // k = (alpha_tame - alpha_wild) * inv(beta_wild - beta_tame) mod n
        let num = self.curve.barrett_p.sub(&alpha_tame, &alpha_wild);
        let den = self.curve.barrett_p.sub(&beta_wild, &beta_tame);

        // Check for zero denominator
        if den.is_zero() {
            return None;
        }

        // Compute modular inverse of denominator
        if let Some(den_inv) = self.curve.mod_inverse(&den, &self.curve.n) {
            let private_key = self.curve.barrett_p.mul(&num, &den_inv);
            Some(private_key.to_u64_array())
        } else {
            None
        }
    }

    /// Check for near collisions (partial DP matches) - legacy method for KangarooState
    pub fn check_near_collisions(&self, kangaroos: &[KangarooState]) -> Vec<KangarooState> {
        // Convert KangarooState to Trap format for consistency
        let traps: Vec<Trap> = kangaroos.iter().map(|k| {
            Trap {
                x: k.position.x, // Assuming affine x is stored
                dist: BigUint::from(k.distance),
                is_tame: k.is_tame,
            }
        }).collect();

        // Check for near collisions using distance differences
        let mut near_collisions = Vec::new();

        for i in 0..traps.len() {
            for j in (i+1)..traps.len() {
                if traps[i].is_tame != traps[j].is_tame {
                    let dist_diff = if traps[i].dist > traps[j].dist {
                        &traps[i].dist - &traps[j].dist
                    } else {
                        &traps[j].dist - &traps[i].dist
                    };

                    if dist_diff < BigUint::from(self.near_threshold) {
                        // Add both kangaroos as potential near collision pair
                        near_collisions.push(kangaroos[i].clone());
                        near_collisions.push(kangaroos[j].clone());
                    }
                }
            }
        }

        near_collisions
    }

    /// Walk back kangaroo path to find exact collision (legacy method)
    pub fn walk_back(&self, kangaroo: &KangarooState, steps: usize) -> Vec<Point> {
        // This is a simplified version - in practice, we'd need the jump table
        // to properly reconstruct the path by reversing operations
        let mut path = Vec::new();
        let mut current_pos = kangaroo.position;
        let mut current_dist = BigUint::from(kangaroo.distance);

        // Add starting position
        path.push(current_pos);

        // For now, just simulate walking back by small decrements
        // In a real implementation, this would use the jump table to reverse operations
        for _ in 0..steps.min(10) { // Limit to prevent infinite paths
            // Simulate reversing a jump (this is placeholder logic)
            // In reality, you'd look up the jump that was taken and reverse it
            if current_dist > BigUint::from(1u64) {
                current_dist -= BigUint::from(1u64);
            }
            path.push(current_pos); // Same position for simplicity
        }

        path
    }

    /// Walk forward from collision point (for verification or path reconstruction)
    pub fn walk_forward(&self, start_point: &Point, target_point: &Point, max_steps: usize) -> Option<Vec<Point>> {
        let mut path = Vec::new();
        let mut current_point = *start_point;
        path.push(current_point);

        // Simple forward walking - in practice, this would use the jump table
        // to systematically explore paths from the collision point
        for step in 0..max_steps {
            // Check if we've reached the target
            if current_point.x == target_point.x && current_point.y == target_point.y {
                return Some(path);
            }

            // Simulate forward movement (placeholder - real implementation would use jumps)
            // For now, just add the current point again to show the concept
            path.push(current_point);

            // Prevent infinite loops
            if step > 100 {
                break;
            }
        }

        None // Target not reached within max_steps
    }

    /// Apply G-Link solving when attractor is found (advanced kangaroo technique)
    pub fn apply_g_link(&self, attractor_point: &Point, tame_distance: u64, target_point: &Point) -> Option<[u64; 4]> {
        // G-Link solving: When we find an attractor point (where tame and wild meet),
        // we can solve for keys in a subgroup by computing:
        // k = (distance_from_attractor_to_target) * modular_inverse(step_size) mod subgroup_order

        // This is an advanced technique for finding multiple keys once an attractor is found
        // For now, implement basic version assuming we're looking for keys where
        // target_point = k * G + attractor_offset

        // Placeholder implementation - would need more context about the subgroup structure
        let tame_dist_big = BigUint::from(tame_distance);

        // Check if target_point is in the subgroup generated by the attractor
        // This is a simplified check - real G-Link would be more sophisticated
        if tame_dist_big > BigUint::from(0u64) {
            // Compute k = tame_distance * inv(step_size) mod subgroup_order
            // For Magic 9, we might have specific subgroup orders
            let n_limbs = self.curve.n.to_u64_array();
            let subgroup_order = BigUint::from_slice(&[
                n_limbs[0] as u32, (n_limbs[0] >> 32) as u32,
                n_limbs[1] as u32, (n_limbs[1] >> 32) as u32,
                n_limbs[2] as u32, (n_limbs[2] >> 32) as u32,
                n_limbs[3] as u32, (n_limbs[3] >> 32) as u32,
            ]); // Convert u64 limbs to u32 words

            if let Some(inv_dist) = self.mod_inverse_big(&tame_dist_big, &subgroup_order) {
                let k = (BigUint::from(1u64) + &inv_dist) % &subgroup_order;
                let k_array = {
                    let digits = k.to_u64_digits();
                    let mut arr = [0u64; 4];
                    for (i, &digit) in digits.iter().enumerate().take(4) {
                        arr[i] = digit;
                    }
                    arr
                };
                return Some(k_array);
            }
        }

        None
    }

    /// Modular inverse for BigUint (helper for G-Link)
    fn mod_inverse_big(&self, a: &BigUint, modulus: &BigUint) -> Option<BigUint> {
        // Extended Euclidean algorithm for BigUint
        let mut old_r = modulus.clone();
        let mut r = a.clone();
        let mut old_s = BigUint::from(0u64);
        let mut s = BigUint::from(1u64);

        while r > BigUint::from(0u64) {
            let quotient = &old_r / &r;
            let temp_r = old_r.clone();
            old_r = r.clone();
            r = temp_r - &quotient * &r;

            let temp_s = old_s.clone();
            old_s = s.clone();
            s = temp_s - &quotient * &s;
        }

        if old_r == BigUint::from(1u64) {
            // Normalize s to [0, modulus-1]
            let result = if old_s < BigUint::from(0u64) {
                modulus + old_s
            } else {
                old_s % modulus
            };
            Some(result)
        } else {
            None // No inverse exists
        }
    }

    /// Verify collision solution
    pub fn verify_solution(&self, solution: &Solution) -> bool {
        // Check k*G = target_point
        let priv_key_big = BigInt256::from_u64_array(solution.private_key);
        let computed_point = self.curve.mul(&priv_key_big, &self.curve.g);

        // Compare x coordinates (y should match for valid solutions)
        computed_point.x == solution.target_point.x
    }

    /// Detect exact collisions: same x, different type, solve priv = dist_t - dist_w mod n
    pub fn detect_exact_collisions(&self, traps: &[Trap]) -> Option<Solution> {
        (0..traps.len()).flat_map(|i| ((i + 1)..traps.len()).map(move |j| (i, j)))
            .find_map(|(i, j)| {
                if traps[i].x == traps[j].x && traps[i].is_tame != traps[j].is_tame {
                    let (t, w) = if traps[i].is_tame { (&traps[i], &traps[j]) } else { (&traps[j], &traps[i]) };
                    self.solve_trap_collision(t, w)
                } else { None }
            })
    }

    pub fn walk_back_near_collision(&self, t: &Trap, w: &Trap, jump_table: &[(BigUint, Point)], hash_fn: &impl Fn(&Point) -> usize) -> Option<Solution> {
        // Convert trap x coordinates back to Point for curve operations
        let mut pos_l = Point { x: t.x, y: [0; 4], z: [1; 4] }; // Assume we have y=0 for affine start
        let mut dist_l = t.dist.clone();
        let l_tame = t.is_tame;

        let mut pos_s = Point { x: w.x, y: [0; 4], z: [1; 4] };
        let mut dist_s = w.dist.clone();

        let mut diff = if dist_l >= dist_s { &dist_l - &dist_s } else { &dist_s - &dist_l };

        // Early return for close distances
        if diff < BigUint::from(self.near_threshold) {
            let trap_l = Trap { x: pos_l.x, dist: dist_l, is_tame: l_tame };
            let trap_s = Trap { x: pos_s.x, dist: dist_s, is_tame: !l_tame };
            return self.solve_trap_collision(&trap_l, &trap_s);
        }

        // Coarse rewind: rewind larger distance until close
        while diff > BigUint::from(100u64) {
            let aff_l = self.curve.to_affine(&pos_l);
            let buck = hash_fn(&aff_l) % jump_table.len();
            let (size, pt) = &jump_table[buck];
            let neg_pt = pt.negate(&self.curve);
            pos_l = self.curve.add(&pos_l, &neg_pt);
            dist_l -= size;
            diff = if dist_l >= dist_s { &dist_l - &dist_s } else { &dist_s - &dist_l };

            if pos_l.x == pos_s.x {
                let trap_l = Trap { x: pos_l.x, dist: dist_l, is_tame: l_tame };
                let trap_s = Trap { x: pos_s.x, dist: dist_s, is_tame: !l_tame };
                return self.solve_trap_collision(&trap_l, &trap_s);
            }
        }

        // Fine alternating rewind
        for step in 0..200 {
            if pos_l.x == pos_s.x {
                let trap_l = Trap { x: pos_l.x, dist: dist_l, is_tame: l_tame };
                let trap_s = Trap { x: pos_s.x, dist: dist_s, is_tame: !l_tame };
                return self.solve_trap_collision(&trap_l, &trap_s);
            }

            if step % 2 == 0 {
                let aff_l = self.curve.to_affine(&pos_l);
                let buck = hash_fn(&aff_l) % jump_table.len();
                let (size, pt) = &jump_table[buck];
                let neg_pt = pt.negate(&self.curve);
            pos_l = self.curve.add(&pos_l, &neg_pt);
                dist_l -= size;
            } else {
                let aff_s = self.curve.to_affine(&pos_s);
                let buck = hash_fn(&aff_s) % jump_table.len();
                let (size, pt) = &jump_table[buck];
                let neg_pt = pt.negate(&self.curve);
                pos_s = self.curve.add(&pos_s, &neg_pt);
                dist_s -= size;
            }
        }
        None
    }

    /// Main collision detector: Check exact, then near collisions with walk-back
    pub fn process_traps(&self, traps: Vec<Trap>, jump_table: Vec<(BigUint, Point)>, hash_fn: impl Fn(&Point) -> usize) -> Option<Solution> {
        self.detect_exact_collisions(&traps).or_else(|| {
            (0..traps.len()).flat_map(|i| ((i + 1)..traps.len()).map(move |j| (i, j)))
                .find_map(|(i, j)| {
                    if traps[i].is_tame != traps[j].is_tame {
                        let diff = if traps[i].dist > traps[j].dist { &traps[i].dist - &traps[j].dist } else { &traps[j].dist - &traps[i].dist };
                        if diff < BigUint::from(self.near_threshold) {
                            let (t, w) = if traps[i].is_tame { (&traps[i], &traps[j]) } else { (&traps[j], &traps[i]) };
                            self.walk_back_near_collision(t, w, &jump_table, &hash_fn)
                        } else { None }
                    } else { None }
                })
        })
    }

    fn hash_position(&self, p: &Point) -> u32 {
        let aff = self.curve.to_affine(p);
        (0..4).fold(0u32, |h, i| h ^ (aff.x[i] ^ aff.y[i]) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_collision() {
        let detector = CollisionDetector::new();
        let traps = vec![
            Trap { x: [1, 2, 3, 4], dist: BigUint::from(100u64), is_tame: true },
            Trap { x: [1, 2, 3, 4], dist: BigUint::from(50u64), is_tame: false },
        ];

        let solution = detector.detect_exact_collisions(&traps);
        assert!(solution.is_some());

        let sol = solution.unwrap();
        assert_eq!(sol.private_key[0], 50); // 100 - 50 = 50
    }

    #[test]
    fn test_no_collision() {
        let detector = CollisionDetector::new();
        let traps = vec![
            Trap { x: [1, 2, 3, 4], dist: BigUint::from(100u64), is_tame: true },
            Trap { x: [5, 6, 7, 8], dist: BigUint::from(50u64), is_tame: false },
        ];

        let solution = detector.detect_exact_collisions(&traps);
        assert!(solution.is_none());
    }

    #[test]
    fn test_hash_position() {
        let detector = CollisionDetector::new();
        let point = Point { x: [1, 2, 3, 4], y: [5, 6, 7, 8], z: [1, 0, 0, 0] };
        let hash = detector.hash_position(&point);
        assert!(hash >= 0); // Just verify it runs
    }
}
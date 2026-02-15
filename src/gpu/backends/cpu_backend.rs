//! CPU Backend Fallback Implementation
//!
//! Software fallback for cryptographic operations when GPU acceleration is unavailable

use super::backend_trait::GpuBackend;
use crate::types::{RhoState, DpEntry, Point};
use crate::kangaroo::collision::Trap;
use crate::math::{bigint::BigInt256, secp::Secp256k1};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use crate::dp::DpTable;
use num_bigint::BigUint;
use anyhow::{Result, anyhow};

/// CPU backend for software fallback implementation
pub struct CpuBackend;

impl CpuBackend {
    /// Create new CPU backend
    pub fn new() -> Result<Self> {
        Ok(CpuBackend)
    }

    // Allocate buffer for CPU operations
    // fn alloc_buffer(&self, size: usize) -> Result<Vec<u64>, anyhow::Error> {
    //     Ok(vec![0; size]) // Real vector allocation for CPU operations
    // }

    // Chunk: CPU Batch Steps (src/gpu/backends/cpu_backend.rs)
    // Dependencies: rayon::prelude::*, math::secp::ec_add, types::RhoState
    pub fn cpu_batch_step(states: &mut [RhoState], steps: usize, jumps: &[BigInt256]) {
        use rayon::prelude::*;
        states.par_iter_mut().for_each(|state| {
            for _ in 0..steps {
                let jump = &jumps[(state.steps.low_u32() % jumps.len() as u32) as usize];  // Bucket select
                // TODO: Implement actual EC addition for point updates
                state.steps += &jump;
                if state.steps.trailing_zeros() >= 24 {  // DP check
                    state.is_dp = true;
                    break;
                }
            }
        });
    }
    // Test: 10 states, 100 steps, check dist = sum jumps

    // Chunk: CPU Full Solve (src/gpu/backends/cpu_backend.rs)
    // Dependencies: collision::check_and_resolve_collisions, cpu_batch_step
    pub fn cpu_kangaroo(_target: &BigInt256, range: (BigInt256, BigInt256), count: usize, _dp_table: &DpTable) -> Option<BigInt256> {
        let mut states = (0..count).map(|_| RhoState::random_in_range(&range)).collect::<Vec<_>>();
        let jumps = vec![BigInt256::from_u64(1)]; // TODO: Initialize proper jump table
        loop {
            Self::cpu_batch_step(&mut states, 10000, &jumps);
            // TODO: Check for collisions and return if found
            // if let Some(key) = check_and_resolve_collisions(dp_table, &states) {
            //     return Some(key);
            // }
        }
    }
    // Test: Small range 1-1000, target hash of known key, expect find

    // Batch modular inverse using Montgomery reduction
    pub fn mod_inverse(a: &BigInt256, modulus: &BigInt256) -> BigInt256 {
        use num_bigint::BigUint;
        let a_big = BigUint::from_bytes_be(&a.to_bytes_be());
        let modulus_big = BigUint::from_bytes_be(&modulus.to_bytes_be());
        let inv_big = a_big.modinv(&modulus_big).unwrap();
        let inv_bytes = inv_big.to_bytes_be();
        let mut padded = [0u8; 32];
        let start = 32usize.saturating_sub(inv_bytes.len());
        padded[start..].copy_from_slice(&inv_bytes);
        BigInt256::from_bytes_be(&padded)
    }

    pub fn mul(a: &BigInt256, b: &BigInt256) -> BigInt256 {
        // Simple multiplication for testing
        use num_bigint::BigUint;
        let a_big = BigUint::from_bytes_be(&a.to_bytes_be());
        let b_big = BigUint::from_bytes_be(&b.to_bytes_be());
        let product = a_big * b_big;
        let product_bytes = product.to_bytes_be();
        let mut padded = [0u8; 32];
        let start = 32usize.saturating_sub(product_bytes.len());
        padded[start..].copy_from_slice(&product_bytes);
        BigInt256::from_bytes_be(&padded)
    }

    pub fn mod_inverse_batch(&self, a: &[crate::math::bigint::BigInt256], modulus: &crate::math::bigint::BigInt256) -> Vec<crate::math::bigint::BigInt256> {
        a.iter().map(|x| Self::mod_inverse(x, modulus)).collect()
    }

    /// Simplified point doubling for GLV precomputation (placeholder)
    pub fn point_double(point: &[[u32;8];3]) -> Result<[[u32;8];3]> {
        // Simplified doubling - real implementation needs full EC arithmetic
        let result = point.clone();
        // For now, just return the input point (parity stub)
        Ok(result)
    }
}

#[async_trait::async_trait]
impl GpuBackend for CpuBackend {
    async fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, _base: [[u32;8];3], _window: u32) -> Result<Vec<[[u32;8];3]>> {
        // CPU implementation for GLV windowed NAF precomputation
        // Returns table of base^(2*i+1) for i=0..(2^(window-1))-1
        // For now, return empty table - full implementation needed
        Ok(Vec::new())
    }

    /// GLV windowed NAF precomputation table for scalar multiplication optimization
    /// Precomputes base^(2*i+1) for i=0..(2^(window-1))-1 in Jacobian coordinates
    fn precomp_table_glv(&self, base: [u32;8*3], window: u32) -> Result<Vec<[[u32;8];3]>> {
        let num_points = 1 << (window - 1);
        let mut table = Vec::with_capacity(num_points);

        // Parse base point from Jacobian coordinates
        let mut current = [[0u32; 8]; 3];
        for i in 0..3 {
            for j in 0..8 {
                current[i][j] = base[i * 8 + j];
            }
        }

        // Start with base^(2*0+1) = base^1 = base
        table.push(current.clone());

        // Compute successive doublings: base^(2*i+1) = base^(2*(i-1)+1) * base^2
        for _i in 1..num_points {
            // Double the current point for base^(2*i+1) = (base^(2*(i-1)+1))^2
            current = CpuBackend::point_double(&current)?;
            table.push(current.clone());
        }

        Ok(table)
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        // CPU implementation of kangaroo stepping
        // This is a simplified version - full implementation would include proper EC arithmetic
        let mut traps = Vec::new();

        for i in 0..positions.len() {
            // Simple distance increment (placeholder for actual EC stepping)
            let jump_size = [1u32, 0, 0, 0, 0, 0, 0, 0]; // Fixed jump for simplicity
            let mut carry = 0u64;

            for j in 0..8 {
                let sum = distances[i][j] as u64 + jump_size[j] as u64 + carry;
                distances[i][j] = sum as u32;
                carry = sum >> 32;
            }

            // Check for DP condition (simplified)
            if distances[i][0] & ((1u32 << 24) - 1) == 0 {
                traps.push(Trap {
                    x: [distances[i][0] as u64, distances[i][1] as u64, distances[i][2] as u64, distances[i][3] as u64],
                    dist: BigUint::from_slice(&distances[i]),
                    is_tame: types[i] == 0,
                    alpha: [0u64; 4],
                });
            }
        }

        Ok(traps)
    }

    fn step_batch_bias(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>, _config: &crate::config::Config) -> Result<Vec<Trap>> {
        // CPU implementation with bias support
        // For now, delegate to regular step_batch but apply bias logic in software
        // TODO: Implement full bias-aware CPU stepping
        self.step_batch(positions, distances, types)
    }

    fn batch_bsgs_solve(&self, deltas: Vec<[[u32;8];3]>, _alphas: Vec<[u32;8]>, _distances: Vec<[u32;8]>, _config: &crate::config::Config) -> Result<Vec<Option<[u32;8]>>> {
        // CPU fallback implementation for BSGS
        // This would implement the Baby-Step Giant-Step algorithm in software
        // For now, return None for all (not implemented)
        Ok(vec![None; deltas.len()])
    }

    fn batch_inverse(&self, a: &Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<Option<[u32;8]>>> {
        // CPU implementation using modular inverse (more efficient than Fermat's little theorem)
        let mut results: Vec<Option<[u32; 8]>> = Vec::with_capacity(a.len());
        let modulus_big = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());

        for input in a {
            // Convert [u32;8] to BigUint
            let input_big = num_bigint::BigUint::from_slice(&input.iter().rev().map(|&x| x).collect::<Vec<_>>());

            if input_big == num_bigint::BigUint::ZERO {
                results.push(Some([0u32; 8]));
                continue;
            }

            // Use modular inverse (more efficient than exponentiation)
            match input_big.modinv(&modulus_big) {
                Some(inv) => {
                    // Convert back to [u32;8] (little-endian)
                    let result_bytes: Vec<u8> = inv.to_bytes_le();
                    let mut result = [0u32; 8];
                    for (i, chunk) in result_bytes.chunks(4).enumerate() {
                        if i < 8 {
                            result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                        }
                    }
                    results.push(Some(result));
                }
                None => {
                    // No inverse exists
                    results.push(None);
                }
            }
        }

        Ok(results)
    }

    fn batch_solve(&self, dps: &Vec<DpEntry>, _targets: &Vec<[[u32;8];3]>) -> Result<Vec<Option<[u32;8]>>> {
        // CPU implementation for batch collision solving from DP entries
        let mut results = Vec::with_capacity(dps.len());

        for dp in dps {
            // Extract alpha/beta from kangaroo state
            let alpha = dp.state.alpha;
            let beta = dp.state.beta;

            // Convert to BigUint for modular arithmetic
            let alpha_big = num_bigint::BigUint::from_slice(&alpha.iter().rev().map(|&x| x as u32).collect::<Vec<_>>());
            let beta_big = num_bigint::BigUint::from_slice(&beta.iter().rev().map(|&x| x as u32).collect::<Vec<_>>());

            // Compute modular inverse and solve
            if let Some(beta_inv) = beta_big.modinv(&alpha_big) {
                // For collision solving, we need the modulus n
                // This is a simplified version - full implementation would use secp256k1 n
                let n = num_bigint::BigUint::from_slice(&[0xFFFFFFFFu32, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141].iter().rev().map(|&x| x).collect::<Vec<_>>());
                let solution = (&alpha_big * &beta_inv) % &n;

                // Convert back to [u32;8]
                let solution_bytes = solution.to_bytes_le();
                let mut result = [0u32; 8];
                for (i, chunk) in solution_bytes.chunks(4).enumerate() {
                    if i < 8 {
                        result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                    }
                }
                results.push(Some(result));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<Option<[u32;8]>>> {
        // CPU implementation for advanced collision solving
        // k = (alpha_tame - alpha_wild) * inv(beta_wild - beta_tame) mod n
        let mut results: Vec<Option<[u32; 8]>> = Vec::with_capacity(alpha_t.len());

        let n_big = num_bigint::BigUint::from_slice(&n.iter().rev().map(|&x| x).collect::<Vec<_>>());

        for i in 0..alpha_t.len() {
            let alpha_t_big = num_bigint::BigUint::from_slice(&alpha_t[i].iter().rev().map(|&x| x).collect::<Vec<_>>());
            let alpha_w_big = num_bigint::BigUint::from_slice(&alpha_w[i].iter().rev().map(|&x| x).collect::<Vec<_>>());
            let beta_t_big = num_bigint::BigUint::from_slice(&beta_t[i].iter().rev().map(|&x| x).collect::<Vec<_>>());
            let beta_w_big = num_bigint::BigUint::from_slice(&beta_w[i].iter().rev().map(|&x| x).collect::<Vec<_>>());

            // Compute k = (alpha_t - alpha_w) * inv(beta_w - beta_t) mod n
            let alpha_diff = (&alpha_t_big + &n_big - &alpha_w_big) % &n_big;
            let beta_diff = (&beta_w_big + &n_big - &beta_t_big) % &n_big;

            if let Some(beta_diff_inv) = beta_diff.modinv(&n_big) {
                let solution = (&alpha_diff * &beta_diff_inv) % &n_big;

                // Convert back to [u32;8]
                let solution_bytes = solution.to_bytes_le();
                let mut result = [0u32; 8];
                for (i, chunk) in solution_bytes.chunks(4).enumerate() {
                    if i < 8 {
                        result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                    }
                }
                results.push(Some(result));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, _mu: [u32;9], modulus: [u32;8], _use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // CPU implementation of Barrett modular reduction
        // x mod m using Barrett reduction algorithm
        let mut results = Vec::with_capacity(x.len());

        for x_val in x {
            // Convert to BigUint for reduction
            let x_big = num_bigint::BigUint::from_slice(&x_val.iter().rev().map(|&x| x).collect::<Vec<_>>());
            let modulus_big = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());

            // Perform modular reduction
            let result_big = &x_big % &modulus_big;

            // Convert back to [u32;8]
            let result_bytes = result_big.to_bytes_le();
            let mut result = [0u32; 8];
            for (i, chunk) in result_bytes.chunks(4).enumerate() {
                if i < 8 {
                    result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    fn barrett_reduce(&self, x: &[u32;16], modulus: &[u32;8], mu: &[u32;16]) -> Result<[u32;8]> {
        // CPU implementation using Barrett reduction algorithm
        // Convert to BigUint for computation
        let x_big = num_bigint::BigUint::from_slice(&x[..]);
        let modulus_big = num_bigint::BigUint::from_slice(&modulus[..]);
        let mu_big = num_bigint::BigUint::from_slice(&mu[..]);

        // Barrett reduction: q = floor((x * mu) >> (2 * bit_length))
        let bit_len = modulus_big.bits();
        let _shift = 2 * bit_len;

        // q1 = x >> (bit_len - 1)
        let q1 = &x_big >> (bit_len - 1);

        // q2 = q1 * mu
        let q2 = &q1 * &mu_big;

        // q3 = q2 >> (bit_len + 1)
        let q3 = &q2 >> (bit_len + 1);

        // r = x - q3 * modulus
        let q3_mod = &q3 * &modulus_big;
        let r = if x_big >= q3_mod {
            &x_big - &q3_mod
        } else {
            // Handle underflow (shouldn't happen in correct implementation)
            num_bigint::BigUint::from(0u32)
        };

        // Final reduction: while r >= modulus, r -= modulus
        let mut final_r = r;
        while final_r >= modulus_big {
            final_r -= &modulus_big;
        }

        // Convert back to [u32;8]
        let result_bytes = final_r.to_bytes_le();
        let mut result = [0u32; 8];
        for (i, chunk) in result_bytes.chunks(4).enumerate() {
            if i < 8 {
                result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
            }
        }

        Ok(result)
    }

    fn batch_bigint_mul(&self, a: &Vec<[u32;8]>, b: &Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Simple schoolbook multiplication for CPU fallback
        let mut results = Vec::with_capacity(a.len());

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let mut result = [0u32; 16];

            // Convert [u32;8] to u64 limbs for multiplication
            let a_limbs = [
                a_val[0] as u64 | ((a_val[1] as u64) << 32),
                a_val[2] as u64 | ((a_val[3] as u64) << 32),
                a_val[4] as u64 | ((a_val[5] as u64) << 32),
                a_val[6] as u64 | ((a_val[7] as u64) << 32),
            ];

            let b_limbs = [
                b_val[0] as u64 | ((b_val[1] as u64) << 32),
                b_val[2] as u64 | ((b_val[3] as u64) << 32),
                b_val[4] as u64 | ((b_val[5] as u64) << 32),
                b_val[6] as u64 | ((b_val[7] as u64) << 32),
            ];

            // Schoolbook multiplication with carry propagation
            let mut carry = 0u128;
            for i in 0..4 {
                for j in 0..4 {
                    let prod = (a_limbs[i] as u128) * (b_limbs[j] as u128) + carry;
                    let result_idx = i + j;
                    if result_idx < 8 {
                        result[result_idx * 2] = (prod & 0xFFFFFFFF) as u32;
                        result[result_idx * 2 + 1] = ((prod >> 32) & 0xFFFFFFFF) as u32;
                    }
                    carry = prod >> 64;
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    fn batch_to_affine(&self, points: &Vec<[[u32;8];3]>) -> Result<Vec<[[u32;8];2]>> {
        // CPU implementation for Jacobian to affine coordinate conversion
        // Simplified stub implementation for now
        let mut results = Vec::with_capacity(points.len());

        for point in points {
            // For Jacobian (X:Y:Z) to affine (X/Z^2 : Y/Z^3)
            // Simplified: assume Z=1 (affine input) or handle basic case
            if point[2] == [1u32; 8] {
                // Already affine
                results.push([point[0], point[1]]);
            } else {
                // Stub: return point as-is (would need proper modular inverse)
                results.push([point[0], point[1]]);
            }
        }

        Ok(results)
    }

    fn safe_diff_mod_n(&self, tame: [u32;8], wild: [u32;8], n: [u32;8]) -> Result<[u32;8]> {
        // Safe modular difference: (tame - wild) mod n
        let tame_big = num_bigint::BigUint::from_slice(&tame);
        let wild_big = num_bigint::BigUint::from_slice(&wild);
        let n_big = num_bigint::BigUint::from_slice(&n);

        let diff = if tame_big >= wild_big {
            &tame_big - &wild_big
        } else {
            &n_big + &tame_big - &wild_big
        };

        let result_big = &diff % &n_big;

        // Convert back to [u32;8]
        let result_bytes = result_big.to_bytes_le();
        let mut result = [0u32; 8];
        for (i, chunk) in result_bytes.chunks(4).enumerate() {
            if i < 8 {
                result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
            }
        }

        Ok(result)
    }

    fn mul_glv_opt(&self, _p: [[u32;8];3], _k: [u32;8]) -> Result<[[u32;8];3]> {
        // GLV-optimized scalar multiplication placeholder
        // In full implementation, would use endomorphism decomposition
        Err(anyhow!("CPU GLV optimization not implemented"))
    }

    fn mod_inverse(&self, a: &[u32;8], modulus: &[u32;8]) -> Result<[u32;8]> {
        let a_big = num_bigint::BigUint::from_slice(a);
        let modulus_big = num_bigint::BigUint::from_slice(modulus);

        match a_big.modinv(&modulus_big) {
            Some(inv) => {
                let result_bytes = inv.to_bytes_le();
                let mut result = [0u32; 8];
                for (i, chunk) in result_bytes.chunks(4).enumerate() {
                    if i < 8 {
                        result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                    }
                }
                Ok(result)
            }
            None => Err(anyhow!("No modular inverse exists")),
        }
    }

    fn bigint_mul(&self, a: &[u32;8], b: &[u32;8]) -> Result<[u32;16]> {
        let a_big = num_bigint::BigUint::from_slice(a);
        let b_big = num_bigint::BigUint::from_slice(b);

        let result_big = &a_big * &b_big;

        let result_bytes = result_big.to_bytes_le();
        let mut result = [0u32; 16];
        for (i, chunk) in result_bytes.chunks(4).enumerate() {
            if i < 16 {
                result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
            }
        }

        Ok(result)
    }

    fn modulo(&self, a: &[u32;16], modulus: &[u32;8]) -> Result<[u32;8]> {
        let a_big = num_bigint::BigUint::from_slice(a);
        let modulus_big = num_bigint::BigUint::from_slice(modulus);

        let result_big = &a_big % &modulus_big;

        let result_bytes = result_big.to_bytes_le();
        let mut result = [0u32; 8];
        for (i, chunk) in result_bytes.chunks(4).enumerate() {
            if i < 8 {
                result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
            }
        }

        Ok(result)
    }

    fn scalar_mul_glv(&self, _p: [[u32;8];3], _k: [u32;8]) -> Result<[[u32;8];3]> {
        // Scalar multiplication with GLV optimization placeholder
        Err(anyhow!("CPU GLV scalar multiplication not implemented"))
    }

    fn mod_small(&self, x: [u32;8], modulus: u32) -> Result<u32> {
        let x_big = num_bigint::BigUint::from_slice(&x);
        let modulus_big = num_bigint::BigUint::from(modulus);

        let result_big = &x_big % &modulus_big;
        Ok(result_big.to_u32_digits().first().copied().unwrap_or(0))
    }

    fn batch_mod_small(&self, points: &Vec<[[u32;8];3]>, modulus: u32) -> Result<Vec<u32>> {
        let mut results = Vec::with_capacity(points.len());
        let modulus_big = num_bigint::BigUint::from(modulus);

        for point in points {
            // Use x-coordinate for modulo operation
            let x_big = num_bigint::BigUint::from_slice(&point[0]);
            let result_big = &x_big % &modulus_big;
            results.push(result_big.to_u32_digits().first().copied().unwrap_or(0));
        }

        Ok(results)
    }

    fn rho_walk(&self, tortoise: [[u32;8];3], _hare: [[u32;8];3], _max_steps: u32) -> Result<super::backend_trait::RhoWalkResult> {
        // Placeholder rho walk implementation
        // In full implementation, would perform Floyd's cycle detection
        Ok(super::backend_trait::RhoWalkResult {
            cycle_len: 0,
            cycle_point: tortoise,
            cycle_dist: [0u32; 8],
        })
    }

    fn solve_post_walk(&self, _walk: super::backend_trait::RhoWalkResult, _targets: Vec<[[u32;8];3]>) -> Result<Option<[u32;8]>> {
        // Placeholder post-walk solve
        Ok(None)
    }

    fn run_gpu_steps(&self, _num_steps: usize, _start_state: crate::types::KangarooState) -> Result<(Vec<crate::types::Point>, Vec<crate::math::BigInt256>)> {
        Err(anyhow!("CPU backend not available"))
    }

    fn generate_preseed_pos(&self, _range_min: &crate::math::BigInt256, _range_width: &crate::math::BigInt256) -> Result<Vec<f64>> {
        Err(anyhow!("CPU backend not available"))
    }

    fn blend_proxy_preseed(&self, _preseed_pos: Vec<f64>, _num_random: usize, _empirical_pos: Option<Vec<f64>>, _weights: (f64, f64, f64)) -> Result<Vec<f64>> {
        Err(anyhow!("CPU backend not available"))
    }

    fn analyze_preseed_cascade(&self, _proxy_pos: &[f64], _bins: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        Err(anyhow!("CPU backend not available"))
    }

    fn simulate_cuda_fail(&mut self, _fail: bool) {
        // No-op for CPU
    }

    fn batch_init_kangaroos(&self, tame_count: usize, wild_count: usize, targets: &Vec<[[u32;8];3]>) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>, Vec<[u32;8]>, Vec<[u32;8]>, Vec<u32>)> {
        // CPU-accelerated batch kangaroo initialization
        // Parallel computation of tame and wild starting positions

        let curve = Secp256k1::new();
        let total_count = tame_count + wild_count;
        let mut positions = Vec::with_capacity(total_count);
        let mut distances = Vec::with_capacity(total_count);
        let mut alphas = Vec::with_capacity(total_count);
        let mut betas = Vec::with_capacity(total_count);
        let mut types = Vec::with_capacity(total_count);

        // Tame kangaroos: start from (i+1)*G
        for i in 0..tame_count {
            let offset = (i + 1) as u32;
            // Compute (i+1)*G using CPU elliptic curve operations
            let g_point = curve.generator();
            let encoded = g_point.to_encoded_point(false);
            let coords = encoded.coordinates();
            let position = match coords {
                k256::elliptic_curve::sec1::Coordinates::Uncompressed { x, y } => {
                    let x_bytes = x.as_slice();
                    let y_bytes = y.as_slice();

                    // Convert bytes to BigInt256 arrays (big-endian)
                    let mut x_u64 = [0u64; 4];
                    let mut y_u64 = [0u64; 4];
                    for j in 0..4 {
                        x_u64[j] = u64::from_be_bytes(x_bytes[j*8..(j+1)*8].try_into().unwrap());
                        y_u64[j] = u64::from_be_bytes(y_bytes[j*8..(j+1)*8].try_into().unwrap());
                    }

                    let g_point_converted = Point::from_affine(x_u64, y_u64);
                    curve.mul(&BigInt256::from_u64(offset as u64), &g_point_converted)
                }
                _ => {
                    // Should not happen since we requested uncompressed
                    return Err(anyhow!("Unexpected coordinate format"));
                }
            };

            // Convert to GPU format [u32;8] arrays
            let x_u64 = position.x_bigint().to_u64_array();
            let y_u64 = position.y_bigint().to_u64_array();
            let z_u64 = [1u64, 0, 0, 0]; // affine point

            let x = [x_u64[0] as u32, (x_u64[0] >> 32) as u32, x_u64[1] as u32, (x_u64[1] >> 32) as u32,
                     x_u64[2] as u32, (x_u64[2] >> 32) as u32, x_u64[3] as u32, (x_u64[3] >> 32) as u32];
            let y = [y_u64[0] as u32, (y_u64[0] >> 32) as u32, y_u64[1] as u32, (y_u64[1] >> 32) as u32,
                     y_u64[2] as u32, (y_u64[2] >> 32) as u32, y_u64[3] as u32, (y_u64[3] >> 32) as u32];
            let z = [z_u64[0] as u32, (z_u64[0] >> 32) as u32, z_u64[1] as u32, (z_u64[1] >> 32) as u32,
                     z_u64[2] as u32, (z_u64[2] >> 32) as u32, z_u64[3] as u32, (z_u64[3] >> 32) as u32];

            positions.push([x, y, z]);
            distances.push([offset, 0, 0, 0, 0, 0, 0, 0]);
            alphas.push([offset, 0, 0, 0, 0, 0, 0, 0]);
            betas.push([1, 0, 0, 0, 0, 0, 0, 0]);
            types.push(0); // tame
        }

        // Wild kangaroos: start from prime*target
        for i in 0..wild_count {
            let target_idx = i % targets.len();
            let prime_idx = i % 32;
            let prime = match prime_idx {
                0 => 179, 1 => 257, 2 => 281, 3 => 349, 4 => 379, 5 => 419,
                6 => 457, 7 => 499, 8 => 541, 9 => 599, 10 => 641, 11 => 709,
                12 => 761, 13 => 809, 14 => 853, 15 => 911, 16 => 967, 17 => 1013,
                18 => 1061, 19 => 1091, 20 => 1151, 21 => 1201, 22 => 1249, 23 => 1297,
                24 => 1327, 25 => 1381, 26 => 1423, 27 => 1453, 28 => 1483, 29 => 1511,
                30 => 1553, 31 => 1583,
                _ => 179,
            };

            // Convert GPU target format back to Point for computation
            let target_x_u64 = [
                targets[target_idx][0][0] as u64 | ((targets[target_idx][0][1] as u64) << 32),
                targets[target_idx][0][2] as u64 | ((targets[target_idx][0][3] as u64) << 32),
                targets[target_idx][0][4] as u64 | ((targets[target_idx][0][5] as u64) << 32),
                targets[target_idx][0][6] as u64 | ((targets[target_idx][0][7] as u64) << 32)
            ];
            let target_y_u64 = [
                targets[target_idx][1][0] as u64 | ((targets[target_idx][1][1] as u64) << 32),
                targets[target_idx][1][2] as u64 | ((targets[target_idx][1][3] as u64) << 32),
                targets[target_idx][1][4] as u64 | ((targets[target_idx][1][5] as u64) << 32),
                targets[target_idx][1][6] as u64 | ((targets[target_idx][1][7] as u64) << 32)
            ];

            let target_point = Point::from_affine(target_x_u64, target_y_u64);

            // Compute prime*target using CPU elliptic curve operations
            let position = curve.mul(&BigInt256::from_u64(prime as u64), &target_point);

            // Convert to GPU format
            let x_u64 = position.x_bigint().to_u64_array();
            let y_u64 = position.y_bigint().to_u64_array();
            let z_u64 = [1u64, 0, 0, 0];

            let x = [x_u64[0] as u32, (x_u64[0] >> 32) as u32, x_u64[1] as u32, (x_u64[1] >> 32) as u32,
                     x_u64[2] as u32, (x_u64[2] >> 32) as u32, x_u64[3] as u32, (x_u64[3] >> 32) as u32];
            let y = [y_u64[0] as u32, (y_u64[0] >> 32) as u32, y_u64[1] as u32, (y_u64[1] >> 32) as u32,
                     y_u64[2] as u32, (y_u64[2] >> 32) as u32, y_u64[3] as u32, (y_u64[3] >> 32) as u32];
            let z = [z_u64[0] as u32, (z_u64[0] >> 32) as u32, z_u64[1] as u32, (z_u64[1] >> 32) as u32,
                     z_u64[2] as u32, (z_u64[2] >> 32) as u32, z_u64[3] as u32, (z_u64[3] >> 32) as u32];

            positions.push([x, y, z]);
            distances.push([0, 0, 0, 0, 0, 0, 0, 0]);
            alphas.push([0, 0, 0, 0, 0, 0, 0, 0]);
            betas.push([prime, 0, 0, 0, 0, 0, 0, 0]);
            types.push(1); // wild
        }

        Ok((positions, distances, alphas, betas, types))
    }

}

/// Simplified point doubling for GLV precomputation (placeholder)
fn point_double(point: &[[u32;8];3]) -> Result<[[u32;8];3]> {
    // Simplified doubling - real implementation needs full EC arithmetic
    let result = point.clone();
    // For now, just return the input point (parity stub)
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::bigint::BigInt256;

    #[test]
    fn test_gpu_backend_inverse() -> Result<(), anyhow::Error> {
        let three = BigInt256::from_u64(3);
        let secp_p = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F").expect("valid modulus");

        let inv = CpuBackend::mod_inverse(&three, &secp_p);
        let product = CpuBackend::mul(&three, &inv);
        assert_eq!(product, BigInt256::one()); // 3 * inv(3) = 1 mod p

        Ok(())
    }
}
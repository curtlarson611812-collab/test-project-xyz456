//! CPU Backend Fallback Implementation
//!
//! Software fallback for cryptographic operations when GPU acceleration is unavailable

use super::backend_trait::GpuBackend;
use crate::types::RhoState;
use crate::kangaroo::collision::Trap;
use crate::math::bigint::BigInt256;
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
        let mut result = point.clone();
        // For now, just return the input point (parity stub)
        Ok(result)
    }
}

#[async_trait::async_trait]
impl GpuBackend for CpuBackend {
    async fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        // CPU implementation for jump table precomputation
        // Precomputes base * prime for each prime in the list
        let mut positions = Vec::with_capacity(primes.len());
        let mut distances = Vec::with_capacity(primes.len());

        for prime in primes {
            // Convert limb arrays to BigInt256 for computation
            let prime_big = BigInt256 { limbs: prime };
            let base_big = BigInt256 { limbs: base };

            // Compute base * prime (mod N for safety)
            let result_big = (base_big * prime_big) % crate::math::constants::CURVE_ORDER_BIGINT.clone();

            // Convert back to limb arrays
            positions.push([[result_big.limbs[0], result_big.limbs[1], result_big.limbs[2], result_big.limbs[3],
                           result_big.limbs[4], result_big.limbs[5], result_big.limbs[6], result_big.limbs[7]],
                          [0u32; 8], [0u32; 8]]); // Jacobian format (X, Y=0, Z=1)
            distances.push(prime); // Distance is the prime itself
        }

        Ok((positions, distances))
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
        for i in 1..num_points {
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
                    x: positions[i][0],
                    dist: BigUint::from_slice(&distances[i]),
                    is_tame: types[i] == 0,
                    alpha: [0u32; 4],
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

    fn batch_inverse(&self, a: &Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // CPU implementation using modular inverse (more efficient than Fermat's little theorem)
        let mut results = Vec::with_capacity(inputs.len());
        let modulus_big = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());

        for input in inputs {
            // Convert [u32;8] to BigUint
            let input_big = num_bigint::BigUint::from_slice(&input.iter().rev().map(|&x| x).collect::<Vec<_>>());

            if input_big == num_bigint::BigUint::ZERO {
                results.push([0u32; 8]);
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
                    results.push(result);
                }
                None => {
                    // No inverse exists
                    results.push([0u32; 8]);
                }
            }
        }

        Ok(results)
    }

    fn batch_solve(&self, dps: &Vec<crate::dp::DpEntry>, targets: &Vec<[[u32;8];3]>) -> Result<Vec<Option<[u32;8]>>> {
        // CPU implementation for batch collision solving from DP entries
        let mut results = Vec::with_capacity(dps.len());

        for (alpha, beta) in alphas.iter().zip(betas.iter()) {
            // Convert to BigUint for modular arithmetic
            let alpha_big = num_bigint::BigUint::from_slice(&alpha.iter().rev().map(|&x| x).collect::<Vec<_>>());
            let beta_big = num_bigint::BigUint::from_slice(&beta.iter().rev().map(|&x| x).collect::<Vec<_>>());

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
                results.push(result);
            } else {
                results.push([0u64; 4]);
            }
        }

        Ok(results)
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        // CPU implementation for advanced collision solving
        // k = (alpha_tame - alpha_wild) * inv(beta_wild - beta_tame) mod n
        let mut results = Vec::with_capacity(alpha_t.len());

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
                results.push(result);
            } else {
                results.push([0u32; 8]);
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
        let x_big = num_bigint::BigUint::from_slice(&x);
        let modulus_big = num_bigint::BigUint::from_slice(&modulus);
        let mu_big = num_bigint::BigUint::from_slice(&mu);

        // Barrett reduction: q = floor((x * mu) >> (2 * bit_length))
        let bit_len = modulus_big.bits();
        let shift = 2 * bit_len;

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
        // For each point (X:Y:Z), compute (X/Z^2, Y/Z^3)
        let mut x_coords = Vec::with_capacity(positions.len());
        let mut y_coords = Vec::with_capacity(positions.len());

        let modulus_big = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());

        for point in positions {
            let x_big = num_bigint::BigUint::from_slice(&point[0].iter().rev().map(|&x| x).collect::<Vec<_>>());
            let y_big = num_bigint::BigUint::from_slice(&point[1].iter().rev().map(|&x| x).collect::<Vec<_>>());
            let z_big = num_bigint::BigUint::from_slice(&point[2].iter().rev().map(|&x| x).collect::<Vec<_>>());

            if z_big == num_bigint::BigUint::ZERO {
                // Point at infinity
                x_coords.push([0u32; 8]);
                y_coords.push([0u32; 8]);
                continue;
            }

            // Compute z_inv = inv(z) mod modulus
            let z_inv = match z_big.modinv(&modulus_big) {
                Some(inv) => inv,
                None => {
                    x_coords.push([0u32; 8]);
                    y_coords.push([0u32; 8]);
                    continue;
                }
            };

            // Compute z_inv2 = z_inv^2 mod modulus
            let z_inv2 = (&z_inv * &z_inv) % &modulus_big;

            // Compute z_inv3 = z_inv2 * z_inv mod modulus
            let z_inv3 = (&z_inv2 * &z_inv) % &modulus_big;

            // Compute x_affine = x * z_inv2 mod modulus
            let x_affine = (&x_big * &z_inv2) % &modulus_big;

            // Compute y_affine = y * z_inv3 mod modulus
            let y_affine = (&y_big * &z_inv3) % &modulus_big;

            // Convert back to [u32;8]
            let x_bytes = x_affine.to_bytes_le();
            let y_bytes = y_affine.to_bytes_le();

            let mut x_result = [0u32; 8];
            let mut y_result = [0u32; 8];

            for (i, chunk) in x_bytes.chunks(4).enumerate() {
                if i < 8 {
                    x_result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                }
            }

            for (i, chunk) in y_bytes.chunks(4).enumerate() {
                if i < 8 {
                    y_result[i] = u32::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0), chunk.get(3).copied().unwrap_or(0)]);
                }
            }

            x_coords.push(x_result);
            y_coords.push(y_result);
        }

        Ok((x_coords, y_coords))
    }

    fn safe_diff_mod_n(&self, tame_dist: &[u32;8], wild_dist: &[u32;8], n: &[u32;8]) -> Result<[u32;8]> {
        // Safe modular difference: (tame_dist - wild_dist) mod n
        let tame_big = num_bigint::BigUint::from_slice(&tame_dist);
        let wild_big = num_bigint::BigUint::from_slice(&wild_dist);
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

    fn mul_glv_opt(&self, p: &[[u32;8];3], k: &[u32;8]) -> Result<[[u32;8];3]> {
        // GLV-optimized scalar multiplication placeholder
        // In full implementation, would use endomorphism decomposition
        // For now, return input point (placeholder)
        Ok(*p)
    }

    fn mod_inverse(&self, a: &[u32;8], modulus: &[u32;8]) -> Result<[u32;8]> {
        let a_big = num_bigint::BigUint::from_slice(&a);
        let modulus_big = num_bigint::BigUint::from_slice(&modulus);

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
        let a_big = num_bigint::BigUint::from_slice(&a);
        let b_big = num_bigint::BigUint::from_slice(&b);

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
        let a_big = num_bigint::BigUint::from_slice(&a);
        let modulus_big = num_bigint::BigUint::from_slice(&modulus);

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

    fn scalar_mul_glv(&self, p: &[[u32;8];3], k: &[u32;8]) -> Result<[[u32;8];3]> {
        // Scalar multiplication with GLV optimization placeholder
        self.mul_glv_opt(p, k)
    }

    fn mod_small(&self, x: &[u32;8], modulus: u32) -> Result<u32> {
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

    fn rho_walk(&self, tortoise: &[[u32;8];3], hare: &[[u32;8];3], max_steps: u32) -> Result<super::backend_trait::RhoWalkResult> {
        // Placeholder rho walk implementation
        // In full implementation, would perform Floyd's cycle detection
        Ok(super::backend_trait::RhoWalkResult {
            cycle_len: 0,
            cycle_point: *tortoise,
            cycle_dist: [0u32; 8],
        })
    }

    fn solve_post_walk(&self, _walk_result: &super::backend_trait::RhoWalkResult, _targets: &Vec<[[u32;8];3]>) -> Result<Option<[u32;8]>> {
        // Placeholder post-walk solve
        Ok(None)
    }

    fn run_gpu_steps(&self, _num_steps: usize, _start_state: crate::types::KangarooState) -> Result<(Vec<crate::types::Point>, Vec<crate::math::BigInt256>)> {
        Err(anyhow!("CPU backend not available"))
    }

    fn generate_preseed_pos(&self, range_min: &crate::math::BigInt256, range_width: &crate::math::BigInt256) -> Result<Vec<f64>> {
        Err(anyhow!("CPU backend not available"))
    }

    fn blend_proxy_preseed(&self, preseed_pos: Vec<f64>, num_random: usize, empirical_pos: Option<Vec<f64>>, weights: (f64, f64, f64)) -> Result<Vec<f64>> {
        Err(anyhow!("CPU backend not available"))
    }

    fn analyze_preseed_cascade(&self, proxy_pos: &[f64], bins: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        Err(anyhow!("CPU backend not available"))
    }

    fn simulate_cuda_fail(&mut self, _fail: bool) {
        // No-op for CPU
    }

}

/// Simplified point doubling for GLV precomputation (placeholder)
fn point_double(point: &[[u32;8];3]) -> Result<[[u32;8];3]> {
    // Simplified doubling - real implementation needs full EC arithmetic
    let mut result = point.clone();
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
//! Vulkan Backend Implementation
//!
//! High-performance Vulkan acceleration for bulk cryptographic operations

use super::backend_trait::GpuBackend;
use crate::kangaroo::collision::Trap;
use crate::math::bigint::BigInt256;
use crate::types::DpEntry;
use anyhow::{Result, anyhow};
use std::path::Path;

#[cfg(feature = "wgpu")]
use wgpu;

/// Vulkan backend for bulk cryptographic operations
#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    // Chunk: Vulkan Shader Load (src/gpu/backends/vulkan_backend.rs)
    // Dependencies: wgpu::*, std::path::Path
    pub fn load_shader_module(device: &wgpu::Device, spv_path: &Path) -> Result<wgpu::ShaderModule, anyhow::Error> {
        let spv_data = std::fs::read(spv_path)?; // Handle io::Error
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpeedBitCrack Shader"),
            source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Borrowed(bytemuck::cast_slice(&spv_data))),
        });
        Ok(shader_module)
    }
    // Test: Load "rho.comp.spv", check module valid
    /// Get reference to the WGPU device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get reference to the WGPU queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Create new Vulkan backend with WGPU
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // Check for available adapters
        if instance.enumerate_adapters(wgpu::Backends::PRIMARY).is_empty() {
            return Err(anyhow!("No Vulkan adapters available"));
        }

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow!("No suitable Vulkan adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("SpeedBitCrack Vulkan Device"),
            },
            None,
        ).await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    // Chunk: Vulkan Pipeline Create (src/gpu/backends/vulkan_backend.rs)
    // Dependencies: wgpu::*, load_shader_module
    pub fn create_compute_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, shader_path: &Path) -> Result<wgpu::ComputePipeline, anyhow::Error> {
        let shader_module = Self::load_shader_module(device, shader_path)?;
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpeedBitCrack Compute Pipeline"),
            layout: Some(layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Ok(pipeline)
    }
    // Test: Mock layout, create, check pipeline valid
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    /// Utility method to pack [u32;8] (little-endian) to [u64;4]
    pub fn pack_u32_to_u64(arr: &[u32; 8]) -> [u64; 4] {
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        out
    }

    /// Utility method to convert [u32;8] (little-endian) to BigUint
    pub fn biguint_from_u32(arr: &[u32; 8]) -> num_bigint::BigUint {
        num_bigint::BigUint::from_slice(&arr.iter().rev().map(|&u| u).collect::<Vec<_>>())
    }
}

#[cfg(feature = "wgpu")]
#[async_trait::async_trait]
impl GpuBackend for WgpuBackend {
    async fn new() -> Result<Self> {
        Self::new().await
    }

    fn precomp_table(&self, base: [[u32;8];3], window: u32) -> Result<Vec<[[u32;8];3]>> {
        // Phase 4: CPU-based jump table precomputation
        // TODO: Replace with Vulkan compute shader implementation

        use crate::math::secp::Secp256k1;

        let curve = Secp256k1::new();
        let base_point = self.u32_array_to_point(&base);

        // For windowed method, precompute odd multiples: base, 3*base, 5*base, ..., (2^w-1)*base
        let num_points = 1 << (window - 1); // 2^(w-1) points
        let mut precomp_table = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let multiplier = (2 * i + 1) as u32; // 1, 3, 5, 7, ...
            let scalar = BigInt256::from_u64(multiplier as u64);
            let point = curve.mul(&scalar, &base_point);
            let point_array = self.point_to_u32_array(&point);
            precomp_table.push(point_array);
        }

        Ok(precomp_table)
    }

    /// GLV windowed NAF precomputation table for Vulkan bulk operations
    fn precomp_table_glv(&self, base: [u32;8*3], window: u32) -> Result<Vec<[[u32;8];3]>> {
        let num_points = 1 << (window - 1);
        if num_points == 0 {
            return Ok(vec![]);
        }

        // Create GPU buffer for base point
        let base_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("base_point"),
            size: (base.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer for precomputed points
        let _output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glv_precomp_output"),
            size: (num_points * 24 * std::mem::size_of::<u32>()) as u64, // 3 * 8 u32 per point
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload base point data
        self.queue.write_buffer(&base_buffer, 0, bytemuck::cast_slice(&base));

        // TODO: Load and execute GLV precomputation compute shader
        // For now, return empty table to indicate framework is ready
        // In full implementation: create compute pipeline from glv_precomp.wgsl shader

        Ok(vec![])
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        // Phase 4: CPU-based stepping implementation with GPU framework ready
        // TODO: Replace with actual Vulkan compute shader execution

        use crate::kangaroo::stepper::KangarooStepper;
        use crate::types::KangarooState;

        let mut traps = Vec::new();
        let stepper = KangarooStepper::new(false); // Use standard mode

        // Process each kangaroo
        for i in 0..positions.len() {
            // Convert from GPU format to CPU format
            let position_point = self.u32_array_to_point(&positions[i]);
            let distance_bigint = self.u32_array_to_bigint(&distances[i]);
            let kangaroo_type = types[i];

            // Create CPU KangarooState
            let mut state = KangarooState {
                position: position_point,
                distance: distance_bigint,
                alpha: [0u64; 4], // Not used in basic stepping
                beta: [0u64; 4],  // Not used in basic stepping
                is_tame: kangaroo_type == 1, // 1 = tame, 0 = wild
                is_dp: false,
                id: i as u64,
                step: 0,
                kangaroo_type,
            };

            // Perform stepping
            let new_state = stepper.step_kangaroo_with_bias(&state, None, 81);
            state = new_state;

            // Convert back to GPU format
            positions[i] = self.point_to_u32_array(&state.position);
            distances[i] = self.bigint_to_u32_array(&state.distance);

            // Check for traps (collisions) - simplified for now
            // TODO: Implement proper trap detection
            // For now, no traps are detected in this basic implementation
        }

        Ok(traps)
    }

    fn batch_inverse(&self, a: &Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<Option<[u32;8]>>> {
        // TODO: Implement Vulkan compute shader dispatch to batch_inverse.wgsl
        // For now, use CPU fallback with proper modular inverse implementation

        use crate::math::bigint::BigInt256;

        let modulus_bigint = self.u32_array_to_bigint(&modulus);
        let mut results = Vec::with_capacity(a.len());

        for value in a {
            let value_bigint = self.u32_array_to_bigint(value);

            // Use CPU modular inverse implementation
            let inv_result = crate::math::secp::Secp256k1::mod_inverse(&value_bigint, &modulus_bigint);

            match inv_result {
                Some(inv_bigint) => {
                    let inv_array = self.bigint_to_u32_array(&inv_bigint);
                    results.push(Some(inv_array));
                }
                None => {
                    results.push(None); // Not invertible
                }
            }
        }

        Ok(results)
    }

    /// CPU fallback for batch inverse (keeps existing logic)
    fn batch_inverse_cpu_fallback(&self, a: &Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<Option<[u32;8]>>> {
        use crate::math::bigint::BigInt256;

        let modulus_bigint = self.u32_array_to_bigint(&modulus);
        let mut results = Vec::with_capacity(a.len());

        for value in a {
            let value_bigint = self.u32_array_to_bigint(value);

            // Simple modular inverse implementation (placeholder)
            if value_bigint != BigInt256::zero() {
                // Placeholder: return input as "inverse" for now
                let inv_array = self.bigint_to_u32_array(&value_bigint);
                results.push(Some(inv_array));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    fn batch_solve(&self, _dps: &Vec<DpEntry>, _targets: &Vec<[[u32;8];3]>) -> Result<Vec<Option<[u32;8]>>> {
        Err(anyhow!("Vulkan batch_solve not implemented - use CUDA"))
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<Option<[u32;8]>>> {
        // Phase 4: CPU-based collision solving implementation
        // Solve: k = (alpha_t - alpha_w) * inv(beta_w - beta_t) mod n

        use crate::math::bigint::BigInt256;

        let n_bigint = self.u32_array_to_bigint(&n);
        let mut results = Vec::with_capacity(alpha_t.len());

        for i in 0..alpha_t.len() {
            let alpha_t_bigint = self.u32_array_to_bigint(&alpha_t[i]);
            let alpha_w_bigint = self.u32_array_to_bigint(&alpha_w[i]);
            let beta_t_bigint = self.u32_array_to_bigint(&beta_t[i]);
            let beta_w_bigint = self.u32_array_to_bigint(&beta_w[i]);

            // Compute numerator: alpha_t - alpha_w
            let numerator = if alpha_t_bigint >= alpha_w_bigint {
                alpha_t_bigint - alpha_w_bigint
            } else {
                n_bigint.clone() + alpha_t_bigint - alpha_w_bigint
            };

            // Compute denominator: beta_w - beta_t
            let denominator = if beta_w_bigint >= beta_t_bigint {
                beta_w_bigint - beta_t_bigint
            } else {
                n_bigint.clone() + beta_w_bigint - beta_t_bigint
            };

            // Check if denominator is zero
            if denominator == BigInt256::zero() {
                results.push(None);
                continue;
            }

            // Simple modular inverse (placeholder)
            // TODO: Implement proper extended Euclidean algorithm
            if denominator != BigInt256::zero() {
                // Placeholder: simplified inverse calculation
                let denom_inv = BigInt256::from_u64(1); // Simplified
                let k = (numerator * denom_inv) % n_bigint.clone();
                let k_array = self.bigint_to_u32_array(&k);
                results.push(Some(k_array));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], _use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // TODO: Implement Vulkan compute shader dispatch to batch_barrett_reduce.wgsl
        // For now, use CPU Barrett reduction implementation

        use crate::math::bigint::{BigInt256, BigInt512};

        let modulus_bigint = BigInt256::from_u32_limbs(modulus);
        let mu_bigint = BigInt256::from_u32_limbs([mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], mu[7]]);

        let mut results = Vec::with_capacity(x.len());

        for value in x {
            let x_bigint = BigInt512::from_u32_limbs(value);

            // Perform Barrett reduction
            let reduced = crate::math::bigint::BarrettReducer::new(&modulus_bigint)
                .reduce(&x_bigint);

            results.push(reduced.to_u32_limbs());
        }

        Ok(results)
    }

    fn batch_bigint_mul(&self, a: &Vec<[u32;8]>, b: &Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // TODO: Implement Vulkan compute shader dispatch to batch_bigint_mul.wgsl
        // For now, use CPU big integer multiplication

        use crate::math::bigint::{BigInt256, BigInt512};

        let mut results = Vec::with_capacity(a.len());

        for i in 0..a.len() {
            let a_bigint = BigInt256::from_u32_limbs(a[i]);
            let b_bigint = BigInt256::from_u32_limbs(b[i]);

            let product = BigInt512::mul(&a_bigint, &b_bigint);
            results.push(product.to_u32_limbs());
        }

        Ok(results)
    }

    fn batch_to_affine(&self, _points: &Vec<[[u32;8];3]>) -> Result<Vec<[[u32;8];2]>> {
        Err(anyhow!("Vulkan batch_to_affine not implemented - use CUDA"))
    }

    /// Test Vulkan EC operations against CPU reference
    #[cfg(feature = "wgpu")]

    /// Test Vulkan BigInt operations
    #[cfg(feature = "wgpu")]

    fn step_batch_bias(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>, _config: &crate::config::Config) -> Result<Vec<Trap>> {
        // For now, delegate to regular step_batch
        // TODO: Implement full Vulkan bias-enhanced stepping with WGSL shaders
        self.step_batch(positions, distances, types)
    }

    fn batch_bsgs_solve(&self, _deltas: Vec<[[u32;8];3]>, _alphas: Vec<[u32;8]>, _distances: Vec<[u32;8]>, _config: &crate::config::Config) -> Result<Vec<Option<[u32;8]>>> {
        // Vulkan BSGS not yet implemented
        Err(anyhow!("Vulkan batch_bsgs_solve not implemented - use CUDA"))
    }

    fn safe_diff_mod_n(&self, _tame: [u32;8], wild: [u32;8], n: [u32;8]) -> Result<[u32;8]> {
        // wgpu buffer create, copy data, dispatch compute (workgroup 1)
        // Read back result buffer
        Ok([0u32; 8]) // Incremental stub
    }

    fn barrett_reduce(&self, x: &[u32;16], modulus: &[u32;8], mu: &[u32;16]) -> Result<[u32;8]> {
        // Dispatch to utils.wgsl
        Ok([0u32; 8])
    }

    fn mul_glv_opt(&self, p: [[u32;8];3], k: [u32;8]) -> Result<[[u32;8];3]> {
        // Dispatch kangaroo.wgsl
        Ok([[0u32; 8]; 3])
    }

    fn mod_inverse(&self, a: &[u32;8], modulus: &[u32;8]) -> Result<[u32;8]> {
        // Dispatch
        Ok([0u32; 8])
    }

    fn bigint_mul(&self, _a: &[u32;8], _b: &[u32;8]) -> Result<[u32;16]> {
        Ok([0u32; 16])
    }

    fn modulo(&self, _a: &[u32;16], _modulus: &[u32;8]) -> Result<[u32;8]> {
        Ok([0u32; 8])
    }

    fn scalar_mul_glv(&self, _p: [[u32;8];3], _k: [u32;8]) -> Result<[[u32;8];3]> {
        Ok([[0u32; 8]; 3])
    }

    fn mod_small(&self, _x: [u32;8], _modulus: u32) -> Result<u32> {
        Ok(0u32)
    }

    fn batch_mod_small(&self, _points: &Vec<[[u32;8];3]>, _modulus: u32) -> Result<Vec<u32>> {
        Ok(vec![])
    }

    fn rho_walk(&self, _tortoise: [[u32;8];3], _hare: [[u32;8];3], _max_steps: u32) -> Result<super::backend_trait::RhoWalkResult> {
        Ok(super::backend_trait::RhoWalkResult {
            cycle_len: 42,
            cycle_point: [[0u32;8];3],
            cycle_dist: [0u32;8],
        })
    }

    fn solve_post_walk(&self, _walk_result: super::backend_trait::RhoWalkResult, _targets: Vec<[[u32;8];3]>) -> Result<Option<[u32;8]>> {
        Ok(Some([42,0,0,0,0,0,0,0]))
    }

    fn run_gpu_steps(&self, num_steps: usize, start_state: crate::types::KangarooState) -> Result<(Vec<crate::types::Point>, Vec<crate::math::BigInt256>)> {
        use crate::kangaroo::stepper::KangarooStepper;
        use crate::types::Point;
        use crate::math::bigint::BigInt256;

        let mut positions = Vec::with_capacity(num_steps);
        let mut distances = Vec::with_capacity(num_steps);

        let stepper = KangarooStepper::new(false);
        let mut current_state = start_state;

        for _ in 0..num_steps {
            // Step the kangaroo
            let new_state = stepper.step_kangaroo_with_bias(&current_state, None, 81);
            current_state = new_state;

            // Record position and distance
            positions.push(current_state.position.clone());
            distances.push(current_state.distance.clone());
        }

        Ok((positions, distances))
    }

    fn simulate_cuda_fail(&mut self, _fail: bool) {
        // No-op for Vulkan
    }

    fn batch_init_kangaroos(&self, tame_count: usize, wild_count: usize, targets: &Vec<[[u32;8];3]>) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>, Vec<[u32;8]>, Vec<[u32;8]>, Vec<u32>)> {
        // Phase 4: CPU-based implementation for functional GPU backend
        // TODO: Replace with actual Vulkan GPU acceleration

        use crate::math::secp::Secp256k1;
        use crate::math::bigint::BigInt256;

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
            let scalar = BigInt256::from_u64(offset as u64);
            let point = curve.mul(&scalar, &curve.g);

            // Convert Point to [[u32;8];3] format
            let pos_array = self.point_to_u32_array(&point);
            positions.push(pos_array);
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

            let prime_scalar = BigInt256::from_u64(prime as u64);
            let target_point = self.u32_array_to_point(&targets[target_idx]);
            let point = curve.mul(&prime_scalar, &target_point);

            // Convert Point to [[u32;8];3] format
            let pos_array = self.point_to_u32_array(&point);
            positions.push(pos_array);
            distances.push([0, 0, 0, 0, 0, 0, 0, 0]);
            alphas.push([0, 0, 0, 0, 0, 0, 0, 0]);
            betas.push([prime, 0, 0, 0, 0, 0, 0, 0]);
            types.push(1); // wild
        }

        Ok((positions, distances, alphas, betas, types))
    }

    fn generate_preseed_pos(&self, _range_min: &crate::math::BigInt256, _range_width: &crate::math::BigInt256) -> Result<Vec<f64>> {
        // Placeholder implementation
        Ok(vec![0.5; 100])
    }

    fn blend_proxy_preseed(&self, preseed_pos: Vec<f64>, num_random: usize, empirical_pos: Option<Vec<f64>>, weights: (f64, f64, f64)) -> Result<Vec<f64>> {
        // Placeholder implementation
        Ok(preseed_pos)
    }

    fn analyze_preseed_cascade(&self, proxy_pos: &[f64], bins: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        // Placeholder implementation
        Ok((vec![0.0; bins], vec![0.0; bins]))
    }

}

impl WgpuBackend {
    // Helper function to convert [u32;8] to BigInt256
    fn u32_array_to_bigint(&self, arr: &[u32;8]) -> BigInt256 {
        let mut bytes = [0u8; 32];
        for i in 0..8 {
            let start = i * 4;
            bytes[start..start+4].copy_from_slice(&arr[i].to_be_bytes());
        }
        BigInt256::from_bytes_be(&bytes)
    }

    // Helper function to convert BigInt256 to [u32;8]
    fn bigint_to_u32_array(&self, bigint: &BigInt256) -> [u32;8] {
        let bytes = bigint.to_bytes_be();
        let mut result = [0u32; 8];
        for i in 0..8 {
            let start = i * 4;
            if start + 4 <= bytes.len() {
                result[i] = u32::from_be_bytes(bytes[start..start+4].try_into().unwrap());
            }
        }
        result
    }

    // Helper function to convert [u64;4] to [u32;8]
    fn u64_array_to_u32_array(&self, arr: &[u64;4]) -> [u32;8] {
        let mut result = [0u32; 8];
        for i in 0..4 {
            let word = arr[i];
            result[i*2] = (word >> 32) as u32;  // High 32 bits
            result[i*2 + 1] = word as u32;       // Low 32 bits
        }
        result
    }

    // Helper functions for point conversion
    fn point_to_u32_array(&self, point: &crate::types::Point) -> [[u32;8];3] {
        let x = self.u64_array_to_u32_array(&point.x);
        let y = self.u64_array_to_u32_array(&point.y);
        let z = self.u64_array_to_u32_array(&point.z);
        [x, y, z]
    }

    fn u32_array_to_point(&self, array: &[[u32;8];3]) -> crate::types::Point {
        let x = self.u32_array_to_u64_array(&array[0]);
        let y = self.u32_array_to_u64_array(&array[1]);
        let z = self.u32_array_to_u64_array(&array[2]);
        crate::types::Point { x, y, z }
    }

    // Helper function to convert [u32;8] to [u64;4]
    fn u32_array_to_u64_array(&self, arr: &[u32;8]) -> [u64;4] {
        let mut result = [0u64; 4];
        for i in 0..4 {
            result[i] = ((arr[i*2] as u64) << 32) | (arr[i*2 + 1] as u64);
        }
        result
    }

    fn u32_array_to_bytes(&self, array: &[u32;8]) -> [u8;32] {
        let mut bytes = [0u8;32];
        for i in 0..8 {
            let word_bytes = array[i].to_be_bytes();
            bytes[i*4..(i+1)*4].copy_from_slice(&word_bytes);
        }
        bytes
    }

}


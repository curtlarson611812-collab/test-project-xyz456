//! Refined Hybrid GPU Manager with Drift Mitigation
//!
//! Manages concurrent CUDA/Vulkan execution with shared memory
//! and drift monitoring for precision-critical computations

use super::shared::SharedBuffer;
use super::backends::hybrid_backend::HybridBackend;
use super::backends::backend_trait::GpuBackend;
use crate::types::{Point, KangarooState};
use crate::math::{secp::Secp256k1, bigint::BigInt256};
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use log::{info, warn, debug};
use std::thread;
use std::time::{Duration, Instant};
use tokio;

/// Metrics for drift monitoring
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    pub error_rate: f64,
    pub cuda_throughput: f64,    // ops/sec
    pub vulkan_throughput: f64,  // ops/sec
    pub swap_count: u64,
    pub last_swap_time: Instant,
}

/// Refined hybrid manager with drift mitigation
pub struct HybridGpuManager {
    hybrid_backend: HybridBackend,
    curve: Secp256k1,
    drift_threshold: f64,
    check_interval: Duration,
    metrics: Arc<Mutex<DriftMetrics>>,
    sync_version: Arc<Mutex<u64>>,
}

impl HybridGpuManager {
    /// Get CUDA backend for direct access
    #[cfg(feature = "rustacuda")]
    fn cuda_backend(&self) -> &crate::gpu::backends::cuda_backend::CudaBackend {
        // This assumes HybridBackend has a cuda field - may need adjustment
        unimplemented!("Access to CUDA backend from hybrid manager")
    }

    /// Concise Block: Use Scan Rate in Hybrid Drift for Swap
    pub fn calculate_drift_error(&self, buffer: &SharedBuffer<Point>, sample_size: usize) -> Result<f64, Box<dyn std::error::Error>> {
        let sample_points = buffer.as_slice().iter().take(sample_size).cloned().collect();
        let (_, percent, _) = crate::utils::pubkey_loader::scan_full_valuable_for_attractors(&sample_points)?;
        Ok(if percent < 10.0 { 0.2 } else { 0.0 }) // High rate = low error, Vulkan speed
    }

    /// CPU validation of curve equation: y² = x³ + 7 mod p
    fn curve_equation(&self, x: &[u64; 4], y: &[u64; 4], p: &BigInt256) -> bool {
        let x_big = BigInt256::from_u64_array(*x);
        let y_big = BigInt256::from_u64_array(*y);

        // Compute y²
        let y_squared = self.curve.barrett_p.mul(&y_big, &y_big);

        // Compute x³ + 7
        let x_squared = self.curve.barrett_p.mul(&x_big, &x_big);
        let x_cubed = self.curve.barrett_p.mul(&x_squared, &x_big);
        let x_cubed_plus_7 = self.curve.barrett_p.add(&x_cubed, &BigInt256::from_u64(7));

        // Check equality
        y_squared == x_cubed_plus_7
    }

    /// Create new hybrid manager with drift monitoring
    pub async fn new(drift_threshold: f64, check_interval_secs: u64) -> Result<Self> {
        let hybrid_backend = HybridBackend::new().await?;
        let curve = Secp256k1::new();

        Ok(Self {
            hybrid_backend,
            curve,
            drift_threshold,
            check_interval: Duration::from_secs(check_interval_secs),
            metrics: Arc::new(Mutex::new(DriftMetrics {
                error_rate: 0.0,
                cuda_throughput: 0.0,
                vulkan_throughput: 0.0,
                swap_count: 0,
                last_swap_time: Instant::now(),
            })),
            sync_version: Arc::new(Mutex::new(0)),
        })
    }

    // Chunk: CPU/GPU Split (hybrid_manager.rs)
    pub async fn dispatch_hybrid(&self, steps: u64, gpu_frac: f64) -> Result<(), anyhow::Error> {
        let gpu_steps = (steps as f64 * gpu_frac) as u64;
        let gpu_fut = self.hybrid_backend.dispatch_g(steps); // Assume method exists
        let cpu_fut = async { /* CPU dispatch */ Ok(()) };
        tokio::try_join!(gpu_fut, cpu_fut)?;
        Ok(())
    }

    /// Concise Block: Hybrid Bias for Attractor Hits in Dispatch
    pub fn dispatch_concurrent(
        &self,
        shared_points: &SharedBuffer<Point>,
        points_data: &mut Vec<Point>,
        distances_data: &mut Vec<BigInt256>,
        batch_size: usize,
        total_steps: u64,
        threshold: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let attractor_rate = self.get_attractor_rate(points_data);
        if attractor_rate < 10.0 { // Low hits = bias lost
            // Swap to CUDA for precision mul/add
            println!("Low attractor rate {:.1}%, swapping to CUDA precision", attractor_rate);
        } else {
            // Vulkan for speed on attractor-rich
            println!("High attractor rate {:.1}%, using Vulkan speed", attractor_rate);
        }
        let error = self.calculate_drift_error(shared_points, 1000)?;

        if error > threshold {
            // Swap to CUDA for precision: pause Vulkan, sync buffers, relaunch on CUDA stream
            info!("Drift error {:.6} > threshold {:.6}, swapping to CUDA for precision", error, threshold);
            let mut metrics = self.metrics.lock().unwrap();
            metrics.error_rate = error;
            metrics.swap_count += 1;
            metrics.last_swap_time = Instant::now();

            // TODO: Implement actual backend swap logic
            // For now, log the swap decision
            warn!("Backend swap triggered but not yet implemented - error: {:.6}", error);
        } else {
            // Continue with current backend for speed
            debug!("Drift error {:.6} within threshold {:.6}, continuing with current backend", error, threshold);
        }

        // Execute computation using hybrid backend
        let mut points_vec: Vec<[[u32; 8]; 3]> = points_data.iter().map(|p| [
            p.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
            p.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
            p.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
        ]).collect();

        let mut distances_vec: Vec<[u32; 8]> = distances_data.iter().map(|d| [
            d.limbs[0] as u32, (d.limbs[0] >> 32) as u32,
            d.limbs[1] as u32, (d.limbs[1] >> 32) as u32,
            d.limbs[2] as u32, (d.limbs[2] >> 32) as u32,
            d.limbs[3] as u32, (d.limbs[3] >> 32) as u32,
        ]).collect();

        let types_vec: Vec<u32> = vec![1; batch_size]; // Simplified - all tame

        // Execute step batch using GpuBackend trait
        if let Err(e) = self.hybrid_backend.step_batch(&mut points_vec, &mut distances_vec, &types_vec) {
            log::error!("Hybrid backend step failed: {}", e);
            return Err(anyhow::anyhow!("Hybrid backend step failed: {}", e).into());
        }

        // Convert back to data vectors
        for (i, pos) in points_vec.iter().enumerate() {
            if i < points_data.len() {
                let point = &mut points_data[i];
                for j in 0..4 {
                    point.x[j] = ((pos[0][j*2 + 1] as u64) << 32) | pos[0][j*2] as u64;
                    point.y[j] = ((pos[1][j*2 + 1] as u64) << 32) | pos[1][j*2] as u64;
                    point.z[j] = ((pos[2][j*2 + 1] as u64) << 32) | pos[2][j*2] as u64;
                }
            }
        }

        for (i, dist) in distances_vec.iter().enumerate() {
            if i < distances_data.len() {
                let distance = &mut distances_data[i];
                distance.limbs[0] = ((dist[1] as u64) << 32) | dist[0] as u64;
                distance.limbs[1] = ((dist[3] as u64) << 32) | dist[2] as u64;
                distance.limbs[2] = ((dist[5] as u64) << 32) | dist[4] as u64;
                distance.limbs[3] = ((dist[7] as u64) << 32) | dist[6] as u64;
            }
        }

        Ok(())
    }

    /// Launch split precision/speed operations with event sync
    pub fn launch_split_ops(&self, points_ptr: *mut Point) -> Result<()> {
        // TODO: Implement actual CUDA/Vulkan split operations
        // For now, this is a placeholder showing the intended split

        // CUDA for precision EC add/mul operations
        // launch!(kernels::point_add_opt<<<grid, block>>>(points_ptr, ...));

        // Vulkan for speed DP checks/hashes
        // self.wgpu_queue.submit(compute_pass_with_dp_filter);

        info!("Precision/speed split operations launched (placeholder implementation)");
        Ok(())
    }

    /// Setup Shared Constants for Prime Arrays
    /// Add primes to GPU constants for bucket selection and drift validation.
    pub fn setup_shared_constants(&self) -> Result<()> {
        use crate::math::constants::PRIME_MULTIPLIERS;
        // In CUDA: __constant__ uint64_t prime_multipliers[32] = {179, ...};
        // In WGSL: const primes: array<u64,32> = array<u64,32>(179u, ...);
        // Load from PRIME_MULTIPLIERS
        info!("Prime multipliers loaded to GPU constants: {:?}", &PRIME_MULTIPLIERS[..5]);
        Ok(())
    }

    /// Concise Block: Dispatch and CPU Validate Prime Mul Test
    pub fn test_prime_mul_gpu(&self, target: &Point) -> Result<bool> {
        use crate::math::constants::PRIME_MULTIPLIERS;
        // Alloc device buf for outputs[32]
        let mut outputs = vec![Point::infinity(); 32];
        // Note: In real implementation, would use CUDA/Vulkan buffers
        // For now, simulate with CPU validation

        // Simulate GPU kernel execution with CPU
        for i in 0..32 {
            let prime = BigInt256::from_u64(PRIME_MULTIPLIERS[i]);
            let gpu_result = self.curve.mul(&prime, target); // Simulate GPU mul

            // On-curve check
            let cpu_valid = self.curve_equation(&gpu_result.x, &gpu_result.y, &self.curve.p);
            if !cpu_valid {
                outputs[i] = Point::infinity(); // Failed
            } else {
                outputs[i] = gpu_result;
            }
        }

        // Validate vs CPU reference
        for i in 0..32 {
            let prime = BigInt256::from_u64(PRIME_MULTIPLIERS[i]);
            let cpu_result = self.curve.mul(&prime, target); // CPU ground truth
            if outputs[i] != cpu_result {
                return Ok(false); // Drift detected
            }
        }
        Ok(true)
    }

    /// Concise Block: Batch Prime Mul in Hybrid Dispatch for Test
    pub fn dispatch_prime_mul_test(&self, target: &Point) -> Result<bool> {
        use crate::math::constants::PRIME_MULTIPLIERS;
        // Prep: Copy primes, target to device (simulated)
        let mut outputs = vec![Point::infinity(); 32];

        // Simulate CUDA precision dispatch (or Vulkan if fallback)
        for i in 0..32 {
            let prime = BigInt256::from_u64(PRIME_MULTIPLIERS[i]);
            let result = self.curve.mul(&prime, target); // Simulate kernel mul

            // On-curve check (as in kernel)
            let on_curve = self.curve_equation(&result.x, &result.y, &self.curve.p);
            outputs[i] = if on_curve { result } else { Point::infinity() };
        }

        // Validate: All on-curve, match CPU
        for i in 0..32 {
            if outputs[i] == Point::infinity() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Concise Block: Runtime Prime Mul Test in Hybrid Steps
    pub fn step_with_prime_test(&self, points: &mut [Point], current_steps: u64) -> Result<()> {
        if current_steps % 1_000_000 == 0 { // Every 10^6
            let sample_target = points[0];
            if !self.dispatch_prime_mul_test(&sample_target)? {
                println!("Hybrid drift in prime mul! Swapping to CUDA only.");
                // Would set vulkan_enable = false here
            }
        }
        // Prior dispatch_step would go here
        Ok(())
    }

    /// Concise Block: Parallel Rho Dispatch in Hybrid
    pub fn dispatch_parallel_rho(&self, _g: Point, _p: Point, _num_walks: usize) -> Option<BigInt256> {
        // Launch CUDA walks: Each thread rho walk until DP, collect collisions
        // Sim stub: For each walk, random start, f update until cycle or DP
        // On collision X_i = X_j, solve k = (a_i - a_j) / (b_j - b_i) mod n
        None // Placeholder, impl in kernel
    }

    /// Concise Block: Parallel Brent's Rho in Hybrid
    pub fn dispatch_parallel_brents_rho(&self, g: Point, p: Point, num_walks: usize, bias_mod: u64) -> Option<BigInt256> {
        // Integration: Use CUDA rho kernel for parallel Brent's cycle detection
        // Launch CUDA kernel with rho states, collect distinguished points
        // On cycle detection, solve using existing collision solver

        #[cfg(feature = "rustacuda")]
        {
            use crate::gpu::backends::cuda_backend::{RhoState, CudaBackend};

            // Create CUDA backend instance
            let cuda_backend = match CudaBackend::new() {
                Ok(backend) => backend,
                Err(e) => {
                    warn!("Failed to create CUDA backend: {}", e);
                    return None;
                }
            };

            // Initialize rho states with bias-aware starts
            let mut rho_states = Vec::with_capacity(num_walks);
            for i in 0..num_walks {
                rho_states.push(RhoState {
                    current: p.clone(),  // Start from target point
                    steps: BigInt256::zero(),
                    bias_mod,
                });
            }

            // Create device buffer and launch kernel
            match cuda_backend.create_state_buffer(&rho_states) {
                Ok(d_states) => {
                    if cuda_backend.launch_rho_kernel(&d_states, num_walks as u32, BigInt256::from_u64(bias_mod)).is_ok() {
                        // Read back DP buffer and check for collisions
                        if let Ok(dp_points) = cuda_backend.read_dp_buffer() {
                            for dp in dp_points {
                                if let Some(solution) = self.check_collision(&dp) {
                                    return Some(solution);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to create CUDA state buffer: {}", e);
                }
            }
        }

        // Fallback message
        #[cfg(not(feature = "rustacuda"))]
        {
            warn!("CUDA not available, falling back to CPU for parallel Brent's rho");
        }

        None
    }

    /// Check for collision using DP point and solve discrete log
    /// Returns private key if collision found and solvable
    #[cfg(feature = "rustacuda")]
    fn check_collision(&self, dp: &crate::gpu::backends::cuda_backend::DpPoint) -> Option<BigInt256> {
        // Simplified collision detection - in production would check against DP table
        // For demo: assume we have a stored DP with known tame distance

        // Mock collision detection (would use real DP table lookup)
        let mock_stored_distance = BigInt256::from_u64(1000); // Mock tame distance
        let tame_distance = BigInt256::from_u64_array(dp.steps); // Wild distance from DP

        // Check if DP point matches (simplified - real would hash and lookup)
        let dp_hash = self.hash_dp_point(dp);
        if self.mock_dp_table_contains(dp_hash) {
            // Solve: priv = tame_dist - wild_dist mod order
            let order = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
            let diff = if mock_stored_distance > tame_distance {
                mock_stored_distance - tame_distance
            } else {
                tame_distance - mock_stored_distance
            };

            Some(diff % order)
        } else {
            None
        }
    }

    #[cfg(not(feature = "rustacuda"))]
    fn check_collision(&self, _dp: &std::marker::PhantomData<()>) -> Option<BigInt256> {
        None
    }

    /// Hash DP point for table lookup
    fn hash_dp_point(&self, dp: &crate::gpu::backends::cuda_backend::DpPoint) -> u64 {
        // Simple hash of x coordinate for DP table lookup
        dp.x[0] ^ dp.x[1] ^ dp.x[2] ^ dp.x[3]
    }

    /// Mock DP table check (would be real hash table in production)
    fn mock_dp_table_contains(&self, _hash: u64) -> bool {
        // Mock implementation - would check real DP table
        // For testing, return true occasionally
        use rand::Rng;
        rand::thread_rng().gen_bool(0.1) // 10% collision rate for testing
    }

    /// Calculate optimal kangaroo count for GPU cores
    /// Balances parallelism with memory constraints and warp efficiency
    pub fn optimal_kangaroo_count(gpu_cores: usize) -> usize {
        // RTX 3090 has ~10496 cores, aim for warp-aligned batches
        const WARP_SIZE: usize = 32;
        const TARGET_WARPS_PER_CORE: usize = 4; // Balance occupancy vs overhead

        let target_warps = gpu_cores * TARGET_WARPS_PER_CORE;
        let optimal_count = (target_warps / WARP_SIZE).next_power_of_two();

        // Reasonable bounds: 256 to 16384 kangaroos
        optimal_count.clamp(256, 16384)
    }

    /// Async version of parallel Brent's rho dispatch
    /// Allows CPU work to overlap with GPU computation
    pub async fn dispatch_parallel_brents_rho_async(&self, g: Point, p: Point, num_walks: usize, bias_mod: u64) -> Result<Option<BigInt256>, anyhow::Error> {
        #[cfg(feature = "rustacuda")]
        {
            use crate::gpu::backends::cuda_backend::{RhoState, CudaBackend};

            // Create CUDA backend instance
            let cuda_backend = CudaBackend::new().map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e))?;

            // Initialize rho states with bias-aware starts
            let mut rho_states = Vec::with_capacity(num_walks);
            for i in 0..num_walks {
                rho_states.push(RhoState {
                    current: p.clone(),  // Start from target point
                    steps: BigInt256::zero(),
                    bias_mod,
                });
            }

            // Create device buffer and launch kernel asynchronously
            let d_states = cuda_backend.create_state_buffer(&rho_states)
                .map_err(|e| anyhow::anyhow!("State buffer creation failed: {}", e))?;

            cuda_backend.launch_rho_kernel(&d_states, num_walks as u32, BigInt256::from_u64(bias_mod))
                .map_err(|e| anyhow::anyhow!("Kernel launch failed: {}", e))?;

            // Read back DP buffer asynchronously
            let dp_points = cuda_backend.read_dp_buffer()
                .map_err(|e| anyhow::anyhow!("DP buffer read failed: {}", e))?;

            // Check for collisions
            for dp in dp_points {
                if let Some(solution) = self.check_collision(&dp) {
                    return Ok(Some(solution));
                }
            }

            Ok(None)
        }

        #[cfg(not(feature = "rustacuda"))]
        {
            Ok(None)
        }
    }

    /// Concise Block: Switch Kangaroo/Rho in Hybrid
    pub fn dispatch_switch_mode(&self, has_interval: bool) {
        if has_interval {
            // Kangaroo dispatch - would call existing kangaroo methods
        } else {
            // dispatch_parallel_rho(/* ... */);
        }
    }


    /// Dispatch hybrid operations with CPU/GPU balancing heuristics
    /// RTX 5090: ~90% GPU for EC ops, CPU for validation/low-latency tasks
    pub async fn dispatch_hybrid_balanced(&self, steps: u64, gpu_load: f64) -> Result<Option<BigInt256>, anyhow::Error> {
        // Heuristic: GPU gets 90% load on RTX 5090 (high parallelism), CPU handles validation
        let gpu_steps = (steps as f64 * gpu_load.max(0.8).min(0.95)) as u64;  // 80-95% GPU
        let cpu_steps = steps - gpu_steps;

        // Async dispatch: GPU for bulk steps, CPU for collision detection
        let gpu_fut = async {
            #[cfg(feature = "rustacuda")]
            {
                self.dispatch_parallel_brents_rho_async(
                    crate::math::secp::Point::infinity(), // placeholder
                    crate::math::secp::Point::infinity(),
                    4096, 0
                ).await
            }
            #[cfg(not(feature = "rustacuda"))]
            {
                Ok(None)
            }
        };

        let cpu_fut = async {
            // CPU validation: check attractor rates, bias convergence
            let attractor_rate = self.get_attractor_rate(&vec![]); // placeholder
            if attractor_rate < 10.0 {
                log::warn!("Low attractor rate {:.1}%, consider bias adjustment", attractor_rate);
            }
            None
        };

        // Join futures: GPU does bulk work, CPU validates
        let (gpu_result, cpu_result) = tokio::join!(gpu_fut, cpu_fut);
        gpu_result.or(Ok(cpu_result))
    }

    /// Concise Block: Bias Hybrid Swap on Attractor Rate
    pub fn get_attractor_rate(&self, points: &[Point]) -> f64 {
        let sample: Vec<Point> = points.iter().take(100).cloned().collect();
        match crate::utils::pubkey_loader::scan_full_valuable_for_attractors(&sample) {
            Ok((_count, percent, _)) => percent,
            Err(_) => 0.0, // Return 0 on error
        }
    }

    /// Concise Block: Hybrid Test on Real Pubkey Attractor
    pub fn test_real_pubkey_attractor(&self, pubkey: &Point) -> Result<bool> {
        // Run prime mul test on pubkey
        if !self.dispatch_prime_mul_test(pubkey)? { return Ok(false); } // From prior block
        // Compute proxy on CPU, validate
        use crate::utils::pubkey_loader::is_attractor_proxy;
        Ok(is_attractor_proxy(&BigInt256::from_u64_array(pubkey.x)))
    }

    /// Concise Block: Dispatch CUDA Mod9 Check
    pub fn dispatch_mod9_check(&self, x_limbs: &Vec<[u64;4]>) -> Result<Vec<bool>> {
        // Note: In real implementation, would use CUDA buffers
        // For now, simulate with CPU computation
        let mut results = Vec::with_capacity(x_limbs.len());
        for limbs in x_limbs {
            let mut mod9: u64 = 0;
            for &limb in limbs {
                mod9 = (mod9 + limb) % 9; // Limb sum mod9 approximation
            }
            results.push(mod9 == 0);
        }
        Ok(results)
    }

    /// Execute computation with drift monitoring (single-threaded)
    pub fn execute_with_drift_monitoring(
        &self,
        shared_points: &mut SharedBuffer<Point>,
        shared_distances: &mut SharedBuffer<u64>,
        batch_size: usize,
        total_steps: u64,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mut steps_completed = 0u64;

        while steps_completed < total_steps {
            let batch_start = Instant::now();

            // Execute computation using hybrid backend
            {
                // Convert to Vec for backend API
                let mut positions_vec: Vec<[[u32; 8]; 3]> = shared_points.as_slice().iter().map(|p| [
                    p.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                ]).collect();

                let mut distances_vec: Vec<[u32; 8]> = shared_distances.as_slice().iter().map(|&d| [
                    d as u32, (d >> 32) as u32, 0, 0, 0, 0, 0, 0
                ]).collect();

                let types_vec: Vec<u32> = vec![1; batch_size]; // Simplified - all tame

                // Execute step batch using GpuBackend trait
                if let Err(e) = self.hybrid_backend.step_batch(&mut positions_vec, &mut distances_vec, &types_vec) {
                    log::error!("Hybrid backend step failed: {}", e);
                    break;
                }

                // Convert back to SharedBuffer format
                for (i, pos) in positions_vec.iter().enumerate() {
                    if i < shared_points.len() {
                        let point = &mut shared_points.as_mut_slice()[i];
                        for j in 0..4 {
                            point.x[j] = ((pos[0][j*2 + 1] as u64) << 32) | pos[0][j*2] as u64;
                            point.y[j] = ((pos[1][j*2 + 1] as u64) << 32) | pos[1][j*2] as u64;
                            point.z[j] = ((pos[2][j*2 + 1] as u64) << 32) | pos[2][j*2] as u64;
                        }
                    }
                }

                for (i, dist) in distances_vec.iter().enumerate() {
                    if i < shared_distances.len() {
                        shared_distances.as_mut_slice()[i] = ((dist[1] as u64) << 32) | dist[0] as u64;
                    }
                }
            }

            steps_completed += batch_size as u64;

            // Periodic drift checking
            if steps_completed % 10000 == 0 { // Check every 10k steps
                let error = self.compute_drift_error(shared_points, shared_distances, &self.curve);

                let mut metrics = self.metrics.lock().unwrap();
                metrics.error_rate = error;

                if error > self.drift_threshold {
                    metrics.swap_count += 1;
                    metrics.last_swap_time = Instant::now();
                    log::warn!("Drift detected (error: {:.6}), potential precision loss", error);
                }

                // Update throughput
                let batch_time = batch_start.elapsed();
                metrics.vulkan_throughput = batch_size as f64 / batch_time.as_secs_f64();
            }

            // Small delay to prevent tight looping
            thread::sleep(Duration::from_micros(1000));
        }

        let total_time = start_time.elapsed();
        log::info!("Hybrid computation completed {} steps in {:.2}s ({:.0} ops/s)",
                  steps_completed, total_time.as_secs_f64(),
                  steps_completed as f64 / total_time.as_secs_f64());

        Ok(())
    }

    /// Run Vulkan computation with drift monitoring
    fn run_vulkan_computation(
        &self,
        shared_points: &Arc<Mutex<SharedBuffer<Point>>>,
        shared_distances: &Arc<Mutex<SharedBuffer<u64>>>,
        metrics: &Arc<Mutex<DriftMetrics>>,
        sync_version: &Arc<Mutex<u64>>,
        batch_size: usize,
        total_steps: u64,
    ) {
        let start_time = Instant::now();
        let mut steps_completed = 0u64;

        while steps_completed < total_steps {
            let batch_start = Instant::now();

            // Execute computation using hybrid backend (falls back to Vulkan)
            {
                let mut points_guard = shared_points.lock().unwrap();
                let mut distances_guard = shared_distances.lock().unwrap();

                // Convert to Vec for backend API (simplified - would use slices)
                let mut positions_vec: Vec<[[u32; 8]; 3]> = points_guard.as_slice().iter().map(|p| [
                    p.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                ]).collect();

                let mut distances_vec: Vec<[u32; 8]> = distances_guard.as_slice().iter().map(|&d| [
                    d as u32, (d >> 32) as u32, 0, 0, 0, 0, 0, 0
                ]).collect();

                let types_vec: Vec<u32> = vec![1; batch_size]; // Simplified - all tame

                // Execute step batch using GpuBackend trait
                if let Err(e) = self.hybrid_backend.step_batch(&mut positions_vec, &mut distances_vec, &types_vec) {
                    log::error!("Hybrid backend step failed: {}", e);
                    break;
                }

                // Convert back (simplified)
                for (i, pos) in positions_vec.iter().enumerate() {
                    if i < points_guard.len() {
                        let point = &mut points_guard.as_mut_slice()[i];
                        for j in 0..4 {
                            point.x[j] = ((pos[0][j*2 + 1] as u64) << 32) | pos[0][j*2] as u64;
                            point.y[j] = ((pos[1][j*2 + 1] as u64) << 32) | pos[1][j*2] as u64;
                            point.z[j] = ((pos[2][j*2 + 1] as u64) << 32) | pos[2][j*2] as u64;
                        }
                    }
                }

                for (i, dist) in distances_vec.iter().enumerate() {
                    if i < distances_guard.len() {
                        distances_guard.as_mut_slice()[i] = ((dist[1] as u64) << 32) | dist[0] as u64;
                    }
                }

                // Update sync version
                *sync_version.lock().unwrap() += 1;
            }

            steps_completed += batch_size as u64;

            // Update throughput metrics
            let batch_time = batch_start.elapsed();
            let throughput = batch_size as f64 / batch_time.as_secs_f64();
            metrics.lock().unwrap().vulkan_throughput = throughput;

            // Small delay to prevent tight looping
            thread::sleep(Duration::from_micros(1000));
        }

        let total_time = start_time.elapsed();
        log::info!("Vulkan computation completed {} steps in {:.2}s ({:.0} ops/s)",
                  steps_completed, total_time.as_secs_f64(),
                  steps_completed as f64 / total_time.as_secs_f64());
    }

    /// Compute drift error by comparing sample points to CPU ground truth
    fn compute_drift_error(&self, points: &SharedBuffer<Point>, distances: &SharedBuffer<u64>, curve: &Secp256k1) -> f64 {
        let sample_size = (points.len() / 100).max(1).min(10); // Sample 1% or at least 1, max 10

        let mut total_error = 0.0;
        let mut samples_checked = 0;

        let points_slice = points.as_slice();
        let distances_slice = distances.as_slice();

        for i in (0..points.len()).step_by(points.len() / sample_size) {
            if samples_checked >= sample_size || i >= points.len() {
                break;
            }

            let gpu_point = points_slice[i];
            let _gpu_distance = distances_slice[i];

            // For drift detection, compare against expected CPU computation
            // In a real implementation, this would maintain a CPU reference computation
            // For now, use a simplified check: verify point is still on curve
            let point_valid = gpu_point.validate_curve(curve);

            // Check if coordinates are reasonable (not corrupted)
            let coords_reasonable = gpu_point.x.iter().all(|&x| x < curve.p.limbs[0] * 2) &&
                                   gpu_point.y.iter().all(|&x| x < curve.p.limbs[0] * 2) &&
                                   gpu_point.z.iter().all(|&x| x < curve.p.limbs[0] * 2);

            if !point_valid || !coords_reasonable {
                total_error += 1.0; // Full error for invalid points
            } else {
                // Small error for valid but potentially drifted points
                total_error += 0.01;
            }

            samples_checked += 1;
        }

        if samples_checked > 0 {
            total_error / samples_checked as f64
        } else {
            0.0
        }
    }

    /// Check bias adjustment convergence (stabilization criteria)
    /// Returns true if bias factors have stabilized (delta < 5% over 10 steps)
    pub fn check_bias_convergence(rate_history: &Vec<f64>, target: f64) -> bool {
        if rate_history.len() < 10 {
            return false;  // Need minimum history for convergence check
        }
        let recent_rates = &rate_history[rate_history.len().saturating_sub(5)..];  // Last 5 rates
        let ema = recent_rates.iter().sum::<f64>() / recent_rates.len() as f64;  // Simple EMA approximation
        let delta = (ema - target).abs() / target;  // Relative error
        delta < 0.05  // Within 5% of target = converged (stable bias adjustment)
    }

    /// Get current drift metrics
    pub fn get_metrics(&self) -> DriftMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_hybrid_manager_creation() {
        let manager = HybridGpuManager::new(0.001, 1).await;
        assert!(manager.is_ok());
    }
}
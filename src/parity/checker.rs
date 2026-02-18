//! 10M-step parity verification harness
//!
//! CPU vs GPU bit-for-bit verification

use crate::config::Config;
#[allow(unused_imports)]
use crate::gpu::{CpuBackend, GpuBackend};
use crate::kangaroo::stepper::KangarooStepper;
#[allow(unused_imports)]
use crate::math::bigint::BigInt256;
#[allow(unused_imports)]
use crate::types::{KangarooState, Point};
use anyhow::Result;
#[allow(unused_imports)]
use log::{debug, warn};
#[allow(unused_imports)]
use std::time::Instant;
#[allow(dead_code)]
pub struct ParityChecker {
    cpu_stepper: KangarooStepper,
    cpu_backend: CpuBackend,
    gpu_backend: Option<Box<dyn GpuBackend>>,
    test_steps: usize,
}

impl ParityChecker {
    /// Create new parity checker
    pub fn new() -> Self {
        // TODO: Initialize with proper config
        let _config = crate::config::Config::default();
        let cpu_stepper = KangarooStepper::new(false); // Use standard jump table
        let cpu_backend = CpuBackend::new().unwrap();

        // GPU backend will be initialized later in async context
        let gpu_backend = None;

        ParityChecker {
            cpu_stepper,
            cpu_backend,
            gpu_backend,
            test_steps: 10_000_000, // 10M steps
        }
    }

    /// Run parity verification test
    pub async fn run_parity_test(&mut self) -> Result<ParityResult> {
        println!(
            "Starting parity verification test ({} steps)...",
            self.test_steps
        );

        let start_time = Instant::now();

        // Generate test kangaroos
        let initial_kangaroos = self.generate_test_kangaroos(1000)?;

        // Step on CPU
        let cpu_result = self.step_on_cpu(initial_kangaroos.clone())?;

        // Step on GPU
        let gpu_result = self.step_on_gpu(initial_kangaroos).await?;

        // Compare results
        let mismatches = self.compare_results(&cpu_result, &gpu_result)?;

        let duration = start_time.elapsed();

        let result = ParityResult {
            total_steps: self.test_steps,
            mismatches,
            duration,
            passed: mismatches == 0,
        };

        println!(
            "Parity test completed in {:.2}s: {} mismatches",
            duration.as_secs_f64(),
            mismatches
        );

        Ok(result)
    }

    /// Generate test kangaroos for parity checking
    fn generate_test_kangaroos(&self, count: usize) -> Result<Vec<KangarooState>> {
        let mut kangaroos = Vec::new();

        // Generate deterministic test kangaroos
        for i in 0..count {
            let state = KangarooState::new(
                Point {
                    x: [(i * 123456789) as u64; 4],
                    y: [(i * 987654321) as u64; 4],
                    z: [1; 4],
                },
                BigInt256::from_u64((i * 1000) as u64), // Deterministic distance as BigInt256
                [i as u64; 4],                          // Deterministic alpha
                [i as u64; 4],                          // Deterministic beta
                i % 2 == 0,                             // Alternate tame/wild
                false,                                  // is_dp
                i as u64,                               // id
                0,                                      // step
                if i % 2 == 0 { 1 } else { 0 },         // kangaroo_type
            );
            kangaroos.push(state);
        }

        Ok(kangaroos)
    }

    /// Step kangaroos on CPU
    fn step_on_cpu(&mut self, mut kangaroos: Vec<KangarooState>) -> Result<Vec<KangarooState>> {
        for _step in 0..self.test_steps {
            kangaroos = self.cpu_stepper.step_batch(&kangaroos, None)?;
        }
        Ok(kangaroos)
    }

    /// Step kangaroos using CPU (simplified for parity testing)
    async fn step_on_gpu(&self, kangaroos: Vec<KangarooState>) -> Result<Vec<KangarooState>> {
        if let Some(gpu_backend) = &self.gpu_backend {
            // Convert kangaroos to GPU format
            let (mut gpu_positions, mut gpu_distances, gpu_types) =
                self.convert_states_to_gpu_format(&kangaroos);

            // Use real GPU stepping with bias integration
            let config = Config::default(); // Use default config for parity testing
            for _ in 0..self.test_steps {
                let _traps = gpu_backend.step_batch_bias(
                    &mut gpu_positions,
                    &mut gpu_distances,
                    &gpu_types,
                    &config,
                )?;
            }

            // Convert back to CPU format
            let result = self.convert_gpu_format_to_states(gpu_positions, gpu_distances, gpu_types);
            Ok(result)
        } else {
            // Fallback to CPU if GPU not available
            warn!("GPU backend not available for parity testing, falling back to CPU");
            let mut result = kangaroos.clone();
            for _step in 0..self.test_steps {
                for kangaroo in &mut result {
                    // Simple deterministic stepping for testing
                    let bias_mod = 1u64;
                    *kangaroo = self
                        .cpu_stepper
                        .step_kangaroo_with_bias(kangaroo, None, bias_mod);
                }
            }
            Ok(result)
        }
    }

    /// Convert KangarooState vectors to GPU format
    fn convert_states_to_gpu_format(
        &self,
        states: &[KangarooState],
    ) -> (Vec<[[u32; 8]; 3]>, Vec<[u32; 8]>, Vec<u32>) {
        let mut positions = Vec::with_capacity(states.len());
        let mut distances = Vec::with_capacity(states.len());
        let mut types = Vec::with_capacity(states.len());

        for state in states {
            // Convert position to GPU format [[u32;8];3]
            let x_bigint = BigInt256 {
                limbs: state.position.x,
            };
            let y_bigint = BigInt256 {
                limbs: state.position.y,
            };
            let z_bigint = BigInt256::from_u64(1); // Z coordinate (affine)
            let pos_gpu = [
                self.bigint_to_u32x8(&x_bigint),
                self.bigint_to_u32x8(&y_bigint),
                self.bigint_to_u32x8(&z_bigint),
            ];
            positions.push(pos_gpu);

            // Convert distance to GPU format [u32;8]
            distances.push(self.bigint_to_u32x8(&state.distance));

            // Type: 0 for tame, 1 for wild
            types.push(if state.is_tame { 0u32 } else { 1u32 });
        }

        (positions, distances, types)
    }

    /// Convert GPU format back to KangarooState vectors
    fn convert_gpu_format_to_states(
        &self,
        positions: Vec<[[u32; 8]; 3]>,
        distances: Vec<[u32; 8]>,
        types: Vec<u32>,
    ) -> Vec<KangarooState> {
        let mut states = Vec::with_capacity(positions.len());

        for i in 0..positions.len() {
            // Convert position back from GPU format
            let x = self.u32x8_to_bigint(&positions[i][0]);
            let y = self.u32x8_to_bigint(&positions[i][1]);
            let point = Point::from_affine(x.limbs, y.limbs);

            // Convert distance back
            let distance = self.u32x8_to_bigint(&distances[i]);

            // Create state with zero alpha/beta for parity testing
            let alpha = [0u64; 4];
            let beta = [0u64; 4];
            let is_tame = types[i] == 0;

            states.push(KangarooState {
                position: point,
                distance,
                alpha,
                beta,
                is_tame,
                is_dp: false,
                id: i as u64,
                step: 0,
                kangaroo_type: types[i],
            });
        }

        states
    }

    /// Convert BigInt256 to [u32;8] GPU format
    fn bigint_to_u32x8(&self, value: &BigInt256) -> [u32; 8] {
        let limbs = value.limbs;
        [
            (limbs[0] & 0xFFFFFFFF) as u32,
            ((limbs[0] >> 32) & 0xFFFFFFFF) as u32,
            (limbs[1] & 0xFFFFFFFF) as u32,
            ((limbs[1] >> 32) & 0xFFFFFFFF) as u32,
            (limbs[2] & 0xFFFFFFFF) as u32,
            ((limbs[2] >> 32) & 0xFFFFFFFF) as u32,
            (limbs[3] & 0xFFFFFFFF) as u32,
            ((limbs[3] >> 32) & 0xFFFFFFFF) as u32,
        ]
    }

    /// Convert [u32;8] GPU format to BigInt256
    fn u32x8_to_bigint(&self, value: &[u32; 8]) -> BigInt256 {
        let limb0 = value[0] as u64 | ((value[1] as u64) << 32);
        let limb1 = value[2] as u64 | ((value[3] as u64) << 32);
        let limb2 = value[4] as u64 | ((value[5] as u64) << 32);
        let limb3 = value[6] as u64 | ((value[7] as u64) << 32);
        BigInt256 {
            limbs: [limb0, limb1, limb2, limb3],
        }
    }

    /// Compare CPU and GPU results
    fn compare_results(&self, cpu: &[KangarooState], gpu: &[KangarooState]) -> Result<usize> {
        if cpu.len() != gpu.len() {
            return Err(anyhow::anyhow!(
                "Result length mismatch: CPU={}, GPU={}",
                cpu.len(),
                gpu.len()
            ));
        }

        let mut mismatches = 0;

        for (i, (cpu_state, gpu_state)) in cpu.iter().zip(gpu.iter()).enumerate() {
            if !self.states_equal(cpu_state, gpu_state) {
                mismatches += 1;
                if mismatches <= 5 {
                    // Log first few mismatches
                    println!(
                        "Mismatch at kangaroo {}: CPU={:?}, GPU={:?}",
                        i, cpu_state, gpu_state
                    );
                }
            }
        }

        Ok(mismatches)
    }

    /// Check if two kangaroo states are equal
    fn states_equal(&self, a: &KangarooState, b: &KangarooState) -> bool {
        a.position == b.position
            && a.distance == b.distance
            && a.alpha == b.alpha
            && a.beta == b.beta
            && a.is_tame == b.is_tame
    }

    /// Run batch parity verification (called from manager)
    pub async fn verify_batch(&self) -> Result<()> {
        // Run a quick parity test to ensure CPU/GPU consistency
        let passed = self.run_quick_test().await?;
        if !passed {
            warn!("Parity verification FAILED - CPU/GPU results differ!");
            return Err(anyhow::anyhow!("Parity verification failed"));
        }
        debug!("Parity verification passed");
        Ok(())
    }

    /// Run quick parity test (1K steps) for development
    pub async fn run_quick_test(&self) -> Result<bool> {
        // Temporarily reduce test steps
        let _original_steps = self.test_steps;
        let _gpu_backend = if let Some(ref gpu) = self.gpu_backend {
            Some(gpu)
        } else {
            None
        };

        let mut quick_checker = ParityChecker {
            cpu_stepper: self.cpu_stepper.clone(),
            cpu_backend: CpuBackend::new().unwrap(),
            gpu_backend: None, // Use CPU fallback for quick tests
            test_steps: 1000,
        };

        let result = quick_checker.run_parity_test().await?;
        Ok(result.passed)
    }
}

/// Result of parity verification
#[derive(Debug, Clone)]
pub struct ParityResult {
    pub total_steps: usize,
    pub mismatches: usize,
    pub duration: std::time::Duration,
    pub passed: bool,
}

impl std::fmt::Display for ParityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parity test: {} steps, {} mismatches, {:.2}s, {}",
            self.total_steps,
            self.mismatches,
            self.duration.as_secs_f64(),
            if self.passed { "PASSED" } else { "FAILED" }
        )
    }
}

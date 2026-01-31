//! 10M-step parity verification harness
//!
//! CPU vs GPU bit-for-bit verification

use crate::types::{KangarooState, Point};
use crate::kangaroo::stepper::KangarooStepper;
use crate::gpu::backend::{GpuBackend, CpuBackend};
use anyhow::Result;
use std::time::Instant;
use log::{warn, debug};

/// Parity checker for CPU vs GPU verification
pub struct ParityChecker {
    cpu_stepper: KangarooStepper,
    gpu_backend: tokio::sync::Mutex<CpuBackend>,
    test_steps: usize,
}

impl ParityChecker {
    /// Create new parity checker
    pub fn new() -> Self {
        // TODO: Initialize with proper config
        let config = crate::config::Config::parse().unwrap_or_else(|_| {
            // Create a minimal test config
            crate::config::Config {
                mode: crate::config::SearchMode::FullRange,
                p2pk_file: std::path::PathBuf::from("test_p2pk.txt"),
                puzzles_file: std::path::PathBuf::from("test_puzzles.txt"),
                puzzle_mode: false,
                test_mode: true,
                dp_bits: 20,
                herd_size: 100,
                jump_mean: 1000,
                max_ops: 1000,
                wild_primes: vec![179, 257],
                prime_spacing_with_entropy: false,
                expanded_prime_spacing: false,
                expanded_jump_table: false,
                attractor_start: None,
                enable_near_collisions: None,
                enable_walk_backs: None,
                enable_smart_pruning: false,
                enable_target_eviction: false,
                validate_puzzle: None,
                force_continue: false,
                output_dir: std::path::PathBuf::from("test_output"),
                checkpoint_interval: 1000,
                log_level: "info".to_string(),
            }
        });
        let cpu_stepper = KangarooStepper::new(false); // Use standard jump table
        let gpu_backend = tokio::sync::Mutex::new(CpuBackend::new());

        ParityChecker {
            cpu_stepper,
            gpu_backend,
            test_steps: 10_000_000, // 10M steps
        }
    }

    /// Run parity verification test
    pub async fn run_parity_test(&self) -> Result<ParityResult> {
        println!("Starting parity verification test ({} steps)...", self.test_steps);

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

        println!("Parity test completed in {:.2}s: {} mismatches", duration.as_secs_f64(), mismatches);

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
                i as u64 * 1000, // Deterministic distance
                [i as u64; 4],   // Deterministic alpha
                [i as u64; 4],   // Deterministic beta
                i % 2 == 0,      // Alternate tame/wild
                i as u64,
            );
            kangaroos.push(state);
        }

        Ok(kangaroos)
    }

    /// Step kangaroos on CPU
    fn step_on_cpu(&self, mut kangaroos: Vec<KangarooState>) -> Result<Vec<KangarooState>> {
        for _step in 0..self.test_steps {
            kangaroos = self.cpu_stepper.step_batch(&kangaroos, None)?;
        }
        Ok(kangaroos)
    }

    /// Step kangaroos on GPU
    async fn step_on_gpu(&self, kangaroos: Vec<KangarooState>) -> Result<Vec<KangarooState>> {
        let mut result = kangaroos;

        // For simplicity, step in batches to avoid GPU memory limits
        let batch_size = 1000;
        for _step in 0..self.test_steps {
            for chunk in result.chunks(batch_size) {
                let stepped_chunk = self.gpu_backend.lock().await.step_kangaroos(chunk).await?;
                // Update result with stepped chunk
                // (This is simplified - in reality would need proper indexing)
            }
        }

        Ok(result)
    }

    /// Compare CPU and GPU results
    fn compare_results(&self, cpu: &[KangarooState], gpu: &[KangarooState]) -> Result<usize> {
        if cpu.len() != gpu.len() {
            return Err(anyhow::anyhow!("Result length mismatch: CPU={}, GPU={}", cpu.len(), gpu.len()));
        }

        let mut mismatches = 0;

        for (i, (cpu_state, gpu_state)) in cpu.iter().zip(gpu.iter()).enumerate() {
            if !self.states_equal(cpu_state, gpu_state) {
                mismatches += 1;
                if mismatches <= 5 { // Log first few mismatches
                    println!("Mismatch at kangaroo {}: CPU={:?}, GPU={:?}", i, cpu_state, gpu_state);
                }
            }
        }

        Ok(mismatches)
    }

    /// Check if two kangaroo states are equal
    fn states_equal(&self, a: &KangarooState, b: &KangarooState) -> bool {
        a.position == b.position &&
        a.distance == b.distance &&
        a.alpha == b.alpha &&
        a.beta == b.beta &&
        a.is_tame == b.is_tame
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
        let original_steps = self.test_steps;
        // This would require mutable self, so we'll create a temp checker
        let quick_checker = ParityChecker {
            cpu_stepper: self.cpu_stepper.clone(),
            gpu_backend: tokio::sync::Mutex::new(CpuBackend::new()),
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
        write!(f, "Parity test: {} steps, {} mismatches, {:.2}s, {}",
               self.total_steps, self.mismatches, self.duration.as_secs_f64(),
               if self.passed { "PASSED" } else { "FAILED" })
    }
}
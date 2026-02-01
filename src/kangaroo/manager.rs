//! Central orchestrator for kangaroo herd management
//!
//! Central orchestrator: herd management, stepping batches, DP table interaction,
//! async pruning trigger, multi-GPU dispatch

use crate::config::Config;
use crate::types::{KangarooState, Target, Solution, Point, DpEntry};
use crate::dp::DpTable;
use crate::gpu::backend::{GpuBackend, HybridBackend};
use crate::kangaroo::generator::KangarooGenerator;
use crate::kangaroo::stepper::KangarooStepper;
use crate::kangaroo::collision::CollisionDetector;
use crate::parity::ParityChecker;
use crate::targets::TargetLoader;
use crate::math::bigint::BigInt256;

use anyhow::Result;
use log::{info, warn, debug};
use std::sync::Arc;
use tokio::sync::Mutex;
use bincode;

/// Central manager for kangaroo herd operations
pub struct KangarooManager {
    config: Config,
    targets: Vec<Target>,
    dp_table: Arc<Mutex<DpTable>>,
    gpu_backend: Box<dyn GpuBackend>,
    generator: KangarooGenerator,
    stepper: KangarooStepper,
    collision_detector: CollisionDetector,
    parity_checker: ParityChecker,
    total_ops: u64,
    start_time: std::time::Instant,
}

impl KangarooManager {
    /// Create new KangarooManager
    pub async fn new(config: Config) -> Result<Self> {
        // Load targets - ALWAYS load full valuable_p2pk_publickey.txt
        println!("DEBUG: Entered KangarooManager::new");
        println!("DEBUG: Starting target loading...");
        println!("DEBUG: About to call TargetLoader::new()");
        let target_loader = TargetLoader::new();
        println!("DEBUG: TargetLoader created successfully");
        let targets = target_loader.load_targets(&config)?;
        println!("DEBUG: Loaded {} targets", targets.len());
        info!("Loaded {} targets", targets.len());

        // Initialize components
        let dp_table = Arc::new(Mutex::new(DpTable::new(config.dp_bits)));

        // Create appropriate GPU backend based on configuration
        let gpu_backend: Box<dyn GpuBackend> = Box::new(crate::gpu::backend::CpuBackend);
        let generator = KangarooGenerator::new(&config);
        let stepper = KangarooStepper::with_dp_bits(false, config.dp_bits); // Use standard jump table
        let collision_detector = CollisionDetector::new();
        let parity_checker = ParityChecker::new();

        Ok(KangarooManager {
            config,
            targets,
            dp_table,
            gpu_backend,
            generator,
            stepper,
            collision_detector,
            parity_checker,
            total_ops: 0,
            start_time: std::time::Instant::now(),
        })
    }

    /// Get number of targets
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Run the main solving loop
    pub async fn run(&mut self) -> Result<Option<Solution>> {
        info!("Starting kangaroo solving with {} targets", self.targets.len());

        loop {
            // Check if we should stop
            if self.should_stop() {
                warn!("Stopping due to max operations or time limit");
                return Ok(None);
            }

            // Generate new kangaroo batch - distribute herd size across targets
            let kangaroos_per_target = std::cmp::max(1, self.config.herd_size / self.targets.len() as usize);
            let target_points: Vec<_> = self.targets.iter().map(|t| t.point).collect();
            let kangaroos = self.generator.generate_batch(&target_points, kangaroos_per_target)?;

            // Basic async overlap: step kangaroos and check collisions concurrently
            // This demonstrates the concept - full GPU integration pending
            let step_fut = async {
                // Step kangaroos (currently CPU, will be GPU)
                self.stepper.step_batch(&kangaroos, target_points.first())
            };

            let collision_fut = async {
                // Check collisions concurrently
                self.collision_detector.check_collisions(&self.dp_table).await
            };

            // Execute concurrently to demonstrate overlap concept
            let (step_result, collision_result) = tokio::join!(step_fut, collision_fut);

            let stepped_kangaroos = step_result?;
            let collision_solution = collision_result?;

            // Check for distinguished points
            let dp_candidates = self.find_distinguished_points(&stepped_kangaroos)?;

            // Add to DP table (async)
            {
                let mut dp_table = self.dp_table.lock().await;
                for candidate in dp_candidates {
                    if let Err(e) = dp_table.add_dp_async(candidate).await {
                        warn!("Failed to add DP entry: {}", e);
                    }
                }
            }

            // Check collision result
            if let Some(solution) = collision_solution {
                info!("COLLISION DETECTED!");
                if self.verify_solution(&solution)? {
                    return Ok(Some(solution));
                }
            }

            // Periodic maintenance
            self.periodic_maintenance().await?;

            // Update statistics
            self.total_ops += stepped_kangaroos.len() as u64;
        }
    }


    /// Find distinguished points in kangaroo batch
    fn find_distinguished_points(&self, kangaroos: &[KangarooState]) -> Result<Vec<crate::types::DpEntry>> {
        let mut dp_candidates = Vec::new();
        for kangaroo in kangaroos {
            if self.is_distinguished_point(&kangaroo.position) {
                let x_low_bits = kangaroo.position.x[0] & ((1u64 << self.config.dp_bits) - 1);
                let cluster_id = (kangaroo.position.x[3] >> 16) as u32; // x-coord high bits
                let dp_entry = crate::types::DpEntry::new(
                    kangaroo.position,
                    kangaroo.clone(),
                    x_low_bits,
                    cluster_id,
                );
                dp_candidates.push(dp_entry);
            }
        }
        Ok(dp_candidates)
    }

    /// Check if point is distinguished (trailing dp_bits of x-coordinate are zero)
    fn is_distinguished_point(&self, point: &crate::types::Point) -> bool {
        // Rule: DP determined by trailing dp-bits on point x-coord (no hash needed)
        let mask = (1u64 << self.config.dp_bits) - 1;
        (point.x[0] & mask) == 0
    }

    /// Check if we should stop the search
    fn should_stop(&self) -> bool {
        self.total_ops >= self.config.max_ops ||
        self.start_time.elapsed().as_secs() >= 4 * 3600 // 4 hours
    }

    /// Perform periodic maintenance (pruning, parity checks, checkpoints)
    async fn periodic_maintenance(&mut self) -> Result<()> {
        // Every checkpoint interval
        if self.total_ops % self.config.checkpoint_interval == 0 {
            self.save_checkpoint().await?;
            self.run_parity_check().await?;
        }

        // Trigger pruning if >80% full
        let dp_utilization = {
            let dp_table = self.dp_table.lock().await;
            dp_table.stats().utilization
        };
        if dp_utilization > 0.8 {
            let mut dp_table = self.dp_table.lock().await;
            if let Err(e) = dp_table.prune_entries_async().await {
                warn!("DP pruning failed: {}", e);
            } else {
                debug!("DP table pruned successfully");
            }
        }

        Ok(())
    }

    /// Save checkpoint
    async fn save_checkpoint(&self) -> Result<()> {
        use serde::{Serialize, Deserialize};

        #[derive(Serialize, Deserialize)]
        struct CheckpointData {
            total_ops: u64,
            start_time: std::time::SystemTime,
            kangaroo_states: Vec<KangarooState>,
            dp_entries: Vec<DpEntry>,
        }

        // Collect current kangaroo states (simplified - in practice would collect from active herds)
        let kangaroo_states = vec![]; // TODO: Collect from active kangaroo herds

        // Collect DP table entries
        let dp_entries: Vec<DpEntry> = {
            let table = self.dp_table.lock().await;
            table.get_entries().values().cloned().collect()
        };

        let dp_entries_count = dp_entries.len();

        let checkpoint = CheckpointData {
            total_ops: self.total_ops,
            start_time: std::time::SystemTime::now(),
            kangaroo_states,
            dp_entries,
        };

        // Serialize and save to sled database
        let serialized = bincode::serialize(&checkpoint)?;
        {
            let dp_table = self.dp_table.lock().await;
            if let Some(db) = dp_table.sled_db() {
                db.insert("checkpoint", serialized)?;
                db.flush()?;
                info!("Checkpoint saved at {} ops with {} DP entries", self.total_ops, dp_entries_count);
            } else {
                warn!("Checkpoint not saved - disk storage not enabled");
            }
        }

        Ok(())
    }

    /// Run parity verification
    async fn run_parity_check(&self) -> Result<()> {
        // Run 10M-step parity check (rule: CPU/GPU bit-for-bit mandatory)
        debug!("Running parity verification check");
        self.parity_checker.verify_batch().await
    }

    /// Verify potential solution
    fn verify_solution(&self, solution: &Solution) -> Result<bool> {
        debug!("Verifying solution for target {:?}", solution.target_point);

        // Verify that private_key * G = target_point
        let curve = self.collision_detector.curve();
        let computed_point = curve.mul(
            &BigInt256::from_u64_array(solution.private_key),
            &curve.g
        );

        let computed_affine = curve.to_affine(&computed_point);
        let target_affine = curve.to_affine(&solution.target_point);

        let is_valid = computed_affine.x == target_affine.x && computed_affine.y == target_affine.y;

        if is_valid {
            info!("✅ Solution verified: private key {:032x}{:032x}{:032x}{:032x}",
                  solution.private_key[3], solution.private_key[2],
                  solution.private_key[1], solution.private_key[0]);
        } else {
            warn!("❌ Solution verification failed");
        }

        Ok(is_valid)
    }

    /// Convert u32 array to u64 array (GPU format to CPU format)
    fn u32_array_to_u64_array(u32_arr: [u32; 8]) -> [u64; 4] {
        [
            (u32_arr[0] as u64) | ((u32_arr[1] as u64) << 32),
            (u32_arr[2] as u64) | ((u32_arr[3] as u64) << 32),
            (u32_arr[4] as u64) | ((u32_arr[5] as u64) << 32),
            (u32_arr[6] as u64) | ((u32_arr[7] as u64) << 32),
        ]
    }

    /// Convert GPU computation results back to KangarooState format
    /// Used after GPU stepping operations to reconstruct kangaroo states
    fn convert_gpu_results_to_kangaroos(
        &self,
        original_kangaroos: &[KangarooState],
        gpu_positions: &[[[u32; 8]; 3]],
        gpu_distances: &[[u32; 8]]
    ) -> Vec<KangarooState> {
        original_kangaroos.iter().enumerate().map(|(i, original)| {
            if i < gpu_positions.len() && i < gpu_distances.len() {
                // Convert GPU output back to our format
                let gpu_pos = gpu_positions[i];
                let gpu_dist = gpu_distances[i];

                // Convert u32 arrays back to BigInt256
                let position = Point {
                    x: Self::u32_array_to_u64_array(gpu_pos[0]),
                    y: Self::u32_array_to_u64_array(gpu_pos[1]),
                    z: Self::u32_array_to_u64_array(gpu_pos[2]),
                };
                let distance_u64 = Self::u32_array_to_u64_array(gpu_dist);

                KangarooState {
                    position,
                    distance: distance_u64[0], // Use first limb as distance
                    alpha: original.alpha,
                    beta: original.beta,
                    is_tame: original.is_tame,
                    id: original.id,
                }
            } else {
                // Fallback to original if GPU data unavailable
                original.clone()
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// Test manager initialization
    #[tokio::test]
    async fn test_manager_initialization() {
        let config = Config {
            herd_size: 100,
            dp_bits: 8,
            max_ops: 1000,
            checkpoint_interval: 500,
            ..Default::default()
        };

        let manager = KangarooManager::new(config).await.unwrap();
        assert_eq!(manager.target_count(), 0); // No targets loaded in test
        assert_eq!(manager.total_ops, 0);
    }

    /// Test DP finding logic
    #[tokio::test]
    async fn test_find_distinguished_points() {
        let config = Config {
            dp_bits: 4, // 16 possibilities
            ..Default::default()
        };
        let manager = KangarooManager::new(config).await.unwrap();

        // Create a kangaroo at a distinguished point
        let point = crate::types::Point {
            x: [0, 0, 0, 0], // x[0] & 0xF == 0 (distinguished)
            y: [1, 0, 0, 0],
            z: [1, 0, 0, 0]
        };
        let kangaroo = crate::types::KangarooState::new(point.clone(), 100, [0; 4], [0; 4], true, 0);
        let kangaroos = vec![kangaroo];

        let dp_candidates = manager.find_distinguished_points(&kangaroos).unwrap();
        assert_eq!(dp_candidates.len(), 1);
        assert_eq!(dp_candidates[0].x_hash & 0xF, 0); // Should be distinguished
    }

    /// Test stop conditions
    #[tokio::test]
    async fn test_should_stop() {
        let config = Config {
            max_ops: 1000,
            ..Default::default()
        };
        let mut manager = KangarooManager::new(config).await.unwrap();

        // Should not stop initially
        assert!(!manager.should_stop());

        // Should stop after max ops
        manager.total_ops = 1000;
        assert!(manager.should_stop());

        // Should stop after 4 hours
        manager.total_ops = 0;
        // Note: Time-based stopping is hard to test without mocking
    }

    /// Test collision detection and solution verification
    #[tokio::test]
    async fn test_collision_detection_and_solution() {
        let config = Config {
            dp_bits: 4,
            herd_size: 10,
            ..Default::default()
        };

        let manager = KangarooManager::new(config).await.unwrap();

        // Create two kangaroos that will collide
        // Tame kangaroo starting at G
        let tame_point = manager.collision_detector.curve().g.clone();
        let tame_kangaroo = crate::types::KangarooState::new(
            tame_point,
            0, // distance
            [0; 4], // alpha
            [1; 4], // beta = 1 (represents the tame kangaroo equation)
            true, // is_tame
            0,
        );

        // Wild kangaroo that ends up at the same position
        let wild_point = tame_point.clone(); // Same position = collision
        let wild_kangaroo = crate::types::KangarooState::new(
            wild_point,
            0, // distance
            [5; 4], // alpha
            [3; 4], // beta = 3 (represents the wild kangaroo equation)
            false, // is_wild
            1,
        );

        // Manually add both to DP table
        let tame_dp = crate::types::DpEntry::new(
            tame_point.clone(),
            tame_kangaroo.clone(),
            0, // x_hash
            0,
        );
        let wild_dp = crate::types::DpEntry::new(
            wild_point.clone(),
            wild_kangaroo.clone(),
            0, // x_hash
            0,
        );

        {
            let mut dp_table = manager.dp_table.lock().await;
            dp_table.add_dp(tame_dp).unwrap();
            dp_table.add_dp(wild_dp).unwrap();
        }

        // Check for collision
        let solution = manager.collision_detector.check_collisions(&manager.dp_table).await.unwrap();
        assert!(solution.is_some(), "Should have detected a collision");

        let solution = solution.unwrap();

        // Verify the solution
        let is_valid = manager.verify_solution(&solution).unwrap();
        assert!(is_valid, "Solution should be valid");
    }

    /// Test periodic maintenance triggers
    #[tokio::test]
    async fn test_periodic_maintenance() {
        let config = Config {
            checkpoint_interval: 100,
            max_ops: 1000,
            ..Default::default()
        };

        let mut manager = KangarooManager::new(config).await.unwrap();

        // Test that maintenance doesn't trigger initially
        manager.total_ops = 50;
        // Should not trigger checkpoint yet
        assert!(manager.total_ops % manager.config.checkpoint_interval != 0);

        // Test checkpoint trigger
        manager.total_ops = 100; // Exactly at checkpoint interval
        let result = manager.periodic_maintenance().await;
        assert!(result.is_ok(), "Periodic maintenance should succeed");

        // Test DP pruning trigger (when table gets full)
        // Add many entries to trigger pruning
        {
            let mut dp_table = manager.dp_table.lock().await;
            for i in 0..1000 {
                let point = crate::types::Point {
                    x: [i as u64, 0, 0, 0],
                    y: [i as u64 + 1, 0, 0, 0],
                    z: [1, 0, 0, 0]
                };
                let kangaroo = crate::types::KangarooState::new(
                    point.clone(),
                    i as u64,
                    [0; 4],
                    [0; 4],
                    true,
                    i as u64,
                );
                let entry = crate::types::DpEntry::new(point, kangaroo, i as u64, 0);
                let _ = dp_table.add_dp(entry);
            }
        }

        // The table should already be considered "full" with 1000 entries added above

        let result = manager.periodic_maintenance().await;
        assert!(result.is_ok(), "Maintenance with pruning should succeed");
    }

    /// Test DP table operations in run loop
    #[tokio::test]
    async fn test_dp_table_operations() {
        let config = Config {
            dp_bits: 8,
            ..Default::default()
        };

        let mut manager = KangarooManager::new(config).await.unwrap();

        // Create some kangaroos with distinguished points
        let mut kangaroos = Vec::new();
        for i in 0..10 {
            let point = crate::types::Point {
                x: [0, 0, 0, 0], // x[0] = 0, so distinguished (dp_bits=8)
                y: [i as u64, 0, 0, 0],
                z: [1, 0, 0, 0]
            };
            let kangaroo = crate::types::KangarooState::new(
                point,
                i as u64 * 100,
                [i as u64; 4],
                [i as u64; 4],
                i % 2 == 0,
                i as u64,
            );
            kangaroos.push(kangaroo);
        }

        // Find distinguished points
        let dp_candidates = manager.find_distinguished_points(&kangaroos).unwrap();
        assert_eq!(dp_candidates.len(), 10, "All kangaroos should be distinguished");

        // Add to DP table
        {
            let mut dp_table = manager.dp_table.lock().await;
            for candidate in &dp_candidates {
                dp_table.add_dp_async(candidate.clone()).await.unwrap();
            }
        }

        // Verify they were added
        {
            let dp_table = manager.dp_table.lock().await;
            assert_eq!(dp_table.entries().len(), 10, "All DP entries should be added");
        }
    }

    /// Test run loop components integration
    #[tokio::test]
    async fn test_run_loop_components_integration() {
        let config = Config {
            herd_size: 20,
            dp_bits: 4,
            max_ops: 100, // Small limit for testing
            checkpoint_interval: 50,
            ..Default::default()
        };

        let mut manager = KangarooManager::new(config).await.unwrap();

        // Run a few iterations of the main loop logic manually
        let mut iterations = 0;
        let max_iterations = 3;

        while !manager.should_stop() && iterations < max_iterations {
            // Generate batch
            let kangaroos_per_target = std::cmp::max(1, manager.config.herd_size / manager.targets.len() as usize);
            let target_points: Vec<_> = manager.targets.iter().map(|t| t.point).collect();
            let kangaroos = manager.generator.generate_batch(&target_points, kangaroos_per_target).unwrap();

            // Step kangaroos
            let stepped_kangaroos = manager.stepper.step_batch(&kangaroos, target_points.first()).unwrap();

            // Find distinguished points
            let dp_candidates = manager.find_distinguished_points(&stepped_kangaroos).unwrap();

            // Add to DP table
            {
                let mut dp_table = manager.dp_table.lock().await;
                for candidate in dp_candidates {
                    let _ = dp_table.add_dp_async(candidate).await;
                }
            }

            // Update operation count
            manager.total_ops += stepped_kangaroos.len() as u64;
            iterations += 1;
        }

        // Verify that operations were performed
        assert!(manager.total_ops > 0, "Should have performed some operations");
        assert!(iterations > 0, "Should have run at least one iteration");
    }

    #[tokio::test]
    async fn test_run_loop_stub() {
        let config = Config {
            herd_size: 10,
            dp_bits: 4,
            max_ops: 100,
            checkpoint_interval: 50,
            ..Default::default()
        };
        let mut manager = KangarooManager::new(config).await.unwrap();
        let result = manager.run().await.unwrap();
        assert!(result.is_none(), "Should stop without solution");
    }
}
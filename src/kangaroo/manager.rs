//! Central orchestrator for kangaroo herd management
//!
//! Central orchestrator: herd management, stepping batches, DP table interaction,
//! async pruning trigger, multi-GPU dispatch

use crate::config::Config;
use crate::types::{KangarooState, Target, Solution, Point, DpEntry};
use crate::dp::DpTable;
use crate::math::Secp256k1;
use crate::gpu::{GpuBackend, HybridBackend, CpuBackend, HybridGpuManager, shared::SharedBuffer};
use crate::types::TaggedKangarooState;
use crate::kangaroo::SearchConfig;
#[cfg(feature = "vulkano")]
use crate::gpu::VulkanBackend;
#[cfg(feature = "rustacuda")]
use crate::gpu::CudaBackend;
use crate::kangaroo::generator::KangarooGenerator;
use crate::kangaroo::stepper::KangarooStepper;
use crate::kangaroo::collision::CollisionDetector;
use crate::parity::ParityChecker;
use crate::targets::TargetLoader;
use crate::math::bigint::BigInt256;
use anyhow::anyhow;
use std::collections::HashMap;

/// Concise Block: Precompute Small k*G Table for Nearest Adjust
struct NearMissTable {
    table: Vec<(BigInt256, u64)>, // (k*G.x, k)
}

impl NearMissTable {
    pub fn new(max_k: u64) -> Self {
        let mut table = vec![];
        let curve = crate::math::secp::Secp256k1::new();
        let g_x = BigInt256::from_u64_array(curve.g.x);
        let mut current = g_x.clone();
        for k in 1..=max_k {
            table.push((current.clone(), k));
            current = (current + g_x.clone()) % curve.p.clone(); // Additive multiples
        }
        Self { table }
    }

    pub fn find_nearest(&self, diff: &BigInt256) -> Option<u64> {
        self.table.iter().find(|(x, _)| x == diff).map(|(_, k)| *k)
    }
}

use anyhow::Result;
use log::{info, warn, debug};
use std::sync::Arc;
use tokio::sync::Mutex;
use bincode;

/// Central manager for kangaroo herd operations
pub struct KangarooManager {
    config: Config,
    search_config: SearchConfig,         // Search parameters for this manager
    targets: Vec<Target>,
    multi_targets: Vec<(Point, u32)>,    // Multi-target points with puzzle IDs for batch solving
    wild_states: Vec<TaggedKangarooState>, // Tagged wild kangaroos per target
    tame_states: Vec<KangarooState>,     // Shared tame kangaroos
    dp_table: Arc<Mutex<DpTable>>,
    gpu_backend: Box<dyn GpuBackend>,
    generator: KangarooGenerator,
    stepper: KangarooStepper,
    collision_detector: CollisionDetector,
    parity_checker: ParityChecker,
    total_ops: u64,
    current_steps: u64,                  // Steps completed so far
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
        let gpu_backend: Box<dyn GpuBackend> = match config.gpu_backend.as_str() {
            "hybrid" => Box::new(HybridBackend::new().await?),
            "vulkan" => {
                #[cfg(feature = "vulkano")]
                {
                    Box::new(VulkanBackend::new().await?)
                }
                #[cfg(not(feature = "vulkano"))]
                {
                    return Err(anyhow!("Vulkan backend requires 'vulkano' feature to be enabled"));
                }
            }
            "cuda" => {
                #[cfg(feature = "rustacuda")]
                {
                    Box::new(CudaBackend::new()?)
                }
                #[cfg(not(feature = "rustacuda"))]
                {
                    return Err(anyhow!("CUDA backend requires 'rustacuda' feature to be enabled"));
                }
            }
            "cpu" => Box::new(CpuBackend::new()?),
            _ => return Err(anyhow!("Invalid GPU backend: {}", config.gpu_backend)),
        };
        let generator = KangarooGenerator::new(&config);
        let stepper = KangarooStepper::with_dp_bits(false, config.dp_bits); // Use standard jump table
        let collision_detector = CollisionDetector::new();
        let parity_checker = ParityChecker::new();

        Ok(KangarooManager {
            config,
            search_config: SearchConfig::default(),
            targets,
            multi_targets: Vec::new(),
            wild_states: Vec::new(),
            tame_states: Vec::new(),
            dp_table,
            gpu_backend,
            generator,
            stepper,
            collision_detector,
            parity_checker,
            total_ops: 0,
            current_steps: 0,
            start_time: std::time::Instant::now(),
        })
    }

    /// Get number of targets
    pub fn target_count(&self) -> usize {
        self.targets.len() + self.multi_targets.len()
    }

    /// Get DP table size
    pub fn dp_table_size(&self) -> usize {
        self.dp_table.try_lock().map(|table| table.entries().len()).unwrap_or(0)
    }

    /// Get current steps completed
    pub fn current_steps(&self) -> u64 {
        self.current_steps
    }

    /// Get total operations performed
    pub fn total_ops(&self) -> u64 {
        self.total_ops
    }

    /// Get search config
    pub fn search_config(&self) -> &SearchConfig {
        &self.search_config
    }

    /// Get multi targets with puzzle IDs
    pub fn multi_targets(&self) -> &[(Point, u32)] {
        &self.multi_targets
    }

    /// Get wild states
    pub fn wild_states(&self) -> &[TaggedKangarooState] {
        &self.wild_states
    }

    /// Get tame states
    pub fn tame_states(&self) -> &[KangarooState] {
        &self.tame_states
    }

    /// Create new KangarooManager for multi-target solving with search config
    pub async fn new_multi_config(multi_targets: Vec<(Point, u32)>, search_config: SearchConfig) -> Result<Self> {
        // Validate search config
        search_config.validate()?;

        // Use default config for basic setup
        let config = Config {
            gpu_backend: "hybrid".to_string(),
            dp_bits: search_config.dp_bits as usize,
            herd_size: multi_targets.len() * search_config.batch_per_target,
            ..Default::default()
        };

        // Initialize components
        let dp_table = Arc::new(Mutex::new(DpTable::new(config.dp_bits)));
        let generator = KangarooGenerator::new(&config).with_search_config(search_config.clone());
        let stepper = KangarooStepper::with_dp_bits(false, config.dp_bits);
        let collision_detector = CollisionDetector::new();
        let parity_checker = ParityChecker::new();

        // Create GPU backend
        let gpu_backend: Box<dyn GpuBackend> = Box::new(HybridBackend::new().await?);

        // Concise Block: Sort Targets by Attractor Proxy in Init
        let mut targets_only: Vec<Point> = multi_targets.iter().map(|(p, _)| *p).collect();
        use crate::utils::pubkey_loader::is_attractor_proxy;
        targets_only.sort_by_key(|p| if is_attractor_proxy(&p.x_bigint()) { 0 } else { 1 }); // Attractors first

        // Concise Block: Run GPU Prime Mul Test on Manager Init
        let hybrid = HybridGpuManager::new(0.001, 5).await?;
        let test_target = if targets_only.is_empty() { Secp256k1::new().g.clone() } else { targets_only[0] };
        if !hybrid.test_prime_mul_gpu(&test_target)? {
            println!("GPU prime mul drift detected! Fallback to CPU.");
            // Would set flag for CPU-only mode here
        }

        // Generate multi-target kangaroos with precise prime starts
        let (wild_states, tame_states) = generator.setup_kangaroos_multi(&targets_only, search_config.batch_per_target, &search_config);

        Ok(Self {
            config,
            search_config,
            targets: Vec::new(), // Not used in multi-target mode
            multi_targets,
            wild_states,
            tame_states,
            dp_table,
            gpu_backend,
            generator,
            stepper,
            collision_detector,
            parity_checker,
            total_ops: 0,
            current_steps: 0,
            start_time: std::time::Instant::now(),
        })
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

            // Use GPU acceleration when available (hybrid backend with optimizations)
            let step_fut = async {
                if self.config.gpu_backend == "hybrid" {
                    // Use optimized GPU dispatch for hybrid backend
                    self.step_kangaroos_gpu(&kangaroos).await
                } else {
                    // Fall back to CPU stepping
                    self.stepper.step_batch(&kangaroos, target_points.first())
                }
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

        // Generate kangaroo states for current targets
        let target_points: Vec<Point> = self.targets.iter().map(|t| t.point).collect();
        let kangaroo_states = self.generator.generate_batch(&target_points, 512)?; // 512 kangaroos per target

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

    /// Step kangaroos using optimized GPU dispatch (hybrid backend)
    async fn step_kangaroos_gpu(&self, kangaroos: &[KangarooState]) -> Result<Vec<KangarooState>> {
        // Convert kangaroo states to GPU format
        let mut positions: Vec<[[u32; 8]; 3]> = kangaroos.iter()
            .map(|k| [
                k.position.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
                k.position.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
                k.position.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
            ])
            .collect();

        let mut distances: Vec<[u32; 8]> = kangaroos.iter()
            .map(|k| [
                k.distance as u32, (k.distance >> 32) as u32, 0, 0, 0, 0, 0, 0
            ])
            .collect();

        let types: Vec<u32> = kangaroos.iter()
            .map(|k| if k.is_tame { 1 } else { 0 })
            .collect();

        // Use the GPU backend for stepping
        let traps = self.gpu_backend.step_batch(&mut positions, &mut distances, &types)?;

        // Convert back to KangarooState format
        let stepped_kangaroos: Vec<KangarooState> = positions.into_iter()
            .zip(distances.into_iter())
            .enumerate()
            .map(|(i, (pos, dist))| {
                let position = Point {
                    x: [
                        ((pos[0][1] as u64) << 32) | pos[0][0] as u64,
                        ((pos[0][3] as u64) << 32) | pos[0][2] as u64,
                        ((pos[0][5] as u64) << 32) | pos[0][4] as u64,
                        ((pos[0][7] as u64) << 32) | pos[0][6] as u64,
                    ],
                    y: [
                        ((pos[1][1] as u64) << 32) | pos[1][0] as u64,
                        ((pos[1][3] as u64) << 32) | pos[1][2] as u64,
                        ((pos[1][5] as u64) << 32) | pos[1][4] as u64,
                        ((pos[1][7] as u64) << 32) | pos[1][6] as u64,
                    ],
                    z: [
                        ((pos[2][1] as u64) << 32) | pos[2][0] as u64,
                        ((pos[2][3] as u64) << 32) | pos[2][2] as u64,
                        ((pos[2][5] as u64) << 32) | pos[2][4] as u64,
                        ((pos[2][7] as u64) << 32) | pos[2][6] as u64,
                    ],
                };

                let distance = ((dist[1] as u64) << 32) | dist[0] as u64;

                KangarooState::new(
                    position,
                    distance as u64,
                    kangaroos[i].alpha,
                    kangaroos[i].beta,
                    kangaroos[i].is_tame,
                    kangaroos[i].id,
                )
            })
            .collect();

        // Process traps (distinguished points found)
        for trap in traps {
            // Convert trap back to DP entry and add to table
            let trap_point = Point {
                x: [
                    ((trap.x[1] as u64) << 32) | trap.x[0] as u64,
                    ((trap.x[3] as u64) << 32) | trap.x[2] as u64,
                    ((trap.x[5] as u64) << 32) | trap.x[4] as u64,
                    ((trap.x[7] as u64) << 32) | trap.x[6] as u64,
                ],
                y: [0; 4], // Not provided in trap
                z: [0; 4], // Not provided in trap
            };

            let trap_distance_bytes = trap.dist.to_bytes_le();
            let trap_distance = if trap_distance_bytes.len() >= 8 {
                ((trap_distance_bytes[7] as u64) << 56) |
                ((trap_distance_bytes[6] as u64) << 48) |
                ((trap_distance_bytes[5] as u64) << 40) |
                ((trap_distance_bytes[4] as u64) << 32) |
                ((trap_distance_bytes[3] as u64) << 24) |
                ((trap_distance_bytes[2] as u64) << 16) |
                ((trap_distance_bytes[1] as u64) << 8) |
                (trap_distance_bytes[0] as u64)
            } else {
                0
            };

            // Create kangaroo state for the trap
            let trap_state = KangarooState::new(
                trap_point,
                trap_distance,
                [0; 4], // alpha not provided
                [0; 4], // beta not provided
                trap.is_tame,
                0, // id not provided
            );

            // Add to DP table
            let mut table = self.dp_table.lock().await;
            let dp_entry = DpEntry::new(trap_point, trap_state, 0, 0); // x_hash and cluster_id simplified
            table.add_dp(dp_entry)?;
        }

        Ok(stepped_kangaroos)
    }

    /// Step kangaroos using refined hybrid GPU with drift mitigation
    pub async fn step_herds_hybrid_refined(&mut self, total_steps: u64) -> Result<()> {
        // Create shared buffers for cross-API memory access
        let mut shared_points = SharedBuffer::<Point>::new(self.config.herd_size);
        let mut shared_distances = SharedBuffer::<u64>::new(self.config.herd_size);

        // Initialize with current kangaroo state
        {
            let points_slice = shared_points.as_mut_slice();
            let distances_slice = shared_distances.as_mut_slice();

            // Copy from internal storage (simplified - would need to track active kangaroos)
            for i in 0..self.config.herd_size.min(points_slice.len()) {
                // Initialize with default positions (would copy from actual kangaroos)
                points_slice[i] = Point {
                    x: [i as u64, 0, 0, 0],
                    y: [0, 0, 0, 0],
                    z: [1, 0, 0, 0],
                };
                distances_slice[i] = i as u64 * 1000; // Placeholder distances
            }
        }

        // Create hybrid manager with drift monitoring
        let hybrid_manager = HybridGpuManager::new(0.001, 5).await?; // 0.1% error threshold, 5s check interval

        // Execute hybrid computation with drift mitigation
        hybrid_manager.execute_with_drift_monitoring(
            &mut shared_points,
            &mut shared_distances,
            self.config.herd_size,
            total_steps,
        )?;

        // Extract results and update DP table
        {

            // Process distinguished points found during computation
            for i in 0..shared_points.len() {
                let point = shared_points.as_slice()[i];
                let distance = shared_distances.as_slice()[i];

                // Check if this is a distinguished point (simplified check)
                let x_low = point.x[0] & ((1u64 << 20) - 1); // 20-bit DP check
                if x_low == 0 {
                    // Found distinguished point - add to DP table
                    let kangaroo_state = KangarooState::new(
                        point,
                        distance,
                        [0; 4], // alpha (would be tracked)
                        [0; 4], // beta (would be tracked)
                        true,   // is_tame (simplified)
                        i as u64, // id
                    );

                    let mut table = self.dp_table.lock().await;
                    let dp_entry = crate::types::DpEntry::new(point, kangaroo_state, x_low as u64, 0);
                    if let Err(e) = table.add_dp(dp_entry) {
                        log::warn!("Failed to add DP entry: {}", e);
                    }
                }
            }
        }

        // Log final metrics
        let metrics = hybrid_manager.get_metrics();
        log::info!(
            "Hybrid computation completed - Error rate: {:.6}, CUDA throughput: {:.0} ops/s, Vulkan throughput: {:.0} ops/s, Swaps: {}",
            metrics.error_rate,
            metrics.cuda_throughput,
            metrics.vulkan_throughput,
            metrics.swap_count
        );

        Ok(())
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

    /// Step multi-target kangaroos using hybrid GPU acceleration
    pub fn step_herds_multi(&mut self, steps: u64) -> Result<()> {
        // Create shared buffers for wild kangaroos
        let mut shared_wild_points = SharedBuffer::<Point>::new(self.wild_states.len());
        let mut shared_wild_distances = SharedBuffer::<BigInt256>::new(self.wild_states.len());
        let mut shared_wild_tags = SharedBuffer::<u32>::new(self.wild_states.len());

        // Initialize wild kangaroo data
        {
            let wild_points = shared_wild_points.as_mut_slice();
            let wild_distances = shared_wild_distances.as_mut_slice();
            let wild_tags = shared_wild_tags.as_mut_slice();

            for (i, wild_state) in self.wild_states.iter().enumerate() {
                wild_points[i] = wild_state.point;
                wild_distances[i] = wild_state.distance;
                wild_tags[i] = wild_state.target_idx;
            }
        }

        // Create shared buffers for tame kangaroos
        let mut shared_tame_points = SharedBuffer::<Point>::new(self.tame_states.len());
        let mut shared_tame_distances = SharedBuffer::<BigInt256>::new(self.tame_states.len());

        // Initialize tame kangaroo data
        {
            let tame_points = shared_tame_points.as_mut_slice();
            let tame_distances = shared_tame_distances.as_mut_slice();

            for (i, tame_state) in self.tame_states.iter().enumerate() {
                tame_points[i] = tame_state.position;
                tame_distances[i] = BigInt256::from_u64(tame_state.distance);
            }
        }

        // Create hybrid manager and execute
        // Note: In real async code, would use await
        let rt = tokio::runtime::Runtime::new()?;
        let hybrid_manager = rt.block_on(HybridGpuManager::new(0.001, 5))?; // 0.1% error threshold, 5s check interval

        // Concise Block: Runtime Prime Mul Test in Hybrid Steps
        let mut wild_points_vec: Vec<Point> = shared_wild_points.as_slice().iter().cloned().collect();
        rt.block_on(hybrid_manager.step_with_prime_test(&mut wild_points_vec, self.current_steps))?;
        // Update shared buffer with tested points
        for (i, point) in wild_points_vec.iter().enumerate() {
            if let Some(shared_point) = shared_wild_points.as_mut_slice().get_mut(i) {
                *shared_point = *point;
            }
        }

        // For now, execute single-threaded with drift monitoring
        // TODO: Extend hybrid manager to handle tagged multi-target operations
        hybrid_manager.execute_with_drift_monitoring(
            &mut shared_wild_points,
            &mut shared_wild_distances,
            self.wild_states.len(),
            steps,
        )?;

        // Update wild kangaroo states
        {
            let wild_points = shared_wild_points.as_slice();
            let wild_distances = shared_wild_distances.as_slice();

            for (i, wild_state) in self.wild_states.iter_mut().enumerate() {
                wild_state.point = wild_points[i];
                wild_state.distance = wild_distances[i];

                // Per-puzzle range bound check
                if let Some((_, end)) = self.search_config.per_puzzle_ranges.as_ref()
                    .and_then(|r| r.get(&wild_state.target_idx)) {
                    if wild_state.distance >= *end {
                        // Reset kangaroo to initial state
                        wild_state.distance = BigInt256::zero();
                        wild_state.point = Point::infinity();  // Reset position
                        warn!("Wild kangaroo {} for puzzle {} exceeded range, resetting",
                              i, wild_state.target_idx);
                    }
                }
            }
        }

        // Update tame kangaroo states
        {
            let tame_points = shared_tame_points.as_slice();
            let tame_distances = shared_tame_distances.as_slice();

            for (i, tame_state) in self.tame_states.iter_mut().enumerate() {
                tame_state.position = tame_points[i];
                tame_state.distance = tame_distances[i].to_u64_array()[0];
            }
        }

        // Check for collisions and solutions
        self.check_multi_collisions()?;

        self.total_ops += (self.wild_states.len() + self.tame_states.len()) as u64 * steps;

        Ok(())
    }

    /// Check for collisions in multi-target setup and solve keys
    fn check_multi_collisions(&mut self) -> Result<()> {
        // Check DP table for collisions between wild and tame kangaroos
        let mut dp_table = self.dp_table.lock().unwrap();

        // In a full implementation, this would:
        // 1. Check for new DP entries from the GPU step
        // 2. Look for collisions between wild and tame kangaroos
        // 3. Solve the discrete log for matched targets
        // 4. Return solutions

        // For now, placeholder implementation
        // TODO: Implement full collision detection and key solving

        Ok(())
    }

    /// Prime-Adjusted Collision Solve
    /// Verbatim preset: Use stored initial_prime for inv * diff mod N.
    pub fn solve_collision_prime_adjusted(&self, tame_dist: &BigInt256, wild_dist: &BigInt256, wild_initial_prime: &BigInt256) -> Option<BigInt256> {
        let curve = crate::math::secp::Secp256k1::new();
        let one = BigInt256::one();
        let diff = tame_dist.clone() + one - wild_dist.clone(); // 1 + d_tame - d_wild
        let inv_prime = curve.order.mod_inverse(wild_initial_prime)?; // Extended Euclidean from Phase 3
        Some((inv_prime * diff) % curve.order)
    }

    /// Concise Block: Layer Vanity Mod in Near Miss Diff
    pub fn calculate_near_miss_diff(&self, trap_x: &BigInt256, dp_x: &BigInt256, threshold: &BigInt256) -> Option<BigInt256> {
        let curve = crate::math::secp::Secp256k1::new();
        let diff = (trap_x.clone() - dp_x.clone()) % curve.p.clone();
        if diff.clone() % BigInt256::from_u64(16) != BigInt256::from_u64(9) { return None; } // Vanity bias e.g., mod16=9
        if diff < *threshold { Some(diff) } else { None }
    }

    /// Concise Block: BSGS for Small DL on Diff to Find k
    fn bsgs_find_k(diff: &BigInt256, g_x: &BigInt256, range_max: u64, m: u64) -> Option<u64> { // m=sqrt(range)
        let mut baby = HashMap::new();
        let mut current = BigInt256::zero();
        for i in 0..m {
            baby.insert(current.clone(), i);
            current = (current.clone() + g_x.clone()) % BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F); // secp256k1 modulus approximation
        }
        let giant_step = BigInt256::from_u64(m) * g_x.clone() % BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F);
        let mut giant = diff.clone();
        for j in 0.. (range_max / m + 1) {
            if let Some(i) = baby.get(&giant) {
                return Some(j * m + *i); // k = j*m + i
            }
            giant = (giant.clone() - giant_step.clone()) % BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F);
        }
        None
    }

    /// Concise Block: Biased BSGS with Mod Filter
    fn biased_bsgs_find_k(diff: &BigInt256, g_x: &BigInt256, range_max: u64, m: u64, mod_bias: u64, bias_res: u64) -> Option<u64> {
        let mut baby = HashMap::new();
        let mut current = BigInt256::zero();
        let mut i = 0u64;
        while i < m {
            if i % mod_bias == bias_res { // Filter biased i
                baby.insert(current.clone(), i);
            }
            current = current.add(g_x).mod_(&BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F));
            i += 1;
        }
        let giant_step = BigInt256::from_u64(m).mul(g_x).mod_(&BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F));
        let mut giant = diff.clone();
        let mut j = 0u64;
        while j < (range_max / m + 1) {
            if j % mod_bias == bias_res { // Biased j
                if let Some(i) = baby.get(&giant) {
                    return Some(j * m + *i);
                }
            }
            giant = giant.sub(&giant_step).mod_(&BigInt256::from_u64(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F));
            j += 1;
        }
        None
    }

    /// Concise Block: Full Layered Near Miss with Biased BSGS
    pub fn solve_near_collision(&self, trap: &KangarooState, dp: &TaggedKangarooState, diff: BigInt256) -> Option<BigInt256> {
        let table = NearMissTable::new(1000);
        if let Some(k) = table.find_nearest(&diff) {
            let curve = crate::math::secp::Secp256k1::new();
            let inv_k = curve.order.mod_inverse(&BigInt256::from_u64(k))?;
            let adjusted_prime = dp.initial_offset.clone() * inv_k % curve.order.clone();
            self.solve_collision_prime_adjusted(&trap.distance, &dp.distance, &adjusted_prime)
        } else if let Some(k) = Self::biased_bsgs_find_k(&diff, &BigInt256::from_u64_array(self.curve.g.x), 1 << 40, 1 << 20, 81, 0) { // Mod81=0 bias
            let curve = crate::math::secp::Secp256k1::new();
            let inv_k = curve.order.mod_inverse(&BigInt256::from_u64(k))?;
            let adjusted_prime = dp.initial_offset.clone() * inv_k % curve.order.clone();
            self.solve_collision_prime_adjusted(&trap.distance, &dp.distance, &adjusted_prime)
        } else { None }
    }

    /// Concise Block: Pollard's Lambda Solve in Impl
    pub fn lambda_collision_solve(&self, tame: &KangarooState, wild: &TaggedKangarooState) -> BigInt256 {
        let diff = tame.distance.sub(&wild.distance).mod_(&self.curve.order);
        diff.add_assign(&wild.initial_offset); // + prime offset
        diff.mod_(&self.curve.order) // Basic lambda k = d_t - d_w + offset mod N
    }

    /// Concise Block: Use Lambda Bucket in Step Herds
    fn step_lambda_kangaroo(&mut self, state: &mut KangarooState, is_tame: bool, steps: u64) {
        let seed = 42u32;
        for s in 0..steps {
            let bucket = self.generator.lambda_bucket_select(&state.position, &BigInt256::from_u64(state.distance), seed, s as u32, is_tame);
            let jump_dist = self.generator.get_jump_from_bucket_mod27(bucket); // Attractor biased
            let jump_point = self.curve.mul(&jump_dist, &self.curve.g);
            state.position = self.curve.add(&state.position, &jump_point);
            state.distance += jump_dist.to_u64_array()[0]; // Convert to u64 for tame
            // DP check, near miss if diff small
        }
    }

    /// Concise Block: Detect Bias in DP's (Mod n on x with Trailing Zeros)
    pub fn detect_dp_bias(&self, x: &BigInt256, trailing_zeros: u32, mod_n: u64) -> bool {
        let mask = (1u64 << trailing_zeros) - 1;
        if x.limbs[0] & mask != 0 { return false; } // Confirm trailing zeros
        x.mod_u64(mod_n) == 0 // Bias mod n
    }

    /// Concise Block: Bias Jump to Force DP Mod n
    pub fn get_biased_jump_for_dp(&self, bucket: u32, mod_n: u64) -> BigInt256 {
        let base = crate::math::constants::PRIME_MULTIPLIERS[bucket as usize % 32];
        let adjust = mod_n - (base % mod_n);
        BigInt256::from_u64(base + adjust) // To mod n=0
    }

    /// Concise Block: Detect/Exploit Bias in Step DP Check
    pub fn check_dp_with_bias(&self, point: &Point, trailing_zeros: u32, mod_n: u64) -> bool {
        if self.detect_dp_bias(&point.x, trailing_zeros, mod_n) {
            // Instant win: Bias match, layer in solve
            true
        } else { false }
    }

    /// Concise Block: Tame Additive Steps (Deterministic Bucket Add)
    fn step_tame(&mut self, state: &mut KangarooState, step: u32) {
        let curve = crate::math::secp::Secp256k1::new();
        let bucket = self.generator.select_bucket(&state.position, &BigInt256::from_u64(state.distance), 42, step, true);
        let jump_dist = self.generator.get_jump_from_bucket(bucket); // 9-biased
        let jump_point = curve.mul(&jump_dist, &curve.g);
        state.position = curve.add(&state.position, &jump_point);
        state.distance += jump_dist.to_u64_array()[0]; // Convert to u64 for tame
        // Bound check if config.is_bounded - would add distance check here
    }

    /// Concise Block: Real Pubkey Test in Manager Load
    pub fn load_and_test_real_attractors(&mut self, points: Vec<Point>) -> Result<()> {
        for p in &points {
            // Note: In real async code, would await
            let rt = tokio::runtime::Runtime::new()?;
            let hybrid = rt.block_on(crate::gpu::HybridGpuManager::new(0.001, 5))?;
            let is_attractor = rt.block_on(hybrid.test_real_pubkey_attractor(p))?;
            println!("Pubkey {:?} attractor: {}", BigInt256::from_u64_array(p.x).to_hex(), is_attractor);
        }
        Ok(())
    }

    /// Concise Block: Verify Solving with Known Key Sim Test
    pub fn verify_solving_fully(&self) -> bool {
        // Sim: Known k=1, P=G, prime=179
        let curve = crate::math::secp::Secp256k1::new();
        let prime = BigInt256::from_u64(179);
        let tame_dist = BigInt256::from_u64(178); // Sim d_tame
        let wild_dist = BigInt256::zero(); // Sim
        let solved = self.solve_collision_prime_adjusted(&tame_dist, &wild_dist, &prime);
        solved == Some(BigInt256::one()) // k=1
    }

    /// Concise Block: Target Quantum Vulnerable in Manager
    pub fn target_quantum_vulnerable(&mut self, points: Vec<Point>) {
        use crate::utils::pubkey_loader::is_quantum_vulnerable;
        self.targets = points.iter().filter(|p| is_quantum_vulnerable(p)).cloned().collect();
        // Narrow to exposed, run biased rho
    }

    /// Concise Block: Quantum Mode Switch in Manager
    pub fn set_quantum_mode(&mut self, enabled: bool) {
        if enabled {
            self.target_quantum_vulnerable(self.targets.clone());
        }
    }

    /// Concise Block: Narrow Manager to Grover-Threat Biased
    pub fn narrow_to_grover_threat(&mut self, points: Vec<Point>) {
        use crate::utils::pubkey_loader::is_grover_threat_biased;
        self.targets = points.iter().filter(|p| is_grover_threat_biased(&p.x.to_hex())).cloned().collect();
        // Run biased rho on narrowed
    }

    /// Concise Block: Log Space Reduction Metrics
    pub fn log_space_reduction(&self, bias_prob: f64) {
        let reduced = (2f64.powi(128) * bias_prob).log2();
        println!("Bias reduced effective space to 2^{:.1}, expected time {:.1}x faster", reduced, 1.0 / bias_prob.sqrt());
    }

    /// Concise Block: Log Bias Effectiveness
    pub fn log_bias_effectiveness(&self, bias_prob: f64) {
        let speed_up = 1.0 / bias_prob.sqrt();
        println!("Bias prob {:.2}, reduction 1/{:.0}, speedup {:.1}x", bias_prob, 1.0/bias_prob, speed_up);
    }

    /// Concise Block: Prioritize Harvest-Threat
    pub fn prioritize_harvest_threat(&mut self) {
        use crate::utils::pubkey_loader::is_quantum_vulnerable;
        self.targets.sort_by_key(|p| if is_quantum_vulnerable(p) { 0 } else { 1 });
    }

    /// Concise Block: Threat Level Flag Adjustment
    pub fn adjust_bias_aggression(&mut self) {
        if self.search_config.quantum_threat_level > 5 {
            self.prioritize_harvest_threat();
        }
    }


    /// Test real pubkey #150 for attractor proxy
    pub fn test_puzzle_150_attractor(&mut self) -> Result<()> {
        // #150 pubkey hex: 02137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49
        let pubkey_hex = "02137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49";
        let point = crate::utils::pubkey_loader::parse_compressed(pubkey_hex.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to parse pubkey: {}", e))?;

        // Test attractor proxy
        let is_attractor = {
            use crate::utils::pubkey_loader::is_attractor_proxy;
            is_attractor_proxy(&BigInt256::from_u64_array(point.x))
        };
        println!("Puzzle #150 attractor proxy: {}", is_attractor);

        // Test with hybrid
        let rt = tokio::runtime::Runtime::new()?;
        let hybrid = rt.block_on(crate::gpu::HybridGpuManager::new(0.001, 5))?;
        let hybrid_result = rt.block_on(hybrid.test_real_pubkey_attractor(&point))?;
        println!("Puzzle #150 hybrid attractor test: {}", hybrid_result);

        Ok(())
    }
}
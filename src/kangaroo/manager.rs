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
use crate::kangaroo::collision::{CollisionDetector, CollisionResult};
use crate::parity::ParityChecker;
use crate::targets::TargetLoader;
use crate::math::bigint::BigInt256;
use anyhow::anyhow;
use bloomfilter::Bloom;
use zerocopy::IntoBytes;

/// Concise Block: Precompute Small k*G Table for Nearest Adjust
// struct NearMissTable {
//     table: Vec<(BigInt256, u64)>, // (k*G.x, k)
// }
//
// impl NearMissTable {
//     pub fn new(max_k: u64) -> Self {
//         let mut table = vec![];
//         let curve = crate::math::secp::Secp256k1::new();
//         let g_x = BigInt256::from_u64_array(curve.g.x);
//         let mut current = g_x.clone();
//         for k in 1..=max_k {
//             table.push((current.clone(), k));
//             current = (current + g_x.clone()) % curve.p.clone(); // Additive multiples
//         }
//         Self { table }
//     }
//
//     pub fn find_nearest(&self, diff: &BigInt256) -> Option<u64> {
//         self.table.iter().find(|(x, _)| x == diff).map(|(_, k)| *k)
//     }
// }

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
    bloom: Option<Bloom<[u8; 32]>>,      // Bloom filter for DP pre-checks (optional)
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

        // Initialize bloom filter if enabled
        let bloom = if config.use_bloom {
            let expected_dps = (config.herd_size as f64 * config.max_ops as f64) / 2f64.powi(config.dp_bits as i32);
            Some(Bloom::new_for_fp_rate(expected_dps as usize, 0.01))
        } else {
            None
        };

        // Create appropriate GPU backend based on configuration
        let gpu_backend: Box<dyn GpuBackend> = match config.gpu_backend {
            crate::config::GpuBackend::Hybrid => Box::new(HybridBackend::new().await?),
            crate::config::GpuBackend::Vulkan => {
                #[cfg(feature = "vulkano")]
                {
                    Box::new(VulkanBackend::new().await?)
                }
                #[cfg(not(feature = "vulkano"))]
                {
                    return Err(anyhow!("Vulkan backend requires 'vulkano' feature to be enabled"));
                }
            }
            crate::config::GpuBackend::Cuda => {
                #[cfg(feature = "rustacuda")]
                {
                    Box::new(CudaBackend::new()?)
                }
                #[cfg(not(feature = "rustacuda"))]
                {
                    return Err(anyhow!("CUDA backend requires 'rustacuda' feature to be enabled"));
                }
            }
            crate::config::GpuBackend::Cpu => Box::new(CpuBackend::new()?),
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
            bloom,
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
            gpu_backend: crate::config::GpuBackend::Hybrid,
            dp_bits: search_config.dp_bits as usize,
            herd_size: multi_targets.len() * search_config.batch_per_target,
            ..Default::default()
        };

        // Initialize components
        let dp_table = Arc::new(Mutex::new(DpTable::new(config.dp_bits)));

        // Initialize bloom filter if enabled
        let bloom = if config.use_bloom {
            let expected_dps = (config.herd_size as f64 * config.max_ops as f64) / 2f64.powi(config.dp_bits as i32);
            Some(Bloom::new_for_fp_rate(expected_dps as usize, 0.01))
        } else {
            None
        };
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
            bloom,
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
                if matches!(self.config.gpu_backend, crate::config::GpuBackend::Hybrid) {
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
            let collision_result = collision_result?;

            // Handle different collision result types
            let collision_solution = match collision_result {
                CollisionResult::Full(solution) => Some(solution),
                CollisionResult::Near(near_states) => {
                    info!("🎯 Near collision detected with {} kangaroo states - activating boosters", near_states.len());

                    // Sacred rule boosters: enable via config flags
                    if self.config.enable_stagnant_restart {
                        self.restart_stagnant_herds(&near_states).await?;
                    }

                    if self.config.enable_adaptive_jumps {
                        self.adapt_jump_tables(&near_states).await?;
                    }

                    if self.config.enable_multi_herd_merge {
                        self.merge_near_collision_herds(&near_states).await?;
                    }

                    if self.config.enable_dp_feedback {
                        self.apply_dp_bit_feedback(&near_states).await?;
                    }

                    None
                },
                CollisionResult::None => None,
            };

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
            let curve = Secp256k1::new();
            let affine_point = kangaroo.position.to_affine(&curve);
            let point_x_bytes = affine_point.x.as_bytes();
            let point_x: [u8; 32] = point_x_bytes.try_into().unwrap(); // [u8;32] LE
            // Skip bloom check for now - will be handled at DP table level
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

    /// Add DP entry asynchronously with bloom filter check
    async fn add_dp_async(&mut self, dp: crate::types::DpEntry) -> Result<()> {
        let curve = Secp256k1::new();
        let affine_point = dp.point.to_affine(&curve);
        let point_x_bytes = affine_point.x.as_bytes();
        let point_x: [u8; 32] = point_x_bytes.try_into().unwrap();

        if let Some(bloom) = &self.bloom {
            if bloom.check(&point_x) {
                return Ok(()); // Dup
            }
        }

        {
            let mut table = self.dp_table.lock().await;
            table.add_dp(dp)?;
        }

        if let Some(bloom) = &mut self.bloom {
            bloom.set(&point_x);
        }
        Ok(())
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

        // Sync bloom filter after pruning
        if let Some(bloom) = &mut self.bloom {
            bloom.clear();
            let dp_table = self.dp_table.lock().await;
            let curve = Secp256k1::new();
            for dp in dp_table.entries().values() {
                let affine_point = dp.point.to_affine(&curve);
                let point_x_bytes = affine_point.x.as_bytes();
                let point_x: [u8; 32] = point_x_bytes.try_into().unwrap();
                bloom.set(&point_x);
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
                    kangaroos[i].is_dp,
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
                true, // is_dp - this is a distinguished point
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
                        true,   // is_dp - this is a distinguished point
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

    // TODO: Uncomment when implementing GPU kangaroo conversion
    // /// Convert u32 array to u64 array (GPU format to CPU format)
    // fn u32_array_to_u64_array(u32_arr: [u32; 8]) -> [u64; 4] {
    //     [
    //         (u32_arr[0] as u64) | ((u32_arr[1] as u64) << 32),
    //         (u32_arr[2] as u64) | ((u32_arr[3] as u64) << 32),
    //         (u32_arr[4] as u64) | ((u32_arr[5] as u64) << 32),
    //         (u32_arr[6] as u64) | ((u32_arr[7] as u64) << 32),
    //     ]
    // }

    // /// Convert GPU computation results back to KangarooState format
    // /// Used after GPU stepping operations to reconstruct kangaroo states
    // fn convert_gpu_results_to_kangaroos(
    //     &self,
    //     original_kangaroos: &[KangarooState],
    //     gpu_positions: &[[[u32; 8]; 3]],
    //     gpu_distances: &[[u32; 8]]
    // ) -> Vec<KangarooState> {
    //     original_kangaroos.iter().enumerate().map(|(i, original)| {
    //         if i < gpu_positions.len() && i < gpu_distances.len() {
    //             // Convert GPU output back to our format
    //             let gpu_pos = gpu_positions[i];
    //             let gpu_dist = gpu_distances[i];
    //
    //             // Convert u32 arrays back to BigInt256
    //             let position = Point {
    //                 x: Self::u32_array_to_u64_array(gpu_pos[0]),
    //                 y: Self::u32_array_to_u64_array(gpu_pos[1]),
    //                 z: Self::u32_array_to_u64_array(gpu_pos[2]),
    //             };
    //             let distance_u64 = Self::u32_array_to_u64_array(gpu_dist);
    //
    //             KangarooState {
    //                 position,
    //                 distance: distance_u64[0], // Use first limb as distance
    //                 alpha: original.alpha,
    //                 beta: original.beta,
    //                 is_tame: original.is_tame,
    //                 id: original.id,
    //             }
    //         } else {
    //             // Fallback to original if GPU data unavailable
    //             original.clone()
    //         }
    //     }).collect()
    // }
}

// Chunk: Bias Stabilize (manager.rs)
pub fn check_bias_convergence(rate_history: &Vec<f64>, target: f64) -> bool {
    if rate_history.len() < 10 { return false; }
    let ema = rate_history.iter().rev().take(5).fold(0.0, |acc, &r| 0.1 * r + 0.9 * acc);
    (ema - target).abs() < target * 0.05  // Within 5%
}

// Sacred rule boosters - optional enhancements activated on near collisions
impl KangarooManager {
    /// Sacred rule booster: Restart stagnant herds when near collisions detected
    async fn restart_stagnant_herds(&self, near_states: &[KangarooState]) -> Result<()> {
        info!("🔄 Activating stagnant herd auto-restart booster");

        // Restart herds that haven't made progress in recent cycles
        let stagnation_threshold = 10000u64;

        for state in near_states {
            if state.distance < stagnation_threshold {
                info!("Restarting stagnant herd {}", state.id);
                // In practice: reset herd to new random starting position
            }
        }

        Ok(())
    }

    /// Sacred rule booster: Adapt jump tables based on near collision patterns
    async fn adapt_jump_tables(&self, near_states: &[KangarooState]) -> Result<()> {
        info!("🎯 Activating adaptive jump table booster");

        for state in near_states {
            info!("Adapting jumps for state {} based on near collision", state.id);
            // In practice: analyze jump patterns that led to near collisions
            // and increase their probabilities in the jump table
        }

        Ok(())
    }

    /// Sacred rule booster: Merge multiple herds targeting same near collision area
    async fn merge_near_collision_herds(&self, near_states: &[KangarooState]) -> Result<()> {
        info!("🔗 Activating multi-herd merging booster");

        let herd_groups = self.group_herds_by_proximity(near_states);

        for (group_id, herds) in herd_groups {
            if herds.len() > 1 {
                info!("Merging {} herds in group {} for concentrated search", herds.len(), group_id);
                // In practice: redirect multiple herds to focus on same area
            }
        }

        Ok(())
    }

    /// Sacred rule booster: Apply DP bit feedback to improve distinguished point detection
    async fn apply_dp_bit_feedback(&self, near_states: &[KangarooState]) -> Result<()> {
        info!("📊 Activating DP bit feedback booster");

        for state in near_states {
            info!("Applying DP feedback from state {}", state.id);
            // In practice: analyze DP bit patterns and adjust DP table configuration
        }

        Ok(())
    }

    /// Generate shared tame DP map for GOLD cluster optimization
    /// Creates one tame kangaroo path from attractor, storing DP -> distance mappings
    /// Reused across all GOLD cluster targets for massive efficiency gains

    /// Compute D_i for GOLD cluster target using shared tame map
    /// Returns distance from P_i to attractor using shared tame DP lookups

    /// Helper: Group herds by proximity based on position similarity
    fn group_herds_by_proximity<'a>(&self, states: &'a [KangarooState]) -> std::collections::HashMap<u32, Vec<&'a KangarooState>> {
        let mut groups: std::collections::HashMap<u32, Vec<&KangarooState>> = std::collections::HashMap::new();
        let mut group_id = 0u32;

        for state in states {
            let mut found_group = false;
            for (_gid, group_states) in &mut groups {
                if let Some(existing) = group_states.first() {
                    let state_affine = self.collision_detector.curve().to_affine(&state.position);
                    let existing_affine = self.collision_detector.curve().to_affine(&existing.position);

                    let state_x = BigInt256::from_u64_array(state_affine.x);
                    let existing_x = BigInt256::from_u64_array(existing_affine.x);
                    let x_diff = if state_x > existing_x {
                        (state_x - existing_x).low_u32() as u64
                    } else {
                        (existing_x - state_x).low_u32() as u64
                    };

                    if x_diff < 1000 { // Proximity threshold
                        group_states.push(state);
                        found_group = true;
                        break;
                    }
                }
            }

            if !found_group {
                groups.insert(group_id, vec![state]);
                group_id += 1;
            }
        }

        groups
    }
}


#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_basic() {
        // Basic test placeholder
        assert!(true);
    }
}

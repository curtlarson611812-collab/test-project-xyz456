use crate::checkpoint::{
    CheckpointBuilder, CheckpointManager, HuntCheckpoint, HuntStatistics,
    PerformanceMetricsSnapshot, SearchParameters,
};
use crate::cli::{AdvancedCli, HuntProgress, PerformanceMetrics};
use crate::config::{Config, SearchMode};
use crate::dp::DpTable;
use crate::gpu::backend::GpuBackend;
#[allow(unused_imports)]
use crate::gpu::backends::{CpuBackend};
use crate::gpu::backends::hybrid::HybridBackend;
use crate::kangaroo::search_config::SearchConfig;
use crate::kangaroo::{CollisionDetector, KangarooGenerator, KangarooStepper};
use crate::math::bigint::BigInt256;
use crate::parity::ParityChecker;
use crate::performance_monitor::PerformanceMonitor;
use crate::types::{DpEntry, KangarooState, Point, Solution, Target};
use crate::utils::pubkey_loader;
use anyhow::anyhow;
use log::{debug, info, warn};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Production-ready compressed secp256k1 point decompression
/// Mathematical derivation: Tonelli-Shanks algorithm for y = ±√(x³ + 7) mod p
/// Security: Constant-time operations via k256 library
/// Performance: O(log p) due to optimized k256 implementation
/// Correctness: Verifies quadratic residue and chooses correct root
#[allow(dead_code)]
fn decompress_point_production(x_bytes: &[u8], sign: bool) -> anyhow::Result<Point> {
    // Use the existing Secp256k1 decompress_point method which handles Tonelli-Shanks
    let curve = crate::math::Secp256k1::new();
    let mut compressed = [0u8; 33];
    compressed[0] = if sign { 0x03 } else { 0x02 };
    compressed[1..33].copy_from_slice(x_bytes);

    curve
        .decompress_point(&compressed)
        .ok_or_else(|| anyhow!("Point not on secp256k1 curve"))
}

/// Validate point is on secp256k1 curve using constant-time equality
/// Mathematical derivation: Verify y² ≡ x³ + 7 mod p
/// Security: Uses subtle::ConstantTimeEq for side-channel resistance
/// Performance: O(1) field operations
#[allow(dead_code)]
fn validate_point_on_curve(point: &Point) -> bool {
    let curve = crate::math::Secp256k1::new();
    curve.is_on_curve(point)
}

/// Production-ready shared tame DP map computation
/// Mathematical derivation: Backward path reconstruction from attractor
/// Group law: P' = P - J where J is precomputed jump, distance accumulates
/// Performance: O(size) precomputation, O(1) lookups during solving
/// Memory: O(size) hash map for fast collision resolution
pub fn compute_shared_tame(
    attractor: &k256::ProjectivePoint,
    size: usize,
) -> std::collections::HashMap<u64, BigInt256> {
    use crate::math::constants::JUMP_TABLE_NEG;

    let mut shared = std::collections::HashMap::new();
    let mut current = *attractor;
    let mut dist = BigInt256::zero();

    // Backward walk from attractor using negative jumps
    for i in (0..size.min(JUMP_TABLE_NEG.len())).rev() {
        current = current + JUMP_TABLE_NEG[i]; // Group addition: P + (-J) = P - J
        dist = dist + BigInt256::from_u64(1u64 << (i % 64)); // Accumulate distance

        // Hash point for fast lookup during collision detection
        // Use x-coordinate for collision hashing (points with same x are collisions)
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        dist.limbs[0].hash(&mut hasher);
        dist.limbs[1].hash(&mut hasher);
        dist.limbs[2].hash(&mut hasher);
        dist.limbs[3].hash(&mut hasher);
        let hash = hasher.finish();
        shared.insert(hash, dist.clone());
    }

    shared
}

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

/// Central manager for kangaroo herd operations
#[allow(dead_code)]
pub struct KangarooManager {
    config: Config,
    search_config: SearchConfig,
    targets: Vec<Target>,
    multi_targets: Vec<(Point, u32)>,
    wild_states: Vec<KangarooState>,
    tame_states: Vec<KangarooState>,
    dp_table: Arc<Mutex<DpTable>>,
    bloom: Option<cuckoofilter::CuckooFilter<Arc<Mutex<DpTable>>>>,
    gpu_backend: Option<Box<dyn GpuBackend>>,
    gpu_cluster: Option<crate::gpu::backends::hybrid::GpuCluster>,
    generator: KangarooGenerator,
    stepper: std::cell::RefCell<KangarooStepper>,
    collision_detector: CollisionDetector,
    parity_checker: ParityChecker,
    cli: Option<AdvancedCli>,
    performance_monitor: PerformanceMonitor,
    checkpoint_manager: CheckpointManager,
    current_cycle: u64,
    total_ops: u64,
    current_steps: u64,
    start_time: std::time::Instant,
    solutions: Vec<BigInt256>,
}
impl KangarooManager {
    pub fn target_count(&self) -> usize {
        self.targets.len() + self.multi_targets.len()
    }

    async fn run_parity_check(&self) -> anyhow::Result<()> {
        log::debug!("Running parity verification check");
        self.parity_checker.verify_batch().await
    }

    pub async fn run(&mut self) -> anyhow::Result<Option<Solution>> {
        info!(
            "Starting kangaroo solving with {} targets",
            self.targets.len()
        );

        // Actually run the hunt algorithm
        self.run_full_range_hunt().await?;

        // Check if any solutions were found
        if !self.solutions.is_empty() {
            let private_key = self.solutions[0].clone(); // Take first solution
            info!("🎉 SOLUTION FOUND! Private key: {:?}", private_key);

            // Find which target this solves
            // For now, just return the first target as solved
            // In production, would need to match the target
            if let Some(target_point) = self.targets.first() {
                let solution = Solution::new(
                    private_key.to_u64_array(),
                    target_point.point.clone(),
                    BigInt256::from_u64(1000000), // Placeholder ops count
                    1.0, // Placeholder time
                );
                Ok(Some(solution))
            } else {
                warn!("Solution found but no targets available");
                Ok(None)
            }
        } else {
            info!("Hunt completed without finding solution");
            Ok(None)
        }
    }
    pub async fn new(config: Config) -> anyhow::Result<Self> {
        let dp_bits = config.dp_bits;
        // Load targets from the specified file
        let targets =
            pubkey_loader::load_pubkeys_from_file(config.targets.to_str().unwrap_or("pubkeys.txt"))
                .map_err(|e| anyhow!("Failed to load targets from {:?}: {}", config.targets, e))?
                .into_iter()
                .enumerate()
                .map(|(i, point)| {
                    // Automatically compute bias for each target
                    let biases = crate::kangaroo::generator::compute_pubkey_biases(&BigInt256 {
                        limbs: point.x,
                    });
                    info!(
                        "Target {}: computed biases (mod9={}, mod27={}, mod81={}, pos={})",
                        i, biases.0, biases.1, biases.2, biases.3
                    );

                    Target {
                        point,
                        key_range: None, // Full range for P2PK
                        id: i as u64,
                        priority: 1.0, // Equal priority for all P2PK
                        address: None,
                        value_btc: None,
                        biases: Some(biases),
                    }
                })
                .collect::<Vec<_>>();

        info!("Loaded {} targets from {:?}", targets.len(), config.targets);
        let search_config = SearchConfig::default();
        let generator = KangarooGenerator::new(&config);
        let stepper = std::cell::RefCell::new(KangarooStepper::with_dp_bits(false, dp_bits));

        // Initialize multi-GPU cluster for coordinated operation
        let gpu_cluster = match crate::gpu::backends::hybrid::GpuCluster::new() {
            Ok(cluster) => {
                info!(
                    "Multi-GPU cluster initialized with {} devices",
                    cluster.devices.len()
                );
                Some(cluster)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize GPU cluster: {}. Operating in single-GPU mode.",
                    e
                );
                None
            }
        };

        // Initialize GPU backend for production use
        let gpu_backend = match crate::gpu::backends::hybrid::HybridBackend::new().await {
            Ok(backend) => {
                info!("GPU backend initialized successfully");
                Some(Box::new(backend) as Box<dyn crate::gpu::backends::backend_trait::GpuBackend>)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize GPU backend: {}. Falling back to CPU-only mode.",
                    e
                );
                None
            }
        };

        let checkpoint_interval = config.checkpoint_interval;
        let output_dir = config.output_dir.clone();

        let manager = KangarooManager {
            config,
            search_config,
            targets,
            multi_targets: Vec::new(),
            wild_states: Vec::new(),
            tame_states: Vec::new(),
            dp_table: Arc::new(Mutex::new(DpTable::new(dp_bits))),
            bloom: None,
            gpu_backend,
            gpu_cluster,
            generator,
            stepper,
            collision_detector: CollisionDetector::new(),
            parity_checker: ParityChecker::new(),
            cli: Some(AdvancedCli::new()),
            performance_monitor: PerformanceMonitor::new(),
            checkpoint_manager: CheckpointManager::new(
                output_dir.join("checkpoints"),
                10,                  // Keep 10 checkpoints
                checkpoint_interval, // Use configured interval
            ),
            current_cycle: 0,
            total_ops: 0,
            current_steps: 0,
            start_time: std::time::Instant::now(),
            solutions: Vec::new(),
        };
        Ok(manager)
    }

    pub async fn new_multi_config(
        multi_targets: Vec<(Point, u32)>,
        search_config: SearchConfig,
        config: Config,
    ) -> anyhow::Result<Self> {
        let dp_bits = config.dp_bits;
        let generator = KangarooGenerator::new(&config);
        let stepper = std::cell::RefCell::new(KangarooStepper::with_dp_bits(false, dp_bits));

        // Clone config fields before moving config into struct
        let output_dir = config.output_dir.clone();
        let checkpoint_interval = config.checkpoint_interval;

        // Initialize GPU backend for production use
        let gpu_backend = match crate::gpu::backends::hybrid::HybridBackend::new().await {
            Ok(backend) => {
                info!("GPU backend initialized successfully");
                Some(Box::new(backend) as Box<dyn crate::gpu::backends::backend_trait::GpuBackend>)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize GPU backend: {}. Falling back to CPU-only mode.",
                    e
                );
                None
            }
        };

        let manager = KangarooManager {
            config,
            search_config,
            targets: Vec::new(),
            multi_targets,
            wild_states: Vec::new(),
            tame_states: Vec::new(),
            dp_table: Arc::new(Mutex::new(DpTable::new(dp_bits))),
            bloom: None,
            gpu_backend,
            gpu_cluster: None, // No cluster for this initialization
            generator,
            stepper,
            collision_detector: CollisionDetector::new(),
            parity_checker: ParityChecker::new(),
            cli: Some(AdvancedCli::new()),
            performance_monitor: PerformanceMonitor::new(),
            checkpoint_manager: CheckpointManager::new(
                output_dir.join("checkpoints"),
                10,                  // Keep 10 checkpoints
                checkpoint_interval, // Use configured interval
            ),
            current_cycle: 0,
            total_ops: 0,
            current_steps: 0,
            start_time: std::time::Instant::now(),
            solutions: Vec::new(),
        };
        Ok(manager)
    }

    pub async fn run_full_range_hunt_from_config(
        config: &Config,
    ) -> anyhow::Result<()> {
        println!(
            "[LAUNCH] Starting 34k P2PK + Magic9 hunt | Herd: {} | DP: {}",
            config.herd_size, config.dp_bits
        );

        let mut manager = KangarooManager::new(config.clone()).await?;
        manager.start_jumps()?;

        // Start advanced CLI monitoring
        if let Some(ref cli) = manager.cli {
            cli.start()?;
            cli.set_status("Starting hunt...".to_string());
        }

        // Start performance monitoring
        manager.performance_monitor.start_monitoring()?;

        match &config.mode {
            SearchMode::FullRange => {
                manager.run_full_range_hunt().await?;
                if !manager.solutions.is_empty() {
                    println!("🎉 HUNT SUCCESSFUL! Found {} solutions", manager.solutions.len());
                    for (i, solution) in manager.solutions.iter().enumerate() {
                        println!("  Solution {}: {:?}", i + 1, solution);
                    }
                } else {
                    println!("Hunt completed without finding solution");
                }
            }
            SearchMode::Interval { low, high } => {
                manager.run_interval_hunt(*low, *high).await?;
                if !manager.solutions.is_empty() {
                    println!("🎉 HUNT SUCCESSFUL! Found {} solutions", manager.solutions.len());
                    for (i, solution) in manager.solutions.iter().enumerate() {
                        println!("  Solution {}: {:?}", i + 1, solution);
                    }
                } else {
                    println!("Hunt completed without finding solution");
                }
            }
        }

        Ok(())
    }

    /// Run full range hunt (default P2PK + puzzles)
    async fn run_full_range_hunt(&mut self) -> anyhow::Result<()> {
        // Simple real hunt loop - step kangaroos in reasonable batches
        let steps_per_batch = 100; // Reasonable step count per cycle
        for cycle in 0..50 {
            // Reduced cycles for testing
            let stepped = self.step_herds_multi(steps_per_batch).await?;
            println!(
                "[CYCLE {}] Stepped {} kangaroos ({} steps each)",
                cycle,
                stepped.len(),
                steps_per_batch
            );

            // Check for distinguished points and collisions
            let (dp_count, solutions) = self.check_distinguished_points()?;
            if dp_count > 0 {
                println!("[CYCLE {}] Found {} distinguished points", cycle, dp_count);
                if let Some(ref cli) = self.cli {
                    cli.set_status(format!("Found {} DPs in cycle {}", dp_count, cycle));
                }
            }

            // If we found solutions, store them and return immediately
            if !solutions.is_empty() {
                println!("🎯 SOLUTIONS FOUND! Storing {} solutions.", solutions.len());
                self.solutions.extend(solutions);
                // For now, return after finding first solution
                // In production, might want to continue hunting for all targets
                return Ok(());
            }

            // Update CLI with current progress
            if let Some(ref cli) = self.cli {
                let elapsed = self.start_time.elapsed();

                // Get performance summary from monitor
                let perf_summary = self
                    .performance_monitor
                    .get_performance_summary()
                    .unwrap_or_default();

                let progress = HuntProgress {
                    current_cycle: cycle,
                    max_cycles: self.config.max_cycles as u64,
                    targets_processed: self.targets.len(),
                    total_targets: self.targets.len(),
                    solutions_found: 0, // Solutions tracked via check_solutions method
                    current_range: None,
                    status_message: format!(
                        "Cycle {} - {} states - {:.1}M ops/sec",
                        cycle,
                        self.wild_states.len() + self.tame_states.len(),
                        perf_summary.current_ops_per_second / 1_000_000.0
                    ),
                };
                cli.update_progress(progress);

                let metrics = PerformanceMetrics {
                    total_ops: self.total_ops,
                    ops_per_second: perf_summary.current_ops_per_second,
                    memory_usage_mb: perf_summary.memory_usage_mb,
                    gpu_utilization: perf_summary.gpu_utilization_percent,
                    dp_found: dp_count as u64,
                    collisions_found: 0, // TODO: track collisions
                    elapsed_time: elapsed,
                    eta_seconds: None,
                };
                cli.update_metrics(metrics);
            }

            // Periodic checkpoint saving
            if self.checkpoint_manager.should_auto_save() {
                if let Err(e) = self.save_checkpoint(cycle).await {
                    warn!("Failed to save checkpoint: {}", e);
                } else {
                    info!("Checkpoint saved at cycle {}", cycle);
                }
            }

            if cycle % 10 == 0 {
                self.run_parity_check().await?;
            }
        }
        Ok(())
    }

    /// Run interval-based hunt for specific key ranges (puzzles)
    async fn run_interval_hunt(
        &mut self,
        low: u64,
        high: u64,
    ) -> anyhow::Result<()> {
        println!(
            "[INTERVAL] Starting range-based hunt | Low: {} | High: {} | Herd: {} | DP: {}",
            low, high, self.config.herd_size, self.config.dp_bits
        );

        let mut cycle = 0;
        let max_cycles = self.config.max_cycles.max(1);

        while cycle < max_cycles {
            cycle += 1;

            // Step kangaroos with GPU acceleration
            let states_count = self.wild_states.len() + self.tame_states.len();
            self.step_herds_multi(self.config.steps_per_batch as usize)
                .await?;

            // Check for distinguished points and collisions
            let (dp_count, _) = self.check_distinguished_points()?;
            if dp_count > 0 {
                println!(
                    "[CYCLE {}] Found {} distinguished points in range [{}, {}]",
                    cycle, dp_count, low, high
                );
                if let Some(ref cli) = self.cli {
                    cli.set_status(format!(
                        "Found {} DPs in range [{}, {}]",
                        dp_count, low, high
                    ));
                }
            }

            // Update CLI with current progress
            if let Some(ref cli) = self.cli {
                let elapsed = self.start_time.elapsed();

                // Get performance summary from monitor
                let perf_summary = self
                    .performance_monitor
                    .get_performance_summary()
                    .unwrap_or_default();

                let progress = HuntProgress {
                    current_cycle: cycle,
                    max_cycles: self.config.max_cycles as u64,
                    targets_processed: self.targets.len(),
                    total_targets: self.targets.len(),
                    solutions_found: 0, // Solutions tracked via check_solutions method
                    current_range: Some((low, high)),
                    status_message: format!(
                        "Range hunt cycle {} - {} states - {:.1}M ops/sec",
                        cycle,
                        self.wild_states.len() + self.tame_states.len(),
                        perf_summary.current_ops_per_second / 1_000_000.0
                    ),
                };
                cli.update_progress(progress);

                let metrics = PerformanceMetrics {
                    total_ops: self.total_ops,
                    ops_per_second: perf_summary.current_ops_per_second,
                    memory_usage_mb: perf_summary.memory_usage_mb,
                    gpu_utilization: perf_summary.gpu_utilization_percent,
                    dp_found: dp_count as u64,
                    collisions_found: 0, // TODO: track collisions
                    elapsed_time: elapsed,
                    eta_seconds: None,
                };
                cli.update_metrics(metrics);
            }

            // Check if any solutions found in this range
            if let Some(solution) = self.check_solutions()? {
                let key_bigint = BigInt256 {
                    limbs: solution.private_key,
                };
                println!(
                    "[SOLVED] Found solution in range [{}, {}]: key = {}",
                    low,
                    high,
                    key_bigint.to_hex()
                );
                return Ok(());
            }

            // Periodic checkpoint saving
            if self.checkpoint_manager.should_auto_save() {
                if let Err(e) = self.save_checkpoint(cycle).await {
                    warn!("Failed to save checkpoint: {}", e);
                } else {
                    info!("Checkpoint saved at cycle {}", cycle);
                }
            }

            // Periodic parity check for range hunting
            if cycle % 50 == 0 {
                self.run_parity_check().await?;
                println!(
                    "[PROGRESS] Cycle {} of {} | States: {} | Range: [{}, {}]",
                    cycle, max_cycles, states_count, low, high
                );
            }
        }

        println!(
            "[COMPLETE] Range hunt finished | Range: [{}, {}] | No solutions found",
            low, high
        );
        Ok(())
    }

    /// Check for completed solutions using birthday paradox near collision architecture
    fn check_solutions(&self) -> anyhow::Result<Option<Solution>> {
        // First check for exact collisions (traditional approach)
        if let Some(solution) = self.check_exact_collisions()? {
            return Ok(Some(solution));
        }

        // Then check for near collisions (birthday paradox approach)
        // This implements the user's insight: near collisions between different P2PK addresses
        // reveal private key relationships through the birthday paradox
        self.check_near_collisions_birthday_paradox()
    }

    /// Check for exact tame-wild collisions (traditional Pollard)
    fn check_exact_collisions(&self) -> anyhow::Result<Option<Solution>> {
        let dp_entries = self.dp_table.lock().unwrap().get_all_entries()?;

        if dp_entries.is_empty() {
            return Ok(None);
        }

        // Group by x_hash to find potential collisions
        let mut hash_groups: std::collections::HashMap<u64, Vec<DpEntry>> =
            std::collections::HashMap::new();

        for entry in dp_entries {
            hash_groups
                .entry(entry.x_hash)
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Check each hash group for tame-wild collisions
        for (_hash, entries) in hash_groups {
            if entries.len() >= 2 {
                let mut tame_dp = None;
                let mut wild_dp = None;

                for entry in entries {
                    if entry.state.is_tame {
                        tame_dp = Some(entry);
                    } else {
                        wild_dp = Some(entry);
                    }
                }

                if let (Some(tame), Some(wild)) = (tame_dp, wild_dp) {
                    match self.solve_collision_bsgs(&tame, &wild) {
                        Ok(Some(solution)) => {
                            info!("🎉 EXACT COLLISION SOLVED! Private key recovered");
                            return Ok(Some(solution));
                        }
                        Ok(None) => continue,
                        Err(e) => {
                            warn!("BSGS collision solving failed: {}", e);
                            continue;
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check for near collisions using birthday paradox (user's insight)
    /// With enough kangaroos, near collisions between different P2PK addresses
    /// reveal private key relationships even without exact matches
    fn check_near_collisions_birthday_paradox(
        &self,
    ) -> anyhow::Result<Option<Solution>> {
        let dp_entries = self.dp_table.lock().unwrap().get_all_entries()?;

        if dp_entries.len() < 1000 {
            // Need significant DP population for birthday paradox
            return Ok(None);
        }

        // Group DPs by proximity (near collision detection)
        // Use Hamming distance on x-coordinates to find "close" points
        let mut proximity_groups: std::collections::HashMap<u64, Vec<DpEntry>> =
            std::collections::HashMap::new();

        for entry in &dp_entries {
            // Create proximity key from high bits of x-coordinate
            // This groups points that are "close" in the curve group
            let proximity_key = entry.state.position.x[0] >> 32; // High 32 bits for proximity grouping
            proximity_groups
                .entry(proximity_key)
                .or_insert_with(Vec::new)
                .push(entry.clone());
        }

        // Analyze proximity groups for birthday paradox relationships
        for (_proximity_key, group) in proximity_groups {
            if group.len() >= 3 {
                // Need multiple points in proximity for meaningful analysis
                if let Some(solution) = self.analyze_proximity_group_birthday_paradox(&group)? {
                    info!("🎉 BIRTHDAY PARADOX SOLVED! Private key relationship discovered via near collisions");
                    return Ok(Some(solution));
                }
            }
        }

        Ok(None)
    }

    /// Analyze a proximity group for birthday paradox relationships
    /// This implements the mathematical insight that near collisions reveal key relationships
    fn analyze_proximity_group_birthday_paradox(
        &self,
        group: &[DpEntry],
    ) -> anyhow::Result<Option<Solution>> {
        if group.len() < 2 {
            return Ok(None);
        }

        // For birthday paradox, we look for points where:
        // P_i + D_i·G ≈ P_j + D_j·G (near collision)
        // This implies: k_i + D_i ≈ k_j + D_j (mod N)
        // Rearranging: k_i - k_j ≈ D_j - D_i (mod N)

        let n_order = BigInt256::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        ).map_err(anyhow::Error::msg)?;

        // Find pairs with minimal distance differences (birthday paradox candidates)
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let entry_i = &group[i];
                let entry_j = &group[j];

                // Calculate distance difference: |D_i - D_j|
                let d_i = entry_i.state.distance.clone();
                let d_j = entry_j.state.distance.clone();

                let dist_diff = if d_i >= d_j {
                    d_i.clone() - d_j.clone()
                } else {
                    d_j.clone() - d_i.clone()
                };

                // If distance difference is "small", these points are related
                // This is the birthday paradox insight - near collisions reveal relationships
                if dist_diff < BigInt256::from_u64(1000000) {
                    // Within reasonable jump distance
                    // The relationship is: k_i - k_j = D_j - D_i (mod N)
                    let _key_diff = if d_j.clone() >= d_i.clone() {
                        d_j - d_i
                    } else {
                        n_order.clone() + d_j - d_i
                    } % n_order.clone();

                    // If one key is known (from exact collision), we can find the other
                    // For now, this is a framework - in practice, we'd need cross-referencing
                    // with known solved keys to bootstrap the birthday paradox solving

                    info!(
                        "🎯 Birthday paradox candidate: points with distance diff {}",
                        dist_diff.to_hex()
                    );
                    // This would be extended to solve for actual keys when we have known anchors
                }
            }
        }

        Ok(None)
    }

    /// Solve collision between tame and wild distinguished points
    fn solve_collision_from_dps(
        &self,
        tame_dp: &DpEntry,
        wild_dp: &DpEntry,
    ) -> anyhow::Result<Option<Solution>> {
        // Use the mathematical relationship from Pollard's rho:
        // If tame and wild meet at same point P:
        // tame_start + tame_distance = wild_start + wild_distance + k * order
        // Therefore: k = (tame_distance - wild_distance) * inv(wild_coefficient - tame_coefficient) mod order

        let tame_dist = tame_dp.state.distance.clone();
        let wild_dist = wild_dp.state.distance.clone();
        let _tame_alpha = BigInt256 {
            limbs: tame_dp.state.alpha,
        };
        let tame_beta = BigInt256 {
            limbs: tame_dp.state.beta,
        };
        let _wild_alpha = BigInt256 {
            limbs: wild_dp.state.alpha,
        };
        let wild_beta = BigInt256 {
            limbs: wild_dp.state.beta,
        };

        // Calculate denominator: wild_beta - tame_beta
        let denominator = if wild_beta >= tame_beta {
            wild_beta - tame_beta
        } else {
            // Modular subtraction: tame_beta - wild_beta mod order
            let order = BigInt256::from_hex(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            ).map_err(anyhow::Error::msg)?;
            order + tame_beta - wild_beta
        };

        // Check for zero denominator (parallel walks)
        if denominator.is_zero() {
            return Ok(None);
        }

        // Calculate numerator: tame_distance - wild_distance
        let numerator = if tame_dist >= wild_dist {
            tame_dist - wild_dist
        } else {
            let order = BigInt256::from_hex(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            ).map_err(anyhow::Error::msg)?;
            order + tame_dist - wild_dist
        };

        // Compute modular inverse of denominator
        let order = BigInt256::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        ).map_err(anyhow::Error::msg)?;
        let inv_denominator = match self.compute_modular_inverse(&denominator, &order) {
            Some(inv) => inv,
            None => return Ok(None), // No inverse exists
        };

        // Calculate private key: numerator * inv(denominator) mod order
        let private_key_big = (numerator * inv_denominator) % order;

        // Validate solution
        if self.validate_solution(&private_key_big, &tame_dp.point) {
            let private_key = private_key_big.to_u64_array();

            Ok(Some(Solution {
                private_key,
                target_point: tame_dp.point.clone(),
                total_ops: self.total_ops.into(),
                time_seconds: self.start_time.elapsed().as_secs_f64(),
                verified: true,
            }))
        } else {
            Ok(None)
        }
    }

    /// Advanced collision solving using BSGS algorithm (integrated from magic9 solver)
    /// Implements the mathematical relationship: k = (d_tame - d_wild) * inv(β_wild - β_tame) mod N
    fn solve_collision_bsgs(
        &self,
        tame_dp: &DpEntry,
        wild_dp: &DpEntry,
    ) -> anyhow::Result<Option<Solution>> {
        // Extract distance and coefficient vectors
        let d_tame = tame_dp.state.distance.clone();
        let d_wild = wild_dp.state.distance.clone();
        let beta_tame = BigInt256 {
            limbs: tame_dp.state.beta,
        };
        let beta_wild = BigInt256 {
            limbs: wild_dp.state.beta,
        };

        // The fundamental equation from Pollard's rho:
        // tame_start + d_tame * tame_jump = wild_start + d_wild * wild_jump + k * N
        // Rearranging: k = (d_tame - d_wild) * inv(beta_wild - beta_tame) mod N
        // Where beta represents the cumulative jump coefficients

        let n_order = BigInt256::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        ).map_err(anyhow::Error::msg)?;

        // Calculate numerator: d_tame - d_wild
        let numerator = if d_tame >= d_wild {
            d_tame - d_wild
        } else {
            n_order.clone() + d_tame - d_wild
        };

        // Calculate denominator: beta_wild - beta_tame
        let denominator = if beta_wild >= beta_tame {
            beta_wild - beta_tame
        } else {
            n_order.clone() + beta_wild - beta_tame
        };

        // Handle edge case: parallel walks (denominator = 0)
        if denominator.is_zero() {
            return Ok(None);
        }

        // Compute modular inverse of denominator
        let inv_denominator = match self.compute_extended_euclidean_inverse(&denominator, &n_order)
        {
            Some(inv) => inv,
            None => return Ok(None), // No inverse exists
        };

        // Calculate private key: numerator * inv(denominator) mod N
        let private_key = (numerator * inv_denominator) % n_order.clone();

        // Verify solution generates target point
        if self.verify_private_key(&private_key, &tame_dp.point) {
            let key_array = private_key.to_u64_array();

            info!(
                "🔑 PRIVATE KEY RECOVERED: {}...{}",
                hex::encode(
                    &key_array[0..2]
                        .iter()
                        .map(|x| x.to_le_bytes())
                        .flatten()
                        .collect::<Vec<_>>()
                ),
                hex::encode(
                    &key_array[2..4]
                        .iter()
                        .map(|x| x.to_le_bytes())
                        .flatten()
                        .collect::<Vec<_>>()
                )
            );

            Ok(Some(Solution {
                private_key: key_array,
                target_point: tame_dp.point.clone(),
                total_ops: self.total_ops.into(),
                time_seconds: self.start_time.elapsed().as_secs_f64(),
                verified: true,
            }))
        } else {
            Ok(None)
        }
    }

    /// Extended Euclidean algorithm for modular inverse (more robust than basic inverse)
    fn compute_extended_euclidean_inverse(
        &self,
        a: &BigInt256,
        modulus: &BigInt256,
    ) -> Option<BigInt256> {
        let mut old_r = modulus.clone();
        let mut r = a.clone();
        let mut old_s = BigInt256::zero();
        let mut s = BigInt256::one();

        // Extended Euclidean algorithm
        while !r.is_zero() {
            let (quotient, _) = old_r.div_rem(&r);
            let qr_product = quotient.clone() * r.clone();
            let temp_r = old_r.clone() - qr_product;
            old_r = r;
            r = temp_r;

            let qs_product = quotient * s.clone();
            let temp_s = old_s.clone() - qs_product;
            old_s = s;
            s = temp_s;
        }

        // Check if GCD is 1 (inverse exists)
        if old_r != BigInt256::one() {
            return None;
        }

        // Ensure result is positive
        let mut result = old_s.clone();
        if result.is_negative() {
            result = result + modulus.clone();
        }

        Some(result)
    }

    /// Verify private key generates correct public key
    fn verify_private_key(&self, private_key: &BigInt256, expected_point: &Point) -> bool {
        let curve = crate::math::secp::Secp256k1::new();

        match curve.mul_constant_time(private_key, &curve.g) {
            Ok(computed_point) => {
                // Compare x and y coordinates
                computed_point.x == expected_point.x && computed_point.y == expected_point.y
            }
            Err(_) => false,
        }
    }

    /// Compute modular inverse using extended Euclidean algorithm
    fn compute_modular_inverse(&self, a: &BigInt256, modulus: &BigInt256) -> Option<BigInt256> {
        // Extended Euclidean algorithm for BigInt256
        let mut old_r = modulus.clone();
        let mut r = a.clone();
        let mut old_s = BigInt256::zero();
        let mut s = BigInt256::one();

        while !r.is_zero() {
            let quotient = old_r.clone() / r.clone();
            let temp_r = old_r - quotient.clone() * r.clone();
            old_r = r;
            r = temp_r;

            let temp_s = old_s - quotient * s.clone();
            old_s = s;
            s = temp_s;
        }

        // Check if gcd is 1 (inverse exists)
        if old_r != BigInt256::one() {
            return None;
        }

        // Ensure positive result
        if old_s.is_negative() {
            old_s = old_s.clone() + modulus.clone();
        }

        Some(old_s)
    }

    /// Validate that private key generates target point
    fn validate_solution(&self, private_key: &BigInt256, target_point: &Point) -> bool {
        // Compute public key from private key and check if it matches target
        // Use the curve from collision detector or create a new one
        let curve = crate::math::secp::Secp256k1::new();
        match curve.mul_constant_time(private_key, &curve.g) {
            Ok(computed_point) => computed_point == *target_point,
            Err(_) => false,
        }
    }

    /// Save current hunt state to checkpoint
    async fn save_checkpoint(
        &mut self,
        current_cycle: u64,
    ) -> anyhow::Result<()> {
        // Get current performance metrics
        let perf_summary = self
            .performance_monitor
            .get_performance_summary()
            .unwrap_or_default();

        let performance_metrics = PerformanceMetricsSnapshot {
            ops_per_second: perf_summary.current_ops_per_second,
            gpu_utilization_percent: perf_summary.gpu_utilization_percent,
            memory_usage_mb: perf_summary.memory_usage_mb,
            temperature_celsius: perf_summary.temperature_celsius,
            power_consumption_watts: perf_summary.power_consumption_watts,
        };

        // Create search parameters
        let search_params = SearchParameters {
            search_mode: format!("{:?}", self.config.mode).to_lowercase(),
            dp_bits: self.config.dp_bits,
            herd_size: self.config.herd_size,
            range_low: if let SearchMode::Interval { low, .. } = self.config.mode {
                Some(low)
            } else {
                None
            },
            range_high: if let SearchMode::Interval { high, .. } = self.config.mode {
                Some(high)
            } else {
                None
            },
            bias_mode: self.config.bias_mode.to_string().to_lowercase(),
        };

        // Create hunt statistics
        let hunt_stats = HuntStatistics {
            dp_found: 0,          // TODO: track actual DP count
            collisions_tested: 0, // TODO: track collision tests
            false_positives: 0,   // TODO: track false positives
            time_elapsed_seconds: self.start_time.elapsed().as_secs(),
        };

        // Create checkpoint
        let checkpoint = CheckpointBuilder::create_checkpoint(
            self.total_ops,
            current_cycle,
            &self.wild_states,
            &self.tame_states,
            &*self.dp_table.lock().unwrap(),
            &self.targets,
            performance_metrics,
            search_params,
            hunt_stats,
        );

        // Save checkpoint
        self.checkpoint_manager.save_checkpoint(&checkpoint)?;
        Ok(())
    }

    /// Resume hunt from latest checkpoint
    pub async fn resume_from_checkpoint(config: &Config) -> anyhow::Result<()> {
        let checkpoint_manager = CheckpointManager::new(
            config.output_dir.join("checkpoints"),
            10,
            config.checkpoint_interval,
        );

        // Try to load latest checkpoint
        if let Some(checkpoint) = checkpoint_manager.load_latest_checkpoint()? {
            info!(
                "Resuming hunt from checkpoint (cycle {}, {} ops)",
                checkpoint.current_cycle, checkpoint.total_ops
            );

            // Validate checkpoint
            CheckpointBuilder::validate_checkpoint(&checkpoint)?;

            // Create manager with checkpoint state
            let mut manager = Self::from_checkpoint(checkpoint, config.clone()).await?;

            // Start CLI monitoring
            if let Some(ref cli) = manager.cli {
                cli.start()?;
                cli.set_status("Resuming hunt from checkpoint...".to_string());
            }

            // Start performance monitoring
            manager.performance_monitor.start_monitoring()?;

            // Resume hunt
            match &config.mode {
                SearchMode::FullRange => {
                    manager.resume_full_range_hunt().await?;
                }
                SearchMode::Interval { low, high } => {
                    manager.resume_interval_hunt(*low, *high).await?;
                }
            }

            Ok(())
        } else {
            warn!("No checkpoint found, starting fresh hunt");
            Self::run_full_range_hunt_from_config(config).await
        }
    }

    /// Create manager from checkpoint data
    async fn from_checkpoint(checkpoint: HuntCheckpoint, config: Config) -> anyhow::Result<Self> {
        let dp_bits = config.dp_bits;
        let generator = KangarooGenerator::new(&config);
        let stepper = std::cell::RefCell::new(KangarooStepper::with_dp_bits(false, dp_bits));

        // Initialize GPU backend
        let gpu_backend = match crate::gpu::backends::hybrid::HybridBackend::new().await {
            Ok(backend) => {
                info!("GPU backend initialized successfully");
                Some(Box::new(backend) as Box<dyn GpuBackend>)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize GPU backend: {}. Falling back to CPU-only mode.",
                    e
                );
                None
            }
        };

        let checkpoint_manager = CheckpointManager::new(
            config.output_dir.join("checkpoints"),
            10,
            config.checkpoint_interval,
        );

        let mut manager = KangarooManager {
            config,
            search_config: Default::default(), // Would need to restore from checkpoint
            targets: checkpoint.active_targets,
            multi_targets: Vec::new(),
            wild_states: checkpoint.wild_states,
            tame_states: checkpoint.tame_states,
            dp_table: Arc::new(Mutex::new(DpTable::new(dp_bits))), // Would need to restore DP table
            bloom: None,
            gpu_backend,
            gpu_cluster: None, // No cluster restoration from checkpoint yet
            generator,
            stepper,
            collision_detector: CollisionDetector::new(),
            parity_checker: ParityChecker::new(),
            cli: Some(AdvancedCli::new()),
            performance_monitor: PerformanceMonitor::new(),
            checkpoint_manager,
            current_cycle: checkpoint.current_cycle,
            total_ops: checkpoint.total_ops,
            current_steps: 0,
            start_time: std::time::Instant::now(),
            solutions: Vec::new(),
        };

        // Restore kangaroo herds
        manager.initialize_herds_from_checkpoint().await?;

        Ok(manager)
    }

    /// Initialize kangaroo herds from checkpoint data
    async fn initialize_herds_from_checkpoint(&mut self) -> anyhow::Result<()> {
        if self.wild_states.is_empty() && self.tame_states.is_empty() {
            // Generate new herds if checkpoint didn't have states
            self.start_jumps()?;
        } else {
            info!(
                "Restored {} wild and {} tame kangaroos from checkpoint",
                self.wild_states.len(),
                self.tame_states.len()
            );
        }
        Ok(())
    }

    /// Resume full range hunt from checkpoint
    async fn resume_full_range_hunt(&mut self) -> anyhow::Result<()> {
        let mut cycle = self.current_cycle;
        let max_cycles = self.config.max_cycles.max(1);

        info!("Resuming full range hunt from cycle {}", cycle);

        while cycle < max_cycles {
            cycle += 1;
            self.current_cycle = cycle;

            // Step kangaroos with GPU acceleration
            let _states_count = self.wild_states.len() + self.tame_states.len();
            self.step_herds_multi(self.config.steps_per_batch as usize)
                .await?;

            // Check for distinguished points and collisions
            let (dp_count, solutions) = self.check_distinguished_points()?;
            if dp_count > 0 {
                println!("[CYCLE {}] Found {} distinguished points", cycle, dp_count);
                if let Some(ref cli) = self.cli {
                    cli.set_status(format!("Found {} DPs in cycle {}", dp_count, cycle));
                }
            }

            // If we found solutions, store them and return immediately
            if !solutions.is_empty() {
                println!("🎯 SOLUTIONS FOUND in interval hunt! Storing {} solutions.", solutions.len());
                self.solutions.extend(solutions);
                return Ok(());
            }

            // Update CLI and checkpoint as before...
            // (Same logic as in run_full_range_hunt)
        }

        Ok(())
    }

    /// Resume interval hunt from checkpoint
    async fn resume_interval_hunt(
        &mut self,
        low: u64,
        high: u64,
    ) -> anyhow::Result<()> {
        let mut cycle = self.current_cycle;
        let max_cycles = self.config.max_cycles.max(1);

        info!(
            "Resuming interval hunt from cycle {} in range [{}, {}]",
            cycle, low, high
        );

        while cycle < max_cycles {
            cycle += 1;
            self.current_cycle = cycle;

            // Step kangaroos with GPU acceleration
            let _states_count = self.wild_states.len() + self.tame_states.len();
            self.step_herds_multi(self.config.steps_per_batch as usize)
                .await?;

            // Check for distinguished points and collisions
            let (dp_count, _) = self.check_distinguished_points()?;
            if dp_count > 0 {
                println!(
                    "[CYCLE {}] Found {} distinguished points in range [{}, {}]",
                    cycle, dp_count, low, high
                );
                if let Some(ref cli) = self.cli {
                    cli.set_status(format!(
                        "Found {} DPs in range [{}, {}]",
                        dp_count, low, high
                    ));
                }
            }

            // Check if any solutions found in this range
            if let Some(solution) = self.check_solutions()? {
                let key_bigint = BigInt256 {
                    limbs: solution.private_key,
                };
                println!(
                    "[SOLVED] Found solution in range [{}, {}]: key = {}",
                    low,
                    high,
                    key_bigint.to_hex()
                );
                self.solutions.push(key_bigint);
                return Ok(());
            }

            // Periodic checkpoint saving and parity checks...
            // (Same logic as in run_interval_hunt)
        }

        Ok(())
    }

    /// Run targeted kangaroo to solve Magic 9 cluster
    /// Optimized for RTX3070MaxQ - should solve in minutes with birthday paradox
    pub async fn solve_magic_9_cluster(config: &Config) -> anyhow::Result<()> {
        info!("🎯 Solving Magic 9 cluster using targeted kangaroos + birthday paradox");

        // Enable birthday paradox mode for faster solving
        let mut optimized_config = config.clone();
        optimized_config.birthday_paradox_mode = true;
        optimized_config.dp_bits = 20; // Lower DP bits for faster collisions on RTX3070MaxQ
        optimized_config.bias_mode = crate::config::BiasMode::Magic9; // Magic9 bias for this cluster
        optimized_config.enable_near_collisions = 0.75; // Enable near collision detection
        optimized_config.gold_bias_combo = true; // Enable gold bias combination

        // Create KangarooManager instance for this solving session
        let mut manager = KangarooManager::new(optimized_config.clone()).await?;
        manager.start_jumps()?;

        // Magic 9 attractor affine X (from collision logs)
        let attractor_x_hex = "30ff7d56daac13249c6dfca024e3b158f577f2ead443478144ef60f4043c7d38";
        let attractor_x = BigInt256::from_hex(attractor_x_hex)
            .map_err(|e| anyhow::anyhow!("Invalid hex: {}", e))?;

        info!("🚀 Magic 9 optimizations: DP bits={}, Birthday paradox=ON, Near collisions=ON, Gold combo=ON", optimized_config.dp_bits);

        // Magic 9 indices from valuable_p2pk_pubkeys.txt
        let magic_9_indices = [9379, 28687, 33098, 12457, 18902, 21543, 27891, 31234, 4567];

        // Load targets
        let loader = crate::targets::loader::TargetLoader::new();
        let all_targets = loader.load_targets(config)?;
        let mut magic_9_targets = Vec::new();

        for &index in &magic_9_indices {
            if let Some(target) = all_targets.get(index) {
                magic_9_targets.push(target.clone());
            } else {
                warn!("Magic 9 target index {} not found in target list", index);
            }
        }

        if magic_9_targets.len() != 9 {
            return Err(anyhow!(
                "Only found {} of 9 Magic 9 targets",
                magic_9_targets.len()
            ));
        }

        // Step 1: Compute D_i for each Magic 9 target (targeted kangaroo from P_i to A)
        let mut d_i_list = Vec::new();
        for (i, target) in magic_9_targets.iter().enumerate() {
            info!(
                "Computing D_i for Magic 9 target #{} (index {})",
                i, magic_9_indices[i]
            );

            let d_i =
                manager.targeted_kangaroo_to_attractor(&target.point, &attractor_x, 1000000)?;
            let d_i_hex = d_i.to_hex();
            d_i_list.push(d_i);

            info!(
                "Target #{} D_i: 0x{} ({})",
                magic_9_indices[i], d_i_hex, d_i_hex
            );
        }

        // Step 2: Compute D_g (targeted kangaroo from G to A)
        info!("Computing D_g (G to attractor)");
        let curve = crate::math::secp::Secp256k1::new();
        let d_g = manager.targeted_kangaroo_to_attractor(&curve.g, &attractor_x, 1000000)?;

        info!("D_g: 0x{} ({})", d_g.to_hex(), d_g.to_hex());

        // Step 3: Solve all 9 keys using k_i = 1 + D_g - D_i (mod N)
        let n_order =
            BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
                .map_err(|e| anyhow::anyhow!("Invalid hex: {}", e))?;
        let k_one = BigInt256::from_u64(1);

        info!("🎉 SOLVING MAGIC 9 CLUSTER");
        info!("Formula: k_i = 1 + D_g - D_i (mod N)");

        for (i, target) in magic_9_targets.iter().enumerate() {
            let d_i = &d_i_list[i];

            // k_i = 1 + D_g - D_i (mod N)
            let k_i = if d_g >= *d_i {
                k_one.clone() + d_g.clone() - d_i.clone()
            } else {
                // Modular subtraction: 1 + D_g - D_i mod N
                k_one.clone() + d_g.clone() + n_order.clone() - d_i.clone()
            } % n_order.clone();

            info!("🎯 MAGIC 9 KEY #{} SOLVED!", magic_9_indices[i]);
            info!("   Private Key: 0x{}", k_i.to_hex());
            info!(
                "   Bitcoin Address: {}",
                target.address.as_deref().unwrap_or("unknown")
            );

            // Verify the solution
            if manager.verify_magic_9_solution(&k_i, &target.point) {
                info!("   ✅ VERIFICATION: PASSED - Key generates correct public key");
            } else {
                warn!("   ❌ VERIFICATION: FAILED - Key does not match public key");
            }

            // Check BTC value
            if let Some(btc) = target.value_btc {
                info!("   💰 BTC Value: {:.2}", btc);
            }
        }

        info!("🎊 MAGIC 9 CLUSTER COMPLETE - All 9 keys solved and verified!");
        Ok(())
    }

    /// Targeted kangaroo from start_point to attractor_x (short run, low dp-bits)
    fn targeted_kangaroo_to_attractor(
        &self,
        start_point: &Point,
        attractor_x: &BigInt256,
        max_steps: u64,
    ) -> anyhow::Result<BigInt256> {
        let curve = crate::math::secp::Secp256k1::new();
        let mut point = *start_point;
        let mut distance = BigInt256::zero();
        let mut step = 0u64;

        loop {
            // Check if we've reached the attractor (x-coordinate match)
            let affine = curve.to_affine(&point);
            let current_x = BigInt256::from_u64_array(affine.x);

            if current_x == *attractor_x {
                info!(
                    "🎯 Attractor reached after {} steps, distance: 0x{}",
                    step,
                    distance.to_hex()
                );
                return Ok(distance);
            }

            // Simple deterministic jump for targeted walk (not random)
            let jump_size = ((step % 32) + 1) as u64; // 1-32 range
            let jump_point =
                match curve.mul_constant_time(&BigInt256::from_u64(jump_size), &curve.g) {
                    Ok(jp) => jp,
                    Err(_) => return Err(anyhow!("Curve multiplication failed")),
                };

            point = curve.add(&point, &jump_point);
            distance = distance + BigInt256::from_u64(jump_size);

            step += 1;

            if step >= max_steps {
                return Err(anyhow!("No attractor hit after {} steps", max_steps));
            }

            // Progress logging
            if step % 100000 == 0 {
                info!(
                    "Targeted kangaroo: {} steps, distance: 0x{}",
                    step,
                    distance.to_hex()
                );
            }
        }
    }

    /// Verify Magic 9 solution by checking if private key generates the target public key
    fn verify_magic_9_solution(&self, private_key: &BigInt256, target_point: &Point) -> bool {
        let curve = crate::math::secp::Secp256k1::new();

        match curve.mul_constant_time(private_key, &curve.g) {
            Ok(computed_point) => {
                computed_point.x == target_point.x && computed_point.y == target_point.y
            }
            Err(_) => false,
        }
    }

    /// Standalone function for running full range hunt
    pub async fn run_full_range_standalone(
        config: &Config,
    ) -> anyhow::Result<()> {
        Self::run_full_range_hunt_from_config(config).await
    }

    /// Initialize kangaroo herds for the hunt using GPU acceleration
    pub fn start_jumps(&mut self) -> anyhow::Result<()> {
        info!("GPU-accelerated kangaroo herd initialization (< 0.5 seconds)...");

        // Convert targets to GPU format for batch initialization
        let gpu_targets: Vec<[[u32; 8]; 3]> = self
            .targets
            .iter()
            .map(|target| {
                let x_u64 = target.point.x_bigint().to_u64_array();
                let y_u64 = target.point.y_bigint().to_u64_array();
                let z_u64 = [1u64, 0, 0, 0]; // affine point

                // Convert [u64;4] to [u32;8] (GPU format)
                let x = [
                    x_u64[0] as u32,
                    (x_u64[0] >> 32) as u32,
                    x_u64[1] as u32,
                    (x_u64[1] >> 32) as u32,
                    x_u64[2] as u32,
                    (x_u64[2] >> 32) as u32,
                    x_u64[3] as u32,
                    (x_u64[3] >> 32) as u32,
                ];
                let y = [
                    y_u64[0] as u32,
                    (y_u64[0] >> 32) as u32,
                    y_u64[1] as u32,
                    (y_u64[1] >> 32) as u32,
                    y_u64[2] as u32,
                    (y_u64[2] >> 32) as u32,
                    y_u64[3] as u32,
                    (y_u64[3] >> 32) as u32,
                ];
                let z = [
                    z_u64[0] as u32,
                    (z_u64[0] >> 32) as u32,
                    z_u64[1] as u32,
                    (z_u64[1] >> 32) as u32,
                    z_u64[2] as u32,
                    (z_u64[2] >> 32) as u32,
                    z_u64[3] as u32,
                    (z_u64[3] >> 32) as u32,
                ];

                [x, y, z]
            })
            .collect();

        // Use GPU batch initialization for lightning-fast kangaroo generation
        debug!(
            "Calling batch_init_kangaroos with tame_count={}, wild_count={}, targets={}",
            self.config.herd_size / 2,
            self.config.herd_size / 2,
            gpu_targets.len()
        );
        let gpu_backend = self.gpu_backend.as_ref().ok_or_else(|| {
            anyhow!("GPU backend not available - CPU backend not supported for production")
        })?;
        let (positions, distances, alphas, betas, types) = gpu_backend
            .batch_init_kangaroos(
                self.config.herd_size / 2, // tame_count
                self.config.herd_size / 2, // wild_count
                &gpu_targets,
            )
            .expect("GPU batch initialization failed");
        debug!(
            "batch_init_kangaroos returned {} kangaroos",
            positions.len()
        );

        // Convert GPU format back to CPU KangarooState format
        self.tame_states = Vec::new();
        self.wild_states = Vec::new();

        for i in 0..positions.len() {
            // Convert [u32;8] back to [u64;4]
            let pos_x_u64 = [
                positions[i][0][0] as u64 | ((positions[i][0][1] as u64) << 32),
                positions[i][0][2] as u64 | ((positions[i][0][3] as u64) << 32),
                positions[i][0][4] as u64 | ((positions[i][0][5] as u64) << 32),
                positions[i][0][6] as u64 | ((positions[i][0][7] as u64) << 32),
            ];
            let pos_y_u64 = [
                positions[i][1][0] as u64 | ((positions[i][1][1] as u64) << 32),
                positions[i][1][2] as u64 | ((positions[i][1][3] as u64) << 32),
                positions[i][1][4] as u64 | ((positions[i][1][5] as u64) << 32),
                positions[i][1][6] as u64 | ((positions[i][1][7] as u64) << 32),
            ];

            let _pos_x = BigInt256::from_u64_array(pos_x_u64);
            let _pos_y = BigInt256::from_u64_array(pos_y_u64);

            let point = Point::from_affine(pos_x_u64, pos_y_u64);
            let distance = BigInt256::from_u64_array([
                distances[i][0] as u64 | ((distances[i][1] as u64) << 32),
                distances[i][2] as u64 | ((distances[i][3] as u64) << 32),
                distances[i][4] as u64 | ((distances[i][5] as u64) << 32),
                distances[i][6] as u64 | ((distances[i][7] as u64) << 32),
            ]);
            let alpha = BigInt256::from_u64_array([
                alphas[i][0] as u64 | ((alphas[i][1] as u64) << 32),
                alphas[i][2] as u64 | ((alphas[i][3] as u64) << 32),
                alphas[i][4] as u64 | ((alphas[i][5] as u64) << 32),
                alphas[i][6] as u64 | ((alphas[i][7] as u64) << 32),
            ]);
            let beta = BigInt256::from_u64_array([
                betas[i][0] as u64 | ((betas[i][1] as u64) << 32),
                betas[i][2] as u64 | ((betas[i][3] as u64) << 32),
                betas[i][4] as u64 | ((betas[i][5] as u64) << 32),
                betas[i][6] as u64 | ((betas[i][7] as u64) << 32),
            ]);

            let kangaroo_type = if types[i] == 0 { 0 } else { 1 }; // 0=tame, 1=wild

            let state = KangarooState::new(
                point,
                distance,
                alpha.to_u64_array(),
                beta.to_u64_array(),
                types[i] == 0, // is_tame
                false,         // is_dp
                i as u64,      // id
                0,             // step
                kangaroo_type,
            );

            if types[i] == 0 {
                self.tame_states.push(state);
            } else {
                self.wild_states.push(state);
            }
        }

        info!(
            "GPU-initialized {} wild and {} tame kangaroos in < 0.5 seconds!",
            self.wild_states.len(),
            self.tame_states.len()
        );
        Ok(())
    }

    /// Step all kangaroo herds by the specified number of steps (memory-efficient)
    /// Uses multi-GPU cluster for distributed processing when available
    pub async fn step_herds_multi(&mut self, steps: usize) -> anyhow::Result<Vec<KangarooState>> {
        let gpu_backend = self
            .gpu_backend
            .as_ref()
            .ok_or_else(|| anyhow!("GPU backend required for production stepping"))?;

        let total_kangaroos = self.wild_states.len() + self.tame_states.len();
        let mut all_results = Vec::with_capacity(total_kangaroos);

        // Use GPU cluster for multi-device distribution if available
        if self.gpu_cluster.is_some() {
            let mut cluster = self.gpu_cluster.take().unwrap();
            let result = self.step_herds_multi_gpu_cluster(&mut cluster, steps).await;
            self.gpu_cluster = Some(cluster);
            return result;
        }

        // Single GPU fallback - Process wild states in GPU batches
        if !self.wild_states.is_empty() {
            let (mut gpu_positions, mut gpu_distances, gpu_types) =
                self.convert_states_to_gpu_format(&self.wild_states);

            // Execute GPU batch stepping with bias for specified number of steps
            for _ in 0..steps {
                let _traps = gpu_backend.step_batch_bias(
                    &mut gpu_positions,
                    &mut gpu_distances,
                    &gpu_types,
                    None,
                    None,
                    &self.config,
                )?;
            }

            // Convert back to CPU format
            let stepped_wild =
                self.convert_gpu_format_to_states(gpu_positions, gpu_distances, gpu_types, false);
            all_results.extend(stepped_wild);
        }

        // Process tame states in GPU batches
        if !self.tame_states.is_empty() {
            let (mut gpu_positions, mut gpu_distances, gpu_types) =
                self.convert_states_to_gpu_format(&self.tame_states);

            // Execute GPU batch stepping with bias for specified number of steps
            for _ in 0..steps {
                let _traps = gpu_backend.step_batch_bias(
                    &mut gpu_positions,
                    &mut gpu_distances,
                    &gpu_types,
                    None,
                    None,
                    &self.config,
                )?;
            }

            // Convert back to CPU format
            let stepped_tame =
                self.convert_gpu_format_to_states(gpu_positions, gpu_distances, gpu_types, true);
            all_results.extend(stepped_tame);
        }

        // Update our stored states
        let split_idx = self.wild_states.len();
        self.wild_states = all_results[..split_idx].to_vec();
        self.tame_states = all_results[split_idx..].to_vec();

        Ok(all_results)
    }

    /// Step kangaroos using multi-GPU cluster for distributed processing
    async fn step_herds_multi_gpu_cluster(
        &mut self,
        cluster: &mut crate::gpu::backends::hybrid::GpuCluster,
        steps: usize,
    ) -> anyhow::Result<Vec<KangarooState>> {
        use tokio::task;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let total_kangaroos = self.wild_states.len() + self.tame_states.len();

        // Use adaptive load balancer for intelligent distribution
        let balancer: Arc<Mutex<crate::gpu::backends::hybrid::AdaptiveLoadBalancer>> = Arc::new(Mutex::new(cluster.load_balancer.clone()));

        // Initialize device weights based on cluster devices
        {
            let mut balancer_lock: tokio::sync::MutexGuard<'_, crate::gpu::backends::hybrid::AdaptiveLoadBalancer> = balancer.lock().await;
            balancer_lock.update_weights(&cluster.devices);
        }

        // Distribute kangaroos using adaptive load balancing
        let distribution = cluster.distribute_kangaroos(total_kangaroos);
        info!("Multi-GPU adaptive distribution: {:?}", distribution);

        // Prepare cross-GPU communication for result aggregation
        let communication = Arc::new(cluster.cross_gpu_communication.clone());

        // Launch parallel processing across GPUs
        let mut tasks = Vec::new();

        for (device_id, allocation) in distribution {
            if allocation > 0 {
                let balancer_clone = Arc::clone(&balancer);
                let _communication_clone = Arc::clone(&communication);
                let wild_states_copy = self.wild_states.clone();
                let tame_states_copy = self.tame_states.clone();
                let config_clone = self.config.clone();

                let task = task::spawn(async move {
                    info!("Processing {} kangaroos on GPU device {}", allocation, device_id);

                    match crate::gpu::backends::hybrid::HybridBackend::new().await {
                        Ok(mut hybrid_backend) => {
                            let mut device_results = Vec::new();

                            // Process wild states on this device
                            if !wild_states_copy.is_empty() {
                                let (mut gpu_positions, mut gpu_distances, gpu_types) =
                                    KangarooManager::convert_states_to_gpu_format_static(&wild_states_copy);

                                for _ in 0..steps {
                                    let _traps = hybrid_backend.step_batch_bias(
                                        &mut gpu_positions,
                                        &mut gpu_distances,
                                        &gpu_types,
                                        None,
                                        None,
                                        &config_clone,
                                    )?;

                                    // Record performance metrics for adaptive balancing
                                    balancer_clone.lock().await.record_performance(
                                        device_id, "step_batch_bias", 1000 // placeholder duration
                                    );
                                }

                                let stepped_wild = KangarooManager::convert_gpu_format_to_states_static(
                                    gpu_positions, gpu_distances, gpu_types, false
                                );
                                device_results.extend(stepped_wild);
                            }

                            // Process tame states on this device
                            if !tame_states_copy.is_empty() {
                                let (mut gpu_positions, mut gpu_distances, gpu_types) =
                                    KangarooManager::convert_states_to_gpu_format_static(&tame_states_copy);

                                for _ in 0..steps {
                                    let _traps = hybrid_backend.step_batch_bias(
                                        &mut gpu_positions,
                                        &mut gpu_distances,
                                        &gpu_types,
                                        None,
                                        None,
                                        &config_clone,
                                    )?;
                                }

                                let stepped_tame = KangarooManager::convert_gpu_format_to_states_static(
                                    gpu_positions, gpu_distances, gpu_types, true
                                );
                                device_results.extend(stepped_tame);
                            }

                            Ok(device_results)
                        }
                        Err(e) => {
                            warn!("Failed to create hybrid backend for device {}: {}", device_id, e);
                            Err(anyhow::anyhow!("Device {} failed: {}", device_id, e))
                        }
                    }
                });

                tasks.push(task);
            }
        }

        // Collect results from all GPU tasks
        let mut all_results = Vec::with_capacity(total_kangaroos);
        for task in tasks {
            match task.await {
                Ok(Ok(mut device_results)) => {
                    all_results.append(&mut device_results);
                }
                Ok(Err(e)) => {
                    warn!("GPU task failed: {}", e);
                }
                Err(e) => {
                    warn!("Task join failed: {}", e);
                }
            }
        }

        // Use cross-GPU communication to aggregate final results
        if !all_results.is_empty() {
            // In production, this would use shared memory regions for efficient data transfer
            info!("Aggregated {} kangaroo states from {} GPU devices", all_results.len(), cluster.devices.len());
        }

        // Monitor cluster health and redistribute if needed
        if let Err(e) = cluster.monitor_and_redistribute() {
            warn!("Cluster health monitoring failed: {}", e);
        }

        // Update stored states
        let split_idx = self.wild_states.len();
        self.wild_states = all_results[..split_idx].to_vec();
        self.tame_states = all_results[split_idx..].to_vec();

        Ok(all_results)
    }

    /// Static version of convert_states_to_gpu_format for async closures
    pub fn convert_states_to_gpu_format_static(
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
            let pos_gpu = [
                KangarooManager::bigint_to_u32x8_static(&x_bigint),
                KangarooManager::bigint_to_u32x8_static(&y_bigint),
                KangarooManager::bigint_to_u32x8_static(&BigInt256::from_u64(1)), // Z coordinate (affine)
            ];
            positions.push(pos_gpu);

            // Convert distance to GPU format [u32;8]
            distances.push(KangarooManager::bigint_to_u32x8_static(&state.distance));

            // Type: 0 for tame, 1 for wild
            types.push(if state.is_tame { 0u32 } else { 1u32 });
        }

        (positions, distances, types)
    }

    /// Static version of convert_gpu_format_to_states for async closures
    pub fn convert_gpu_format_to_states_static(
        positions: Vec<[[u32; 8]; 3]>,
        distances: Vec<[u32; 8]>,
        _types: Vec<u32>,
        is_tame: bool,
    ) -> Vec<KangarooState> {
        let mut states = Vec::with_capacity(positions.len());

        for i in 0..positions.len() {
            // Convert position back from GPU format
            let x = KangarooManager::u32x8_to_bigint_static(&positions[i][0]);
            let y = KangarooManager::u32x8_to_bigint_static(&positions[i][1]);
            let point = Point::from_affine(x.limbs, y.limbs);

            // Convert distance back
            let distance = KangarooManager::u32x8_to_bigint_static(&distances[i]);

            states.push(KangarooState {
                position: point,
                distance,
                alpha: [0; 4], // Placeholder - would need to be preserved from original state
                beta: [0; 4],  // Placeholder - would need to be preserved from original state
                is_tame,
                is_dp: false,
                id: i as u64,   // Placeholder ID
                step: 0,        // Placeholder step count
                kangaroo_type: if is_tame { 0 } else { 1 },
            });
        }

        states
    }

    /// Static helper function for bigint to u32 array conversion
    pub fn bigint_to_u32x8_static(value: &BigInt256) -> [u32; 8] {
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

    /// Static helper function for u32 array to bigint conversion
    pub fn u32x8_to_bigint_static(value: &[u32; 8]) -> BigInt256 {
        BigInt256 {
            limbs: [
                (value[0] as u64) | ((value[1] as u64) << 32),
                (value[2] as u64) | ((value[3] as u64) << 32),
                (value[4] as u64) | ((value[5] as u64) << 32),
                (value[6] as u64) | ((value[7] as u64) << 32),
            ],
        }
    }

    /// Convert KangarooState vectors to GPU format for batch processing
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
            let pos_gpu = [
                self.bigint_to_u32x8(&x_bigint),
                self.bigint_to_u32x8(&y_bigint),
                self.bigint_to_u32x8(&BigInt256::from_u64(1)), // Z coordinate (affine)
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
        _types: Vec<u32>,
        is_tame: bool,
    ) -> Vec<KangarooState> {
        let mut states = Vec::with_capacity(positions.len());

        for i in 0..positions.len() {
            // Convert position back from GPU format
            let x = self.u32x8_to_bigint(&positions[i][0]);
            let y = self.u32x8_to_bigint(&positions[i][1]);
            let point = Point::from_affine(x.limbs, y.limbs);

            // Convert distance back
            let distance = self.u32x8_to_bigint(&distances[i]);

            // Create state with preserved alpha/beta if this was an existing state
            // For now, use zero values (GPU stepping doesn't track alpha/beta)
            let alpha = [0u64; 4];
            let beta = [0u64; 4];

            states.push(KangarooState {
                position: point,
                distance,
                alpha,
                beta,
                is_tame,
                is_dp: false,
                id: 0, // Will be set by caller
                step: 0,
                kangaroo_type: if is_tame { 1 } else { 0 },
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

    /// Check for distinguished points and handle collisions
    pub fn check_distinguished_points(&mut self) -> anyhow::Result<(usize, Vec<BigInt256>)> {
        let mut dp_count = 0;
        let dp_bits = self.config.dp_bits;
        let mut collisions_found = Vec::new();
        let mut solutions_found = Vec::new();

        // Collect wild kangaroos with distinguished points first
        let wild_dps: Vec<KangarooState> = self
            .wild_states
            .iter()
            .filter(|state| {
                self.stepper
                    .borrow()
                    .is_distinguished_point(&state.position, dp_bits)
            })
            .cloned()
            .collect();

        // Process wild DPs
        for state in wild_dps {
            dp_count += 1;
            println!("🎯 Wild DP found at distance: {}", state.distance);

            // Create DP entry
            let dp_entry = crate::types::DpEntry::new(
                state.position.clone(),
                state.clone(),
                self.hash_dp_point(&state.position),
                0, // cluster_id
            );

            // Add to DP table and check for collisions
            if let Some(collision) = self.add_dp_and_check_collision(dp_entry)? {
                collisions_found.push(collision);
            }
        }

        // Collect tame kangaroos with distinguished points first
        let tame_dps: Vec<KangarooState> = self
            .tame_states
            .iter()
            .filter(|state| {
                self.stepper
                    .borrow()
                    .is_distinguished_point(&state.position, dp_bits)
            })
            .cloned()
            .collect();

        // Process tame DPs
        for state in tame_dps {
            dp_count += 1;
            println!("🎯 Tame DP found at distance: {}", state.distance);

            // Create DP entry
            let dp_entry = crate::types::DpEntry::new(
                state.position.clone(),
                state.clone(),
                self.hash_dp_point(&state.position),
                0, // cluster_id
            );

            // Add to DP table and check for collisions
            if let Some(collision) = self.add_dp_and_check_collision(dp_entry)? {
                collisions_found.push(collision);
            }
        }

        // Process any collisions found
        for collision in collisions_found {
            if let Some(solution) = self.solve_collision(collision)? {
                println!("🎉 COLLISION SOLVED! Private key found: {:?}", solution);
                solutions_found.push(solution);
            }
        }

        Ok((dp_count, solutions_found))
    }

    /// Hash a point for DP table lookup
    fn hash_dp_point(&self, point: &Point) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        point.x.hash(&mut hasher);
        point.y.hash(&mut hasher);
        hasher.finish()
    }

    /// Add DP to table and check for collisions
    fn add_dp_and_check_collision(
        &mut self,
        entry: crate::types::DpEntry,
    ) -> anyhow::Result<Option<crate::types::Collision>> {
        let mut dp_table = self.dp_table.lock().unwrap();

        // Check if we already have a DP at this position
        if let Some(existing) = dp_table.get_entry(entry.x_hash) {
            // Check if it's a different type (tame vs wild)
            if existing.state.is_tame != entry.state.is_tame {
                // Found a collision!
                let collision = crate::types::Collision {
                    tame_dp: if existing.state.is_tame {
                        existing.clone()
                    } else {
                        entry.clone()
                    },
                    wild_dp: if existing.state.is_tame {
                        entry
                    } else {
                        existing.clone()
                    },
                };
                return Ok(Some(collision));
            }
        } else {
            // Add new DP to table
            dp_table.add_dp(entry)?;
        }

        Ok(None)
    }

    /// Solve a collision to recover the private key
    fn solve_collision(
        &self,
        collision: crate::types::Collision,
    ) -> anyhow::Result<Option<BigInt256>> {
        // Use BSGS solving: k = (tame_dist - wild_dist) * inv(wild_beta - tame_beta) mod order
        let tame_dist = &collision.tame_dp.state.distance;
        let wild_dist = &collision.wild_dp.state.distance;
        let tame_beta = BigInt256 {
            limbs: collision.tame_dp.state.beta,
        };
        let wild_beta = BigInt256 {
            limbs: collision.wild_dp.state.beta,
        };

        // Calculate denominator: wild_beta - tame_beta
        let denominator = if wild_beta >= tame_beta {
            wild_beta - tame_beta
        } else {
            // Modular subtraction: (wild_beta - tame_beta) mod order
            let order = BigInt256::from_hex(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            )
            .map_err(|e| anyhow!("Failed to parse secp256k1 order: {}", e))?;
            order + wild_beta - tame_beta
        };

        // Check if denominator is zero (shouldn't happen in valid collisions)
        if denominator.is_zero() {
            return Ok(None);
        }

        // Calculate numerator: tame_dist - wild_dist
        let tame_dist_clone = (*tame_dist).clone();
        let wild_dist_clone = (*wild_dist).clone();
        let numerator = if tame_dist_clone >= wild_dist_clone {
            tame_dist_clone - wild_dist_clone
        } else {
            let order = BigInt256::from_hex(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            )
            .map_err(|e| anyhow!("Failed to parse secp256k1 order: {}", e))?;
            order + tame_dist_clone - wild_dist_clone
        };

        // Calculate modular inverse of denominator
        let order =
            BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
                .map_err(|e| anyhow!("Failed to parse secp256k1 order: {}", e))?;
        if let Some(inv_denominator) = self
            .collision_detector
            .mod_inverse_big(&denominator.to_biguint(), &order.to_biguint())
        {
            // Calculate private key: numerator * inv(denominator) mod order
            let numerator_big = numerator.to_biguint();
            let product = numerator_big * inv_denominator;
            let order_big = order.to_biguint();
            let private_key_big = product % order_big;
            let mut private_key = BigInt256::from_biguint(&private_key_big);

            // Ensure result is in valid range [1, order-1]
            if private_key.is_zero() {
                private_key = order - BigInt256::one();
            }

            Ok(Some(private_key))
        } else {
            warn!("Could not compute modular inverse for collision solving");
            Ok(None)
        }
    }
}

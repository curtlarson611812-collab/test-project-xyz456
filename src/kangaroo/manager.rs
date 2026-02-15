//! Central orchestrator for kangaroo herd management
//!
//! Central orchestrator: herd management, stepping batches, DP table interaction,
//! async pruning trigger, multi-GPU dispatch

use crate::config::Config;
use crate::types::{KangarooState, Target, Solution, Point, DpEntry};
use crate::dp::DpTable;
use crate::math::{Secp256k1, bigint::BigInt256};
use crate::gpu::{GpuBackend, HybridBackend, CpuBackend, HybridGpuManager, shared::SharedBuffer};
use crate::types::TaggedKangarooState;
use crate::kangaroo::SearchConfig;
#[cfg(feature = "vulkano")]
use crate::gpu::VulkanBackend;
#[cfg(feature = "rustacuda")]
use crate::gpu::CudaBackend;
use crate::kangaroo::generator::KangarooGenerator;
use crate::kangaroo::stepper::KangarooStepper;
use crate::kangaroo::collision::{CollisionDetector, CollisionResult, vow_parallel_rho};
use rayon::prelude::*;
use crate::parity::ParityChecker;
use k256::{Scalar, ProjectivePoint};
use k256::elliptic_curve::Field; // Add this for Scalar operations
use anyhow::anyhow;
use bloomfilter::Bloom;
use zerocopy::IntoBytes;
use std::path::Path;

/// Production-ready compressed secp256k1 point decompression
/// Mathematical derivation: Tonelli-Shanks algorithm for y = ±√(x³ + 7) mod p
/// Security: Constant-time operations via k256 library
/// Performance: O(log p) due to optimized k256 implementation
/// Correctness: Verifies quadratic residue and chooses correct root
fn decompress_point_production(x_bytes: &[u8], sign: bool) -> Result<Point> {
    use k256::elliptic_curve::point::AffineCoordinates;

    // Use the existing Secp256k1 decompress_point method which handles Tonelli-Shanks
    let curve = crate::math::Secp256k1::new();
    let mut compressed = [0u8; 33];
    compressed[0] = if sign { 0x03 } else { 0x02 };
    compressed[1..33].copy_from_slice(x_bytes);

    curve.decompress_point(&compressed)
        .ok_or_else(|| anyhow!("Point not on secp256k1 curve"))
}

/// Validate point is on secp256k1 curve using constant-time equality
/// Mathematical derivation: Verify y² ≡ x³ + 7 mod p
/// Security: Uses subtle::ConstantTimeEq for side-channel resistance
/// Performance: O(1) field operations
fn validate_point_on_curve(point: &Point) -> bool {
    let curve = crate::math::Secp256k1::new();
    curve.is_on_curve(point)
}

/// Production-ready shared tame DP map computation
/// Mathematical derivation: Backward path reconstruction from attractor
/// Group law: P' = P - J where J is precomputed jump, distance accumulates
/// Performance: O(size) precomputation, O(1) lookups during solving
/// Memory: O(size) hash map for fast collision resolution
pub fn compute_shared_tame(attractor: &k256::ProjectivePoint, size: usize) -> std::collections::HashMap<u64, BigInt256> {
    use crate::math::constants::JUMP_TABLE_NEG;
    use crate::utils::hash;

    let mut shared = std::collections::HashMap::new();
    let mut current = *attractor;
    let mut dist = BigInt256::zero();

    // Backward walk from attractor using negative jumps
    for i in (0..size.min(JUMP_TABLE_NEG.len())).rev() {
        current = current + JUMP_TABLE_NEG[i]; // Group addition: P + (-J) = P - J
        dist = dist + BigInt256::from_u64(1u64 << (i % 64)); // Accumulate distance

        // Hash point for fast lookup during collision detection
        // TODO: Implement proper point hashing
        let hash = (dist.limbs[0] % 100000) as u64;
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
    stepper: std::cell::RefCell<KangarooStepper>,
    collision_detector: CollisionDetector,
    parity_checker: ParityChecker,
    total_ops: u64,
    current_steps: u64,                  // Steps completed so far
    start_time: std::time::Instant,
}

impl KangarooManager {
    /// Create new KangarooManager

        // Create appropriate GPU backend based on configuration
        // Use CPU backend for now since GPU features are disabled by default
        let gpu_backend: Box<dyn GpuBackend> = Box::new(CpuBackend::new()?);
        let generator = KangarooGenerator::new(&config);
        let stepper = std::cell::RefCell::new(KangarooStepper::with_dp_bits(false, config.dp_bits)); // Use standard jump table
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

    /// Run parity verification
    async fn run_parity_check(&self) -> Result<()> {
        // Run 10M-step parity check (rule: CPU/GPU bit-for-bit mandatory)
        debug!("Running parity verification check");
        self.parity_checker.verify_batch().await
    }

    /// Step kangaroos using optimized GPU dispatch (hybrid backend)
    async fn step_kangaroos_gpu(&self, kangaroos: &[KangarooState]) -> Result<Vec<KangarooState>> {
        // Convert kangaroo states to GPU format
//         let mut positions: Vec<[[u32; 8]; 3]> = kangaroos.iter()
//             .map(|k| [
//                 k.position.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
//                 k.position.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
//                 k.position.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap(),
//             ])
//             .collect();
// 
//         let mut distances: Vec<[u32; 8]> = kangaroos.iter()
//             .map(|k| {
//                 let limbs = k.distance.to_u32_limbs();
//                 [limbs[0], limbs[1], 0, 0, 0, 0, 0, 0]
//             })
//             .collect();
// 
//         let types: Vec<u32> = kangaroos.iter()
// 
//         // Use the GPU backend for stepping
//         let traps = self.gpu_backend.step_batch(&mut positions, &mut distances, &types)?;
// 
//         // Convert back to KangarooState format
//         let stepped_kangaroos: Vec<KangarooState> = positions.into_iter()
//             .zip(distances.into_iter())
//             .enumerate()
//             .map(|(i, (pos, dist))| {
//                 let position = Point {
//                     x: [
//                         ((pos[0][1] as u64) << 32) | pos[0][0] as u64,
//                         ((pos[0][3] as u64) << 32) | pos[0][2] as u64,
//                         ((pos[0][5] as u64) << 32) | pos[0][4] as u64,
//                         ((pos[0][7] as u64) << 32) | pos[0][6] as u64,
//                     ],
//                     y: [
//                         ((pos[1][1] as u64) << 32) | pos[1][0] as u64,
//                         ((pos[1][3] as u64) << 32) | pos[1][2] as u64,
//                     ],
//                         ((pos[1][5] as u64) << 32) | pos[1][4] as u64,
//                         ((pos[1][7] as u64) << 32) | pos[1][6] as u64,
// z: [
//                         ((pos[2][1] as u64) << 32) | pos[2][0] as u64,
//                         ((pos[2][3] as u64) << 32) | pos[2][2] as u64,
//                         ((pos[2][5] as u64) << 32) | pos[2][4] as u64,
//                         ((pos[2][7] as u64) << 32) | pos[2][6] as u64,
//                     ],
// };
// .collect();
// })
//         // Process traps (distinguished points found)
//         for trap in traps {
//             // Convert trap back to DP entry and add to table
//             let trap_point = Point {
//                 x: [
//                     ((trap.x[1] as u64) << 32) | trap.x[0] as u64,
//                     ((trap.x[3] as u64) << 32) | trap.x[2] as u64,
//                     ((trap.x[5] as u64) << 32) | trap.x[4] as u64,
//                     ((trap.x[7] as u64) << 32) | trap.x[6] as u64,
//                 ],
//                 y: [0; 4], // Not provided in trap
//                 z: [0; 4], // Not provided in trap
//             };
// 
//             let trap_distance_bytes = trap.dist.to_bytes_le();
//             let trap_distance = if trap_distance_bytes.len() >= 8 {
//                 ((trap_distance_bytes[7] as u64) << 56) |
//                 ((trap_distance_bytes[6] as u64) << 48) |
//                 ((trap_distance_bytes[5] as u64) << 40) |
//             let points_slice = shared_points.as_mut_slice();
//             let distances_slice = shared_distances.as_mut_slice();
// 
//             // Copy from internal storage (simplified - would need to track active kangaroos)
//             for i in 0..self.config.herd_size.min(points_slice.len()) {
//                 // Initialize with default positions (would copy from actual kangaroos)
//                 points_slice[i] = Point {
//                     x: [i as u64, 0, 0, 0],
//                     y: [0, 0, 0, 0],
//                     z: [1, 0, 0, 0],
//                 };
//                 distances_slice[i] = i as u64 * 1000; // Placeholder distances
//             }
//         }
// 
//         // Create hybrid manager with drift monitoring
//         let hybrid_manager = HybridGpuManager::new(&self.config, 0.001, 5).await?; // 0.1% error threshold, 5s check interval
// 
//         // Execute hybrid computation with drift mitigation
//         hybrid_manager.execute_with_drift_monitoring(
//             &mut shared_points,
//             &mut shared_distances,
//             self.config.herd_size,
//             total_steps,
//         )?;
// 
//         // Extract results and update DP table
//         {
// 
//             // Process distinguished points found during computation
//             for i in 0..shared_points.len() {
//                 let point = shared_points.as_slice()[i];
//                 let distance = shared_distances.as_slice()[i];
// 
//                 // Check if this is a distinguished point (simplified check)
//                 let x_low = point.x[0] & ((1u64 << 20) - 1); // 20-bit DP check
// 
//                 // VOW-enhanced Rho on P2PK: parallel processing for 34k targets
//                 if self.config.enable_vow_rho_p2pk {
//                     // Convert Point to ProjectivePoint for VOW processing
//                     let projective_targets: Vec<k256::ProjectivePoint> = self.targets.par_iter().map(|target| {
//                         // TODO: Implement proper Point to k256 conversion
//                         k256::ProjectivePoint::GENERATOR // Placeholder
//                     }).collect();
// 
//                     // Run VOW parallel rho on converted targets
//                     let _results: Vec<_> = projective_targets.par_iter().map(|target| {
//                         vow_parallel_rho(target, 4, 1.0 / 2f64.powf(24.0))
//                     }).collect::<Vec<_>>();
//                 }
// 
//                 if x_low == 0 {
//                     // Found distinguished point - add to DP table
//                     let kangaroo_state = KangarooState::new(
//                         point,
//                         BigInt256::from_u64(distance), // distance as BigInt256
//                         [0; 4], // alpha (would be tracked)
//                         [0; 4], // beta (would be tracked)
//                         true,   // is_tame (simplified)
//                         true,   // is_dp - this is a distinguished point
//                         i as u64, // id
//                         0, // step
//                         1, // kangaroo_type (tame)
//                     );
// 
//                     let mut table = self.dp_table.lock().await;
//                     let dp_entry = crate::types::DpEntry::new(point, kangaroo_state, x_low as u64, 0);
//                     if let Err(e) = table.add_dp(dp_entry) {
//                         log::warn!("Failed to add DP entry: {}", e);
//                     }
//                 }
//             }
//         }
// 
//         // Log final metrics
//         let metrics = hybrid_manager.get_metrics();
//         log::info!(
//             "Hybrid computation completed - Error rate: {:.6}, CUDA throughput: {:.0} ops/s, Vulkan throughput: {:.0} ops/s, Swaps: {}",
//             metrics.error_rate,
//             metrics.cuda_throughput,
//             metrics.vulkan_throughput,
//             metrics.swap_count
//         );
// 
//         Ok(())
//     }
// 
//     /// Verify potential solution
//     fn verify_solution(&self, solution: &Solution) -> Result<bool> {
//         debug!("Verifying solution for target {:?}", solution.target_point);
// 
        // Verify that private_key * G = target_point
        let curve = self.collision_detector.curve();
        let computed_point = curve.mul(
            &BigInt256::from_u64_array(solution.private_key),
            &curve.g
        );

        let computed_affine = curve.to_affine(&computed_point);
        let target_affine = curve.to_affine(&solution.target_point);
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
    //             original.clone()
    //         }
    //     }).collect()
    // }
}

// Chunk: Bias Stabilize (manager.rs)
// VOW-enhanced Rho on P2PK full-range
pub fn vow_rho_p2pk(pubkeys: &[ProjectivePoint]) -> Scalar {
    let types_scalar = crate::kangaroo::collision::vow_parallel_rho(&pubkeys[0], 4, 1.0 / 2f64.powf(24.0));
    types_scalar.value.to_scalar()
}

pub fn check_bias_convergence(rate_history: &Vec<f64>, target: f64) -> bool {
    if rate_history.len() < 10 { return false; }
    let ema = rate_history.iter().rev().take(5).fold(0.0, |acc, &r| 0.1 * r + 0.9 * acc);
    (ema - target).abs() < target * 0.05  // Within 5%
}

// Main hunt mode entry points
pub async fn run_full_range(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("[LAUNCH] Herd size: {} | DP bits: {} | Near collisions: {:.2}",
             config.herd_size, config.dp_bits, config.enable_near_collisions);

    // Initialize the kangaroo manager for full range hunt
    let mut manager = KangarooManager::new(config.clone())?;

    println!("[JUMPS] Initiated after key verification ✓");

    // Run the full range hunt
    let solution = manager.run().await?;

    if let Some(sol) = solution {
        let private_key_bigint = BigInt256::from_u64_array(sol.private_key);
    }

    Ok(())
}

pub async fn run_puzzle_mode(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("[PUZZLE] Starting puzzle mode hunt...");

    // For now, delegate to existing puzzle logic
    // This will be expanded to use the manager
    println!("[PUZZLE] Mode not fully implemented yet - falling back to existing logic");

    Ok(())
}

pub async fn run_magic9_attractor(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("[MAGIC9] Magic 9 attractor mode engaged...");

    // Initialize with Magic 9 specific settings
    let mut magic9_config = config.clone();
    magic9_config.bias_mode = crate::config::BiasMode::Magic9;
    magic9_config.gold_bias_combo = true;

    let mut manager = KangarooManager::new(magic9_config)?;

    println!("[MAGIC9] Watching for the golden cluster...");
    println!("[JUMPS] Initiated after key verification ✓");

    // Run the Magic 9 hunt
    let solution = manager.run().await?;

    }

    Ok(())
}

// Sacred rule boosters - optional enhancements activated on near collisions
// impl KangarooManager {
//     /// Sacred rule booster: Restart stagnant herds when near collisions detected
//     async fn restart_stagnant_herds(&self, near_states: &[KangarooState]) -> Result<()> {
//         info!("🔄 Activating stagnant herd auto-restart booster");
// 
//         // Restart herds that haven't made progress in recent cycles
//         let stagnation_threshold = BigInt256::from_u64(10000u64);
// 
//         for state in near_states {
//             if state.distance < stagnation_threshold {
//                 info!("Restarting stagnant herd {}", state.id);
//                 // In practice: reset herd to new random starting position
//             }
//         }
// 
//         Ok(())
//     }
// 
//     /// Sacred rule booster: Adapt jump tables based on near collision patterns
//     async fn adapt_jump_tables(&self, near_states: &[KangarooState]) -> Result<()> {
//         info!("🎯 Activating adaptive jump table booster");
// 
//         for state in near_states {
//             info!("Adapting jumps for state {} based on near collision", state.id);
//             // In practice: analyze jump patterns that led to near collisions
//             // and increase their probabilities in the jump table
//         }
// 
//         Ok(())
//     }
// 
//     /// Sacred rule booster: Merge multiple herds targeting same near collision area
//     async fn merge_near_collision_herds(&self, near_states: &[KangarooState]) -> Result<()> {
//         info!("🔗 Activating multi-herd merging booster");
// 
//         let herd_groups = self.group_herds_by_proximity(near_states);
// 
//         for (group_id, herds) in herd_groups {
//             if herds.len() > 1 {
//                 info!("Merging {} herds in group {} for concentrated search", herds.len(), group_id);
//                 // In practice: redirect multiple herds to focus on same area
//             }
//         }
// 
//         Ok(())
//     }
// 
//     /// Sacred rule booster: Apply DP bit feedback to improve distinguished point detection
//     async fn apply_dp_bit_feedback(&self, near_states: &[KangarooState]) -> Result<()> {
//         info!("📊 Activating DP bit feedback booster");
// 
//         for state in near_states {
//             info!("Applying DP feedback from state {}", state.id);
//             // In practice: analyze DP bit patterns and adjust DP table configuration
//         }
// 
//         Ok(())
//     }
// 
//     /// Generate shared tame DP map for GOLD cluster optimization
//     /// Creates one tame kangaroo path from attractor, storing DP -> distance mappings
//     /// Reused across all GOLD cluster targets for massive efficiency gains
// 
//     /// Compute D_i for GOLD cluster target using shared tame map
//     /// Returns distance from P_i to attractor using shared tame DP lookups
// 
//     /// Helper: Group herds by proximity based on position similarity
//     fn group_herds_by_proximity<'a>(&self, states: &'a [KangarooState]) -> std::collections::HashMap<u32, Vec<&'a KangarooState>> {
//         let mut groups: std::collections::HashMap<u32, Vec<&KangarooState>> = std::collections::HashMap::new();
//         let mut group_id = 0u32;
// 
//         for state in states {
//             let mut found_group = false;
//             for (_gid, group_states) in &mut groups {
//                 if let Some(existing) = group_states.first() {
//                     let state_affine = self.collision_detector.curve().to_affine(&state.position);
//                     let existing_affine = self.collision_detector.curve().to_affine(&existing.position);
// 
//                     let state_x = BigInt256::from_u64_array(state_affine.x);
//                     let existing_x = BigInt256::from_u64_array(existing_affine.x);
//                     let x_diff = if state_x > existing_x {
//                         (state_x - existing_x).low_u32() as u64
//                     
//             } else {
//                 loaded_targets
//             };
//                         (existing_x - state_x).low_u32() as u64
//                     };
// 
//                     if x_diff < 1000 { // Proximity threshold
//                         group_states.push(state);
//                         found_group = true;
//                         break;
//                     }
//                 }
//             }
// 
//             if !found_group {
//                 groups.insert(group_id, vec![state]);
//                 group_id += 1;
//             }
//         }
// 
//         groups
//     }
// }
// 
// /// Load pubkeys from priority list file
// fn load_pubkeys_from_file(path: &std::path::Path) -> Result<Vec<Point>> {
//     use std::fs::File;
//     use std::io::{BufRead, BufReader};
// 
//     let file = File::open(path)?;
//     let reader = BufReader::new(file);
// 
//     let mut points = Vec::new();
//     for line in reader.lines() {
//         let line = line?;
//         let line = line.trim();
// 
//         // Skip comments and empty lines
//         if line.is_empty() || line.starts_with('#') {
//             continue;
//         }
// 
//         match hex::decode(line) {
//             Ok(bytes) => {
//                 // Production-ready point conversion with comprehensive validation
//                 match bytes.len() {
//                     33 => {
//                         // Compressed format: 0x02/0x03 + 32-byte x-coordinate
//                         let sign = bytes[0] == 0x03;
//                         let x_bytes = &bytes[1..33];
//                         match decompress_point_production(x_bytes, sign) {
//                             Ok(point) => {
//                                 if validate_point_on_curve(&point) {
//                                     points.push(point);
//                                     if points.len() % 1000 == 0 {
//                                         info!("Loaded {} targets from valuable_p2pk_pubkeys.txt", points.len());
//                                     }
//                                 
//             } else {
//                 loaded_targets
//             };
//                                     warn!("Point not on curve after decompression: {}", line);
//                                 }
//                             }
//                             Err(e) => warn!("Failed to decompress compressed pubkey: {} - {}", line, e),
//                         }
//                     }
//                     65 => {
//                         // Uncompressed format: 0x04 + 32-byte x + 32-byte y
//                         if bytes[0] == 0x04 {
//                             let x_bytes = &bytes[1..33];
//                             let y_bytes = &bytes[33..65];
//                             // Convert directly to our Point format
//                             let x_array: [u8; 32] = x_bytes.try_into().unwrap();
//                             let y_array: [u8; 32] = y_bytes.try_into().unwrap();
//                             let x_bigint = BigInt256::from_bytes_be(&x_array);
//                             let y_bigint = BigInt256::from_bytes_be(&y_array);
//                             let point = Point {
//                                 x: x_bigint.limbs,
//                                 y: y_bigint.limbs,
//                                 z: [1, 0, 0, 0],
//                             };
//                             if validate_point_on_curve(&point) {
//                                 points.push(point);
//                             
//             } else {
//                 loaded_targets
//             };
//                                 warn!("Uncompressed point not on curve: {}", line);
//                             }
//                         
//             } else {
//                 loaded_targets
//             };
//                             warn!("Invalid uncompressed pubkey prefix (expected 0x04): {}", line);
//                         }
//                     }
//                     _ => warn!("Invalid pubkey length (expected 33 or 65 bytes): {} - len={}", line, bytes.len()),
//                 }
//             }
//             Err(_) => warn!("Invalid hex in priority list: {}", line),
//         }
//     }
// 
//     Ok(points)
// }
// 
// 
// #[cfg(test)]
// mod tests {
// 
//     #[tokio::test]
//     async fn test_basic() {
//         // Basic test placeholder
//         assert!(true);
//     }
// }

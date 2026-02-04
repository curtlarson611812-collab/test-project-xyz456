//! Structured logging utilities
//!
//! Logs for attractors, convergence, pruning stats, checkpoint summaries

use log::{info, error, debug};
use std::time::Instant;
use std::collections::HashMap;
use std::fs::File;
use serde_json::{to_writer, from_reader};
use std::fs::read_to_string;
use regex::Regex;
use serde_json;

/// Setup structured logging
pub fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    Ok(())
}

/// Log kangaroo statistics
pub fn log_kangaroo_stats(total_kangaroos: usize, tame_count: usize, wild_count: usize) {
    info!("Kangaroo stats: {} total ({} tame, {} wild)",
          total_kangaroos, tame_count, wild_count);
}

/// Log attractor convergence
pub fn log_attractor_convergence(attractor_hash: &str, hit_count: usize, convergence_rate: f64) {
    info!("Attractor convergence: {} hits on {}, rate {:.4}",
          hit_count, attractor_hash, convergence_rate);
}

/// Log DP table statistics
pub fn log_dp_stats(total_entries: usize, utilization: f64, clusters: usize) {
    info!("DP table: {} entries ({:.1}% utilization), {} clusters",
          total_entries, utilization * 100.0, clusters);
}

/// Log pruning operation
pub fn log_pruning_operation(entries_removed: usize, duration_ms: u64, method: &str) {
    info!("DP pruning: removed {} entries in {}ms using {}",
          entries_removed, duration_ms, method);
}

/// Log collision detection
pub fn log_collision_found(kangaroo_id: u64, distance: u64, total_ops: u64) {
    info!("COLLISION DETECTED: kangaroo {}, distance {}, total ops {}",
          kangaroo_id, distance, total_ops);
}

/// Log solution verification
pub fn log_solution_verification(private_key: &str, verified: bool) {
    if verified {
        info!("SOLUTION VERIFIED: {}", private_key);
    } else {
        error!("SOLUTION VERIFICATION FAILED: {}", private_key);
    }
}

/// Log checkpoint summary
pub fn log_checkpoint_summary(ops_completed: u64, elapsed_seconds: f64,
                             ops_per_second: f64, targets_processed: usize) {
    info!("Checkpoint: {} ops in {:.1}s ({:.0} ops/sec), {} targets processed",
          ops_completed, elapsed_seconds, ops_per_second, targets_processed);
}

/// Performance monitoring timer
pub struct PerformanceTimer {
    start: Instant,
    label: String,
}

impl PerformanceTimer {
    pub fn new(label: &str) -> Self {
        debug!("Starting timer: {}", label);
        PerformanceTimer {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.start.elapsed().as_millis()
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        debug!("Timer {} completed in {:.2}ms",
               self.label, elapsed.as_millis());
    }
}

/// Progress tracker for long-running operations
pub struct ProgressTracker {
    total: u64,
    current: u64,
    start_time: Instant,
    last_report: Instant,
    report_interval: u64, // Report every N items
}

impl ProgressTracker {
    pub fn new(total: u64, report_interval: u64) -> Self {
        ProgressTracker {
            total,
            current: 0,
            start_time: Instant::now(),
            last_report: Instant::now(),
            report_interval,
        }
    }

    pub fn increment(&mut self, amount: u64) {
        self.current += amount;

        if self.current % self.report_interval == 0 ||
           self.last_report.elapsed().as_secs() >= 10 {
            self.report_progress();
            self.last_report = Instant::now();
        }
    }

    pub fn set_current(&mut self, current: u64) {
        self.current = current;
        self.report_progress();
    }

    fn report_progress(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let progress = self.current as f64 / self.total as f64;
        let eta_seconds = if progress > 0.0 {
            elapsed / progress * (1.0 - progress)
        } else {
            0.0
        };

        info!("Progress: {}/{} ({:.1}%), {:.0}s elapsed, ETA {:.0}s",
              self.current, self.total, progress * 100.0, elapsed, eta_seconds);
    }

    pub fn complete(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;
        info!("Completed: {} items in {:.2}s ({:.0} items/sec)",
              self.current, elapsed, rate);
    }
}

// Chunk: Comprehensive Nsight Metrics Parsing (src/utils/logging.rs)
// Dependencies: serde_json, regex, std::fs::read_to_string, std::collections::HashMap

#[derive(Debug, Clone)]
pub struct NsightMetrics {
    pub sm_efficiency: f64,
    pub achieved_occupancy: f64,
    pub warp_execution_efficiency: f64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub dram_utilization: f64,
    pub alu_utilization: f64,
    pub inst_throughput: f64,
    pub warp_nonpred_efficiency: f64,
    pub register_usage: u32,
    pub launched_blocks: u32,
    pub launched_threads: u32,
    pub optimization_recommendations: Vec<String>,
}

impl Default for NsightMetrics {
    fn default() -> Self {
        NsightMetrics {
            sm_efficiency: 0.0,
            achieved_occupancy: 0.0,
            warp_execution_efficiency: 0.0,
            l1_hit_rate: 0.0,
            l2_hit_rate: 0.0,
            dram_utilization: 0.0,
            alu_utilization: 0.0,
            inst_throughput: 0.0,
            warp_nonpred_efficiency: 0.0,
            register_usage: 0,
            launched_blocks: 0,
            launched_threads: 0,
            optimization_recommendations: Vec::new(),
        }
    }
}

pub fn load_comprehensive_nsight_metrics(path: &str) -> Option<NsightMetrics> {
    let json_str = read_to_string(path).ok()?;
    let all_metrics: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    let rho_kernel = all_metrics.get("rho_kernel")?;
    let mut metrics = NsightMetrics::default();

    // Parse occupancy metrics
    if let Some(eff) = rho_kernel.get("occ_sm_efficiency") {
        if let Some(s) = eff.as_str() {
            metrics.sm_efficiency = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(occ) = rho_kernel.get("occ_achieved_occupancy") {
        if let Some(s) = occ.as_str() {
            metrics.achieved_occupancy = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(warp_eff) = rho_kernel.get("occ_warp_execution_efficiency") {
        if let Some(s) = warp_eff.as_str() {
            metrics.warp_execution_efficiency = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    // Parse memory metrics
    if let Some(l1_hit) = rho_kernel.get("mem_l1tex__t_bytes_hit_rate") {
        if let Some(s) = l1_hit.as_str() {
            metrics.l1_hit_rate = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(l2_hit) = rho_kernel.get("mem_l2tex__t_bytes_hit_rate") {
        if let Some(s) = l2_hit.as_str() {
            metrics.l2_hit_rate = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(dram_pct) = rho_kernel.get("mem_dram__bytes_read.sum.pct_of_peak_sustained_active") {
        if let Some(s) = dram_pct.as_str() {
            metrics.dram_utilization = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    // Parse compute metrics
    if let Some(alu_util) = rho_kernel.get("compute_sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active") {
        if let Some(s) = alu_util.as_str() {
            metrics.alu_utilization = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(inst_tp) = rho_kernel.get("compute_sm__inst_executed.avg.pct_of_peak_sustained_active") {
        if let Some(s) = inst_tp.as_str() {
            metrics.inst_throughput = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    if let Some(warp_eff) = rho_kernel.get("compute_warp_nonpred_execution_efficiency") {
        if let Some(s) = warp_eff.as_str() {
            metrics.warp_nonpred_efficiency = s.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
        }
    }

    // Parse launch config metrics
    if let Some(regs) = rho_kernel.get("launch_register_usage") {
        if let Some(s) = regs.as_str() {
            metrics.register_usage = s.parse().unwrap_or(0);
        }
    }

    if let Some(blocks) = rho_kernel.get("launch_launched_blocks") {
        if let Some(s) = blocks.as_str() {
            metrics.launched_blocks = s.parse().unwrap_or(0);
        }
    }

    if let Some(threads) = rho_kernel.get("launch_launched_threads") {
        if let Some(s) = threads.as_str() {
            metrics.launched_threads = s.parse().unwrap_or(0);
        }
    }

    // Parse optimization recommendations
    if let Some(recs) = all_metrics.get("optimization_recommendations") {
        if let Some(arr) = recs.as_array() {
            metrics.optimization_recommendations = arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();
        }
    }

    Some(metrics)
}

pub fn load_nsight_util(path: &str) -> Option<f64> {
    load_comprehensive_nsight_metrics(path).map(|m| m.sm_efficiency)
}

pub fn get_avg_temp(log_path: &str) -> Option<u32> {
    let data = read_to_string(log_path).ok()?;
    let re = Regex::new(r"(\d+)C").ok()?;
    let temps: Vec<u32> = re.find_iter(&data).filter_map(|m| m.as_str().trim_end_matches('C').parse().ok()).collect();
    if temps.len() > 10 { Some(temps.iter().sum::<u32>() / temps.len() as u32) } else { None }
}

/// Generate optimization recommendations based on metrics
pub fn generate_metric_based_recommendations(metrics: &NsightMetrics) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Occupancy recommendations
    if metrics.sm_efficiency < 0.7 {
        recommendations.push("Low SM efficiency (<70%) - reduce register usage or kernel unrolling".to_string());
    }

    if metrics.achieved_occupancy < 0.6 {
        recommendations.push("Low occupancy (<60%) - reduce block size or increase parallelism".to_string());
    }

    if metrics.warp_execution_efficiency < 0.9 {
        recommendations.push("Low warp execution efficiency (<90%) - reduce divergent branches".to_string());
    }

    // Memory recommendations
    if metrics.l2_hit_rate < 0.7 {
        recommendations.push("Low L2 cache hit rate (<70%) - optimize data layout and access patterns".to_string());
    }

    if metrics.dram_utilization > 0.8 {
        recommendations.push("High DRAM utilization (>80%) - memory bandwidth bound, improve coalescing".to_string());
    }

    // Compute recommendations
    if metrics.alu_utilization < 0.8 {
        recommendations.push("Low ALU utilization (<80%) - fuse operations or reduce memory stalls".to_string());
    }

    if metrics.inst_throughput < 0.7 {
        recommendations.push("Low instruction throughput (<70%) - optimize kernel code".to_string());
    }

    if metrics.warp_nonpred_efficiency < 0.9 {
        recommendations.push("Low warp non-predicate efficiency (<90%) - reduce control flow divergence".to_string());
    }

    // Launch config recommendations
    if metrics.register_usage > 64 {
        recommendations.push(format!("High register usage ({} > 64) - reduce local variables", metrics.register_usage));
    }

    recommendations
}

/// ML-based optimization history management functions
/// Append profiling history for ML training data
pub fn append_history(path: &str, eff: f64, mem: f64, alu: f64, frac: f64) -> Result<(), Box<dyn std::error::Error>> {
    let mut hist: Vec<(f64, f64, f64, f64)> = load_history(path);
    hist.push((eff, mem, alu, frac));

    // Keep only last 100 entries to prevent file from growing too large
    let max_entries = 100;
    if hist.len() > max_entries {
        let start_idx = hist.len() - max_entries;
        hist = hist[start_idx..].to_vec();
    }

    let file = File::create(path)?;
    to_writer(file, &hist)?;
    Ok(())
}

/// Load historical profiling data for ML prediction
pub fn load_history(path: &str) -> Vec<(f64, f64, f64, f64)> {
    File::open(path)
        .and_then(|file| from_reader(file).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "JSON parse error")))
        .unwrap_or_default()
}

/// Integrate ML prediction into GPU config tuning
pub fn tune_ml_predict(config: &mut crate::config::GpuConfig) {
    use crate::gpu::backends::hybrid_backend::HybridBackend;

    // This would need to be called from the hybrid backend context
    // For now, just ensure the functions are available
    let _hist = load_history("history.json");
    // Prediction would be done in HybridBackend::predict_frac
}

/// Suggest stream usage based on Nsight metrics for low IPC scenarios
pub fn suggest_streams(metrics: &HashMap<String, f64>) {
    if let Some(ipc) = metrics.get("sm__inst_executed.avg.pct_of_peak_sustained_active") {
        if *ipc < 0.7 {
            log::info!("Nsight Suggestion: IPC {:.1}% < 70% - enable CUDA streams for compute/memory overlap", ipc * 100.0);
        }
    }
}
//! Structured logging utilities
//!
//! Logs for attractors, convergence, pruning stats, checkpoint summaries

use log::{info, error, debug};
use std::time::Instant;
use std::collections::HashMap;
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

// Chunk: Parse Metrics (src/utils/logging.rs)
// Dependencies: serde_json::from_str, regex::Regex, std::fs::read_to_string
pub fn load_nsight_util(path: &str) -> Option<f64> {
    let json_str = read_to_string(path).ok()?;
    let metrics: HashMap<String, HashMap<String, String>> = serde_json::from_str(&json_str).ok()?;
    metrics.get("rho_kernel").and_then(|m: &HashMap<String, String>| m.get("efficiency").and_then(|s: &String| s.parse().ok())).map(|e: f64| e / 100.0)
}

pub fn get_avg_temp(log_path: &str) -> Option<u32> {
    let data = read_to_string(log_path).ok()?;
    let re = Regex::new(r"(\d+)C").ok()?;
    let temps: Vec<u32> = re.find_iter(&data).filter_map(|m| m.as_str().trim_end_matches('C').parse().ok()).collect();
    if temps.len() > 10 { Some(temps.iter().sum::<u32>() / temps.len() as u32) } else { None }
}
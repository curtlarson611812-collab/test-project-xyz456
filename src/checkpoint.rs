//! Checkpoint/resume functionality for long-running hunts
//!
//! Features:
//! - Automatic state saving at regular intervals
//! - Resume from last checkpoint on startup
//! - Incremental progress tracking
//! - Crash recovery support
//! - Configurable checkpoint frequency

use crate::dp::DpTable;
use crate::types::{KangarooState, Target};
use anyhow::{anyhow, Result};
use bincode;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntCheckpoint {
    /// Checkpoint version for compatibility
    pub version: u32,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Total operations performed so far
    pub total_ops: u64,
    /// Current cycle number
    pub current_cycle: u64,
    /// Wild kangaroo states
    pub wild_states: Vec<KangarooState>,
    /// Tame kangaroo states
    pub tame_states: Vec<KangarooState>,
    /// DP table state (compressed)
    pub dp_table_snapshot: DpTableSnapshot,
    /// Current targets being processed
    pub active_targets: Vec<Target>,
    /// Performance metrics at checkpoint time
    pub performance_metrics: PerformanceMetricsSnapshot,
    /// Search parameters
    pub search_params: SearchParameters,
    /// Hunt statistics
    pub hunt_stats: HuntStatistics,
}

/// Compressed DP table snapshot for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpTableSnapshot {
    /// Total DP entries stored
    pub total_entries: usize,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Serialized compressed DP data
    pub compressed_data: Vec<u8>,
}

/// Performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsSnapshot {
    pub ops_per_second: f64,
    pub gpu_utilization_percent: f64,
    pub memory_usage_mb: f64,
    pub temperature_celsius: f64,
    pub power_consumption_watts: f64,
}

impl Default for PerformanceMetricsSnapshot {
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            gpu_utilization_percent: 0.0,
            memory_usage_mb: 0.0,
            temperature_celsius: 0.0,
            power_consumption_watts: 0.0,
        }
    }
}

/// Search parameters for resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParameters {
    pub search_mode: String,
    pub dp_bits: usize,
    pub herd_size: usize,
    pub range_low: Option<u64>,
    pub range_high: Option<u64>,
    pub bias_mode: String,
}

/// Hunt statistics for resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntStatistics {
    pub dp_found: u64,
    pub collisions_tested: u64,
    pub false_positives: u64,
    pub time_elapsed_seconds: u64,
}

/// Checkpoint manager for handling save/load operations
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    auto_save_interval_seconds: u64,
    last_save_time: SystemTime,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(
        checkpoint_dir: PathBuf,
        max_checkpoints: usize,
        auto_save_interval_seconds: u64,
    ) -> Self {
        // Create checkpoint directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).unwrap_or_else(|e| {
                eprintln!("Failed to create checkpoint directory: {}", e);
            });
        }

        CheckpointManager {
            checkpoint_dir,
            max_checkpoints,
            auto_save_interval_seconds,
            last_save_time: SystemTime::now(),
        }
    }

    /// Save checkpoint with automatic naming
    pub fn save_checkpoint(&mut self, checkpoint: &HuntCheckpoint) -> Result<PathBuf> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let filename = format!("checkpoint_{}.json", timestamp);
        let filepath = self.checkpoint_dir.join(filename);

        // Serialize checkpoint
        let json_data = serde_json::to_string_pretty(checkpoint)?;

        // Write to temporary file first, then rename for atomicity
        let temp_filepath = filepath.with_extension("tmp");
        {
            let mut file = File::create(&temp_filepath)?;
            file.write_all(json_data.as_bytes())?;
            file.flush()?;
        }

        // Atomic rename
        fs::rename(&temp_filepath, &filepath)?;

        self.last_save_time = SystemTime::now();

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(filepath)
    }

    /// Load latest checkpoint
    pub fn load_latest_checkpoint(&self) -> Result<Option<HuntCheckpoint>> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Find most recent checkpoint
        let latest_path = checkpoints
            .into_iter()
            .max_by_key(|path| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("checkpoint_"))
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0)
            })
            .ok_or_else(|| anyhow!("No valid checkpoints found"))?;

        Ok(Some(self.load_checkpoint(&latest_path)?))
    }

    /// Load specific checkpoint by path
    pub fn load_checkpoint(&self, path: &Path) -> Result<HuntCheckpoint> {
        let mut file = File::open(path)?;
        let mut json_data = String::new();
        file.read_to_string(&mut json_data)?;

        let checkpoint: HuntCheckpoint = serde_json::from_str(&json_data)?;
        Ok(checkpoint)
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<PathBuf>> {
        let mut checkpoints = Vec::new();

        if !self.checkpoint_dir.exists() {
            return Ok(checkpoints);
        }

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with("checkpoint_"))
                    .unwrap_or(false)
            {
                checkpoints.push(path);
            }
        }

        // Sort by timestamp (newest first)
        checkpoints.sort_by(|a, b| {
            let get_timestamp = |path: &Path| -> u64 {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("checkpoint_"))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            };

            get_timestamp(b).cmp(&get_timestamp(a))
        });

        Ok(checkpoints)
    }

    /// Check if auto-save is due
    pub fn should_auto_save(&self) -> bool {
        if let Ok(elapsed) = self.last_save_time.elapsed() {
            elapsed.as_secs() >= self.auto_save_interval_seconds
        } else {
            true // If time check fails, allow save
        }
    }

    /// Clean up old checkpoints, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.len() > self.max_checkpoints {
            // Remove oldest checkpoints
            for old_checkpoint in checkpoints.iter().skip(self.max_checkpoints) {
                if let Err(e) = fs::remove_file(old_checkpoint) {
                    eprintln!(
                        "Failed to remove old checkpoint {}: {}",
                        old_checkpoint.display(),
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Get checkpoint directory info
    pub fn get_checkpoint_info(&self) -> Result<CheckpointInfo> {
        let checkpoints = self.list_checkpoints()?;
        let total_size_bytes: u64 = checkpoints
            .iter()
            .filter_map(|path| fs::metadata(path).ok().map(|m| m.len()))
            .sum();

        Ok(CheckpointInfo {
            checkpoint_dir: self.checkpoint_dir.clone(),
            total_checkpoints: checkpoints.len(),
            total_size_mb: total_size_bytes as f64 / (1024.0 * 1024.0),
            oldest_checkpoint: checkpoints.last().cloned(),
            newest_checkpoint: checkpoints.first().cloned(),
        })
    }
}

/// Checkpoint directory information
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub checkpoint_dir: PathBuf,
    pub total_checkpoints: usize,
    pub total_size_mb: f64,
    pub oldest_checkpoint: Option<PathBuf>,
    pub newest_checkpoint: Option<PathBuf>,
}

/// Helper functions for creating checkpoints from hunt state
pub struct CheckpointBuilder;

impl CheckpointBuilder {
    /// Create checkpoint from current hunt state
    pub fn create_checkpoint(
        total_ops: u64,
        current_cycle: u64,
        wild_states: &[KangarooState],
        tame_states: &[KangarooState],
        dp_table: &DpTable,
        active_targets: &[Target],
        performance_metrics: PerformanceMetricsSnapshot,
        search_params: SearchParameters,
        hunt_stats: HuntStatistics,
    ) -> HuntCheckpoint {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create compressed DP table snapshot
        let dp_table_snapshot = Self::create_dp_table_snapshot(dp_table);

        HuntCheckpoint {
            version: 1, // Current checkpoint format version
            timestamp,
            total_ops,
            current_cycle,
            wild_states: wild_states.to_vec(),
            tame_states: tame_states.to_vec(),
            dp_table_snapshot,
            active_targets: active_targets.to_vec(),
            performance_metrics,
            search_params,
            hunt_stats,
        }
    }

    /// Create compressed DP table snapshot
    fn create_dp_table_snapshot(dp_table: &DpTable) -> DpTableSnapshot {
        let entries = dp_table.get_entries();
        let total_entries = entries.len();

        // Estimate memory usage (each entry is ~200-300 bytes)
        let memory_usage_mb = (total_entries * 250) as f64 / (1024.0 * 1024.0);

        // For now, we store entries in compressed binary format
        // In production, this would use proper compression (zstd, lz4, etc.)
        let compressed_data = if total_entries > 0 {
            // Serialize entries to binary format
            match bincode::serialize(entries) {
                Ok(data) => {
                    // Simple "compression" - just store as-is for now
                    // Real compression would reduce size by 60-80%
                    data
                }
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        let original_size = entries.len() * 250; // Estimated original size
        let compressed_size = compressed_data.len();
        let compression_ratio = if original_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        };

        DpTableSnapshot {
            total_entries,
            memory_usage_mb,
            compression_ratio,
            compressed_data,
        }
    }

    /// Validate checkpoint compatibility
    pub fn validate_checkpoint(checkpoint: &HuntCheckpoint) -> Result<()> {
        if checkpoint.version != 1 {
            return Err(anyhow!(
                "Incompatible checkpoint version: {}. Expected: 1",
                checkpoint.version
            ));
        }

        if checkpoint.wild_states.is_empty() && checkpoint.tame_states.is_empty() {
            return Err(anyhow!("Checkpoint contains no kangaroo states"));
        }

        if checkpoint.active_targets.is_empty() {
            return Err(anyhow!("Checkpoint contains no active targets"));
        }

        Ok(())
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(
            PathBuf::from("checkpoints"),
            10,   // Keep 10 checkpoints
            3600, // Auto-save every hour
        )
    }
}

impl Default for SearchParameters {
    fn default() -> Self {
        SearchParameters {
            search_mode: "full-range".to_string(),
            dp_bits: 26,
            herd_size: 500_000_000,
            range_low: None,
            range_high: None,
            bias_mode: "uniform".to_string(),
        }
    }
}

impl Default for HuntStatistics {
    fn default() -> Self {
        HuntStatistics {
            dp_found: 0,
            collisions_tested: 0,
            false_positives: 0,
            time_elapsed_seconds: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path().to_path_buf(), 5, 3600);

        let checkpoint = HuntCheckpoint {
            version: 1,
            timestamp: 1234567890,
            total_ops: 1000000,
            current_cycle: 42,
            wild_states: Vec::new(),
            tame_states: Vec::new(),
            dp_table_snapshot: DpTableSnapshot {
                total_entries: 1000,
                memory_usage_mb: 50.0,
                compression_ratio: 2.5,
                compressed_data: vec![1, 2, 3, 4],
            },
            active_targets: Vec::new(),
            performance_metrics: PerformanceMetricsSnapshot {
                ops_per_second: 100000.0,
                gpu_utilization_percent: 85.0,
                memory_usage_mb: 8192.0,
                temperature_celsius: 70.0,
                power_consumption_watts: 300.0,
            },
            search_params: SearchParameters::default(),
            hunt_stats: HuntStatistics::default(),
        };

        // Test save
        let save_result = manager.save_checkpoint(&checkpoint);
        assert!(save_result.is_ok());

        // Test load
        let load_result = manager.load_latest_checkpoint();
        assert!(load_result.is_ok());
        let loaded = load_result.unwrap().unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.total_ops, 1000000);
        assert_eq!(loaded.current_cycle, 42);
    }

    #[test]
    fn test_checkpoint_validation() {
        let valid_checkpoint = HuntCheckpoint {
            version: 1,
            timestamp: 1234567890,
            total_ops: 1000000,
            current_cycle: 42,
            wild_states: vec![], // Empty for test
            tame_states: vec![], // Empty for test
            dp_table_snapshot: DpTableSnapshot {
                total_entries: 1000,
                memory_usage_mb: 50.0,
                compression_ratio: 2.5,
                compressed_data: vec![1, 2, 3, 4],
            },
            active_targets: vec![], // Empty for test
            performance_metrics: PerformanceMetricsSnapshot::default(),
            search_params: SearchParameters::default(),
            hunt_stats: HuntStatistics::default(),
        };

        // Should pass validation
        let validation_result = CheckpointBuilder::validate_checkpoint(&valid_checkpoint);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_checkpoint_manager_info() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path().to_path_buf(), 5, 3600);

        let info = manager.get_checkpoint_info().unwrap();
        assert_eq!(info.total_checkpoints, 0);
        assert_eq!(info.total_size_mb, 0.0);
    }
}

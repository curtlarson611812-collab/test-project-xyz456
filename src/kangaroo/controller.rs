//! Kangaroo Controller for managing multiple target lists
//!
//! Orchestrates separate KangarooManager instances for different types of targets:
//! - Valuable P2PK (unbounded, large-scale)
//! - Test puzzles (bounded, quick validation)
//! - Unsolved puzzles (bounded, long-running)

use super::{KangarooManager, SearchConfig};
use crate::types::Point;
use crate::config::Config;
use crate::utils::pubkey_loader::{load_valuable_p2pk_keys, load_test_puzzle_keys, load_unsolved_puzzle_keys};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Statistics for controller operations
#[derive(Debug, Clone)]
pub struct ManagerStats {
    pub name: String,
    pub targets_loaded: usize,
    pub total_steps: u64,
    pub solutions_found: usize,
    pub active_time: std::time::Duration,
}

/// Controller for managing multiple kangaroo managers
pub struct KangarooController {
    managers: Vec<(KangarooManager, ManagerStats)>,
}

impl KangarooController {
    /// Create controller with specified target lists
    pub async fn new_with_lists(
        config: Config,
        load_valuable: Option<String>, // Path to valuable P2PK file
        load_test: bool,
        load_unsolved: bool,
    ) -> Result<Self> {
        let mut managers = Vec::new();

        // Load valuable P2PK targets
        if let Some(path) = load_valuable {
            let (points, search_config) = load_valuable_p2pk_keys(&path)?;
            // Convert to (Point, puzzle_id) tuples with ID 0 for P2PK
            let targets_with_ids: Vec<(Point, u32)> = points.into_iter().map(|p| (p, 0)).collect();
            let manager = KangarooManager::new_multi_config(targets_with_ids, search_config).await?;
            let stats = ManagerStats {
                name: "valuable_p2pk".to_string(),
                targets_loaded: manager.multi_targets().len(),
                total_steps: 0,
                solutions_found: 0,
                active_time: std::time::Duration::ZERO,
            };
            managers.push((manager, stats));
        }

        // Load test puzzles
        if load_test {
            let (points, search_config) = load_test_puzzle_keys();
            let manager = KangarooManager::new_multi_config(points, search_config).await?;
            let stats = ManagerStats {
                name: "test_puzzles".to_string(),
                targets_loaded: manager.multi_targets().len(),
                total_steps: 0,
                solutions_found: 0,
                active_time: std::time::Duration::ZERO,
            };
            managers.push((manager, stats));
        }

        // Load unsolved puzzles
        if load_unsolved {
            let (points, search_config) = load_unsolved_puzzle_keys();
            let manager = KangarooManager::new_multi_config(points, search_config).await?;
            let stats = ManagerStats {
                name: "unsolved_puzzles".to_string(),
                targets_loaded: manager.multi_targets().len(),
                total_steps: 0,
                solutions_found: 0,
                active_time: std::time::Duration::ZERO,
            };
            managers.push((manager, stats));
        }

        Ok(Self { managers })
    }

    /// Run all managers in parallel for specified steps
    pub fn run_parallel(&mut self, steps_per_cycle: u64) -> Result<()> {
        // Use Rayon for parallel execution across managers
        self.managers.par_iter_mut().for_each(|(manager, stats)| {
            let start_time = std::time::Instant::now();

            // Run steps on this manager
            if let Err(e) = manager.step_herds_multi(steps_per_cycle) {
                log::error!("Manager {} failed: {}", stats.name, e);
                return;
            }

            let elapsed = start_time.elapsed();
            stats.total_steps += steps_per_cycle;
            stats.active_time += elapsed;

            // Check for solutions (simplified - would need proper solution tracking)
            // stats.solutions_found += manager.solutions_found();

            log::info!("Manager {}: {} steps completed in {:.2}s",
                      stats.name, steps_per_cycle, elapsed.as_secs_f64());
        });

        Ok(())
    }

    /// Get current statistics for all managers
    pub fn get_stats(&self) -> Vec<ManagerStats> {
        self.managers.iter().map(|(_, stats)| stats.clone()).collect()
    }

    /// Check if any manager should continue running
    pub fn should_continue(&self) -> bool {
        self.managers.iter().any(|(manager, _)| {
            // Continue if manager hasn't exceeded max steps and no solution found
            manager.total_ops < manager.search_config.max_steps
        })
    }

    /// Get total statistics across all managers
    pub fn get_total_stats(&self) -> ManagerStats {
        let mut total = ManagerStats {
            name: "total".to_string(),
            targets_loaded: 0,
            total_steps: 0,
            solutions_found: 0,
            active_time: std::time::Duration::ZERO,
        };

        for (_, stats) in &self.managers {
            total.targets_loaded += stats.targets_loaded;
            total.total_steps += stats.total_steps;
            total.solutions_found += stats.solutions_found;
            total.active_time += stats.active_time;
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_controller_creation() {
        let config = Config {
            gpu_backend: "cpu".to_string(), // Use CPU for testing
            ..Default::default()
        };

        // Test with test puzzles only (no file I/O)
        let controller = KangarooController::new_with_lists(
            config,
            None,    // No valuable P2PK
            true,    // Load test puzzles
            false,   // No unsolved puzzles
        ).await;

        assert!(controller.is_ok());
        let controller = controller.unwrap();
        assert_eq!(controller.managers.len(), 1);
        assert_eq!(controller.managers[0].1.name, "test_puzzles");
    }
}
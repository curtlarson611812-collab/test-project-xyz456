//! Power management for RTX 5090 cluster
//!
//! Intelligent power optimization and thermal coordination
//! across multiple high-performance GPUs

use super::cluster::EfficiencyOptimizer;
use anyhow::Result;

/// Power management for RTX 5090 cluster
#[derive(Debug)]
pub struct PowerManager {
    power_limit_per_gpu: f64, // Watts
    total_cluster_limit: f64, // Watts
    efficiency_optimizer: EfficiencyOptimizer,
}

impl PowerManager {
    /// Create new power manager
    pub fn new(power_limit_per_gpu: f64, total_cluster_limit: f64) -> Self {
        PowerManager {
            power_limit_per_gpu,
            total_cluster_limit,
            efficiency_optimizer: EfficiencyOptimizer {
                power_efficiency_target: 0.85,
                performance_per_watt: std::collections::HashMap::new(),
            },
        }
    }

    /// Check if device is within power limits
    pub fn check_power_limits(&self, device_power: f64, total_cluster_power: f64) -> bool {
        device_power <= self.power_limit_per_gpu && total_cluster_power <= self.total_cluster_limit
    }

    /// Calculate optimal power allocation for workload
    pub fn get_optimal_power_allocation(&self, device_count: usize, workload_intensity: f64) -> Vec<f64> {
        let total_available = self.total_cluster_limit * workload_intensity;
        let per_device = total_available / device_count as f64;

        // Ensure we don't exceed per-device limits
        let allocated_per_device = per_device.min(self.power_limit_per_gpu);

        vec![allocated_per_device; device_count]
    }

    /// Get power efficiency target
    pub fn get_efficiency_target(&self) -> f64 {
        self.efficiency_optimizer.power_efficiency_target
    }

    /// Update power efficiency metrics
    pub fn update_efficiency_metrics(&mut self, device_id: usize, performance: f64, power_consumption: f64) {
        if power_consumption > 0.0 {
            let efficiency = performance / power_consumption;
            self.efficiency_optimizer.performance_per_watt.insert(device_id, efficiency);
        }
    }

    /// Get most power-efficient device
    pub fn get_most_efficient_device(&self) -> usize {
        self.efficiency_optimizer
            .performance_per_watt
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| *id)
            .unwrap_or(0)
    }
}
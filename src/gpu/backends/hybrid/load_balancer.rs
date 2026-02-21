//! Adaptive load balancing across GPU cluster
//!
//! Intelligent distribution of workloads across multiple RTX 5090 GPUs
//! with thermal, power, and performance awareness

use super::cluster::{GpuDevice, GpuApiType, BalancingStrategy, WorkloadPattern, PerformanceSnapshot, PatternType};
use crate::gpu::HybridOperation;
use anyhow::Result;
use std::collections::HashMap;

/// Adaptive load balancing across GPU cluster
#[derive(Debug, Clone)]
pub struct AdaptiveLoadBalancer {
    device_weights: HashMap<usize, f64>,
    workload_patterns: Vec<WorkloadPattern>,
    performance_history: Vec<PerformanceSnapshot>,
    balancing_strategy: BalancingStrategy,
}

impl AdaptiveLoadBalancer {
    pub fn new() -> Self {
        AdaptiveLoadBalancer {
            device_weights: HashMap::new(),
            workload_patterns: Vec::new(),
            performance_history: Vec::new(),
            balancing_strategy: BalancingStrategy::RoundRobin,
        }
    }
}

impl Default for AdaptiveLoadBalancer {
    fn default() -> Self {
        AdaptiveLoadBalancer {
            device_weights: std::collections::HashMap::new(),
            workload_patterns: Vec::new(),
            performance_history: Vec::new(),
            balancing_strategy: super::cluster::BalancingStrategy::Adaptive,
        }
    }
}

impl AdaptiveLoadBalancer {
    /// Initialize load balancer with default device weights
    pub fn initialize_load_balancer() -> AdaptiveLoadBalancer {
        let mut device_weights = std::collections::HashMap::new();

        // Initial equal weighting for all devices
        for i in 0..16 {
            // 8 Vulkan + 8 CUDA
            device_weights.insert(i, 1.0);
        }

        AdaptiveLoadBalancer {
            device_weights,
            workload_patterns: Vec::new(),
            performance_history: Vec::new(),
            balancing_strategy: super::cluster::BalancingStrategy::Adaptive,
        }
    }

    /// Update device weights based on current performance
    pub fn update_weights(&mut self, devices: &[GpuDevice]) {
        for device in devices {
            let base_weight = match device.api_type {
                GpuApiType::Cuda => 1.5, // CUDA typically faster
                GpuApiType::Vulkan => 1.0,
                GpuApiType::Hybrid => 1.3,
            };

            // Adjust weight based on current load (prefer less loaded devices)
            let load_factor = 1.0 - device.current_load;
            // Adjust weight based on temperature (prefer cooler devices)
            let temp_factor = if device.temperature > 75.0 { 0.8 } else { 1.0 };
            // Adjust weight based on power efficiency
            let power_factor = if device.power_consumption > device.power_consumption * 0.9 {
                0.9
            } else {
                1.0
            };

            let total_weight = base_weight * load_factor * temp_factor * power_factor;
            self.device_weights.insert(device.id, total_weight);
        }
    }

    /// Distribute operations across devices based on balancing strategy
    pub fn distribute_operations(
        &self,
        operations: Vec<HybridOperation>,
    ) -> Result<HashMap<usize, Vec<HybridOperation>>> {
        let mut distribution = HashMap::new();

        match self.balancing_strategy {
            BalancingStrategy::RoundRobin => {
                self.distribute_round_robin(&operations, &mut distribution);
            }
            BalancingStrategy::LoadBalanced => {
                self.distribute_load_balanced(&operations, &mut distribution);
            }
            BalancingStrategy::PerformanceBased => {
                self.distribute_performance_based(&operations, &mut distribution);
            }
            BalancingStrategy::Adaptive => {
                self.distribute_adaptive(&operations, &mut distribution);
            }
        }

        Ok(distribution)
    }

    /// Record performance metrics for adaptive balancing
    pub fn record_performance(&mut self, device_id: usize, _operation: &str, duration_ms: u128) {
        let snapshot = PerformanceSnapshot {
            timestamp: std::time::Instant::now(),
            device_loads: std::iter::once((device_id, 0.5)).collect(), // Placeholder load
            throughput: 1000.0 / duration_ms as f64, // ops per ms
        };
        self.performance_history.push(snapshot);

        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }

    /// Analyze workload patterns for optimization
    pub fn analyze_workload(&mut self, operation: &str, _data_size: usize) {
        let pattern = WorkloadPattern {
            operation_type: operation.to_string(),
            device_preference: self.device_weights.clone(),
            expected_duration: std::time::Duration::from_millis(100), // Placeholder
            pattern_type: PatternType::ComputationIntensive, // Placeholder - could be analyzed
            optimal_backend: "vulkan".to_string(), // Placeholder - could be learned
            observed_frequency: 1,
            confidence_score: 0.5, // Placeholder - could be calculated
        };

        // Update or add pattern
        if let Some(existing) = self.workload_patterns.iter_mut()
            .find(|p| p.operation_type == operation) {
            *existing = pattern;
        } else {
            self.workload_patterns.push(pattern);
        }
    }

    fn distribute_round_robin(
        &self,
        operations: &[HybridOperation],
        distribution: &mut HashMap<usize, Vec<HybridOperation>>,
    ) {
        let device_ids: Vec<usize> = self.device_weights.keys().cloned().collect();
        for (i, op) in operations.iter().enumerate() {
            let device_id = device_ids[i % device_ids.len()];
            distribution
                .entry(device_id)
                .or_insert_with(Vec::new)
                .push(op.clone());
        }
    }

    fn distribute_load_balanced(
        &self,
        operations: &[HybridOperation],
        distribution: &mut HashMap<usize, Vec<HybridOperation>>,
    ) {
        for op in operations {
            let best_device = self
                .device_weights
                .iter()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(id, _)| *id)
                .unwrap_or(0);

            distribution
                .entry(best_device)
                .or_insert_with(Vec::new)
                .push(op.clone());
        }
    }

    fn distribute_performance_based(
        &self,
        operations: &[HybridOperation],
        distribution: &mut HashMap<usize, Vec<HybridOperation>>,
    ) {
        // Use historical performance data for distribution
        for op in operations {
            let op_type = self.get_operation_type(op);
            let best_device = self.find_best_device_for_operation(&op_type);

            distribution
                .entry(best_device)
                .or_insert_with(Vec::new)
                .push(op.clone());
        }
    }

    fn distribute_adaptive(
        &self,
        operations: &[HybridOperation],
        distribution: &mut HashMap<usize, Vec<HybridOperation>>,
    ) {
        // Combine multiple strategies based on current system state
        let total_weight: f64 = self.device_weights.values().sum();

        for op in operations {
            // Weighted random selection based on device weights
            let mut cumulative_weight = 0.0;
            let random_value = (rand::random::<f64>() * total_weight) as f64;

            let selected_device = self
                .device_weights
                .iter()
                .find(|(_, weight)| {
                    cumulative_weight += **weight;
                    cumulative_weight >= random_value
                })
                .map(|(id, _)| *id)
                .unwrap_or(0);

            distribution
                .entry(selected_device)
                .or_insert_with(Vec::new)
                .push(op.clone());
        }
    }

    fn get_operation_type(&self, op: &HybridOperation) -> String {
        match op {
            HybridOperation::BatchInverse(_, _) => "batch_inverse".to_string(),
            HybridOperation::BatchBarrettReduce(_, _, _, _) => "batch_barrett_reduce".to_string(),
            HybridOperation::BatchBigIntMul(_, _) => "batch_bigint_mul".to_string(),
            HybridOperation::StepBatch(_, _, _) => "step_batch".to_string(),
            HybridOperation::DpCheck(_, _) => "dp_check".to_string(),
            HybridOperation::BigIntMul(_, _) => "bigint_mul".to_string(),
            HybridOperation::BarrettReduce(_, _, _) => "barrett_reduce".to_string(),
            HybridOperation::BsgsSolve(_, _, _) => "bsgs_solve".to_string(),
            HybridOperation::Inverse(_, _) => "inverse".to_string(),
            HybridOperation::SolveCollision(_, _, _, _, _, _) => "solve_collision".to_string(),
            HybridOperation::Custom(_, _) => "custom".to_string(),
        }
    }

    fn find_best_device_for_operation(&self, _op_type: &str) -> usize {
        // Find device with best historical performance for this operation type
        // For now, just return the highest weighted device
        self.device_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| *id)
            .unwrap_or(0)
    }
}
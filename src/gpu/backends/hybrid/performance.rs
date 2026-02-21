//! Performance monitoring, profiling, and optimization
//!
//! Advanced performance analysis, Nsight integration, and adaptive optimization

use super::monitoring::{NsightRuleResult, HybridOperationMetrics};
use crate::gpu::backends::hybrid::SchedulingContext;
use crate::config::GpuConfig;
use crate::utils::logging;
use anyhow::Result;
use std::collections::HashMap;

/// Performance and profiling operations
pub trait PerformanceOperations {
    /// Clear accumulated performance metrics
    fn clear_performance_metrics(&mut self);

    /// Get raw performance metrics
    fn get_raw_metrics(&self) -> &[HybridOperationMetrics];

    /// Get performance summary metrics
    fn get_performance_summary(&self) -> HashMap<String, f64>;

    /// Record operation performance for analysis
    fn record_operation_performance(
        &mut self,
        operation: &str,
        backend: &str,
        duration_ms: u128,
        data_size: usize,
        success: bool,
    );

    /// Apply Nsight rules for GPU optimization
    fn apply_nsight_rules(&self, config: &mut GpuConfig) -> Result<Vec<NsightRuleResult>>;

    /// Analyze Nsight rule results for divergence
    fn analyze_ecdlp_divergence(&self, metrics: &HashMap<String, f64>) -> NsightRuleResult;

    /// Optimize configuration based on performance metrics
    fn optimize_based_on_metrics(&self, config: &mut GpuConfig, metrics: &logging::NsightMetrics);

    /// Tune ML predictions for configuration optimization
    fn tune_ml_predict(&self, config: &mut GpuConfig);

    /// Analyze ECDLP bias efficiency
    fn analyze_ecdlp_bias_efficiency(&self, config: &GpuConfig, metrics: &HashMap<String, f64>) -> f64;
}

/// Performance operations implementation
pub struct PerformanceOperationsImpl {
    performance_metrics: Vec<HybridOperationMetrics>,
}

impl PerformanceOperationsImpl {
    /// Create new performance operations implementation
    pub fn new() -> Self {
        PerformanceOperationsImpl {
            performance_metrics: Vec::new(),
        }
    }
}

impl PerformanceOperations for PerformanceOperationsImpl {
    fn clear_performance_metrics(&mut self) {
        self.performance_metrics.clear();
    }

    fn get_raw_metrics(&self) -> &[HybridOperationMetrics] {
        &self.performance_metrics
    }

    fn get_performance_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        if self.performance_metrics.is_empty() {
            return summary;
        }

        let total_operations = self.performance_metrics.len() as f64;
        let successful_operations = self.performance_metrics.iter()
            .filter(|m| m.success)
            .count() as f64;

        let avg_duration = self.performance_metrics.iter()
            .map(|m| m.duration_ms as f64)
            .sum::<f64>() / total_operations;

        let total_data_processed = self.performance_metrics.iter()
            .map(|m| m.data_size)
            .sum::<usize>() as f64;

        let throughput = if avg_duration > 0.0 {
            total_data_processed / (avg_duration / 1000.0) // bytes per second
        } else {
            0.0
        };

        summary.insert("total_operations".to_string(), total_operations);
        summary.insert("success_rate".to_string(), successful_operations / total_operations);
        summary.insert("avg_duration_ms".to_string(), avg_duration);
        summary.insert("total_data_bytes".to_string(), total_data_processed);
        summary.insert("throughput_bytes_sec".to_string(), throughput);

        // Backend-specific metrics
        let backends: std::collections::HashSet<String> = self.performance_metrics.iter()
            .map(|m| m.backend.clone())
            .collect();

        for backend in backends {
            let backend_metrics: Vec<&HybridOperationMetrics> = self.performance_metrics.iter()
                .filter(|m| m.backend == backend)
                .collect();

            if !backend_metrics.is_empty() {
                let backend_avg = backend_metrics.iter()
                    .map(|m| m.duration_ms as f64)
                    .sum::<f64>() / backend_metrics.len() as f64;

                summary.insert(format!("{}_avg_duration_ms", backend), backend_avg);
                summary.insert(format!("{}_operations", backend), backend_metrics.len() as f64);
            }
        }

        summary
    }

    fn record_operation_performance(
        &mut self,
        operation: &str,
        backend: &str,
        duration_ms: u128,
        data_size: usize,
        success: bool,
    ) {
        let metric = HybridOperationMetrics {
            operation: operation.to_string(),
            backend: backend.to_string(),
            duration_ms,
            data_size,
            success,
            timestamp: std::time::Instant::now(),
        };

        self.performance_metrics.push(metric);

        // Keep only recent metrics to prevent unbounded growth
        if self.performance_metrics.len() > 10000 {
            // Keep only the most recent 5000 metrics
            self.performance_metrics = self.performance_metrics
                .split_off(self.performance_metrics.len() - 5000);
        }
    }


    fn apply_nsight_rules(&self, config: &mut GpuConfig) -> Result<Vec<NsightRuleResult>> {
        let mut results = Vec::new();

        // Apply various Nsight performance rules
        // These would analyze GPU performance counters and suggest optimizations

        // Example rules (would be expanded with actual Nsight integration)
        let rules = vec![
            ("Memory Coalescing", 0.75, "Improve memory access patterns for better coalescing"),
            ("Occupancy", 0.82, "Increase thread blocks for better GPU utilization"),
            ("Branch Divergence", 0.68, "Reduce conditional branches in kernels"),
            ("Shared Memory Usage", 0.91, "Optimize shared memory allocation"),
            ("Register Pressure", 0.73, "Reduce register usage per thread"),
        ];

        for (rule_name, score, suggestion) in rules {
            let result = NsightRuleResult::new(rule_name, score, suggestion);
            results.push(result);

            // Apply automatic adjustments based on rule results
            match rule_name {
                "Memory Coalescing" if score < 0.8 => {
                    config.max_regs = (config.max_regs * 2).min(128);
                }
                "Occupancy" if score < 0.8 => {
                    config.max_kangaroos = (config.max_kangaroos as f64 * 1.2) as usize;
                }
                "Branch Divergence" if score < 0.7 => {
                    // Would apply divergence-reducing optimizations
                }
                _ => {}
            }
        }

        Ok(results)
    }

    fn analyze_ecdlp_divergence(&self, metrics: &HashMap<String, f64>) -> NsightRuleResult {
        // Analyze divergence patterns specific to ECDLP operations
        let divergence_score = metrics.get("branch_divergence").cloned().unwrap_or(0.5);
        let occupancy_score = metrics.get("occupancy").cloned().unwrap_or(0.8);

        let overall_score = (divergence_score + occupancy_score) / 2.0;

        let suggestion = if overall_score < 0.6 {
            "High divergence detected in ECDLP operations. Consider using more uniform workloads or reducing conditional logic in kernels."
        } else if overall_score < 0.8 {
            "Moderate divergence in ECDLP operations. Some optimization opportunities exist."
        } else {
            "Good divergence characteristics for ECDLP operations."
        };

        NsightRuleResult::new("ECDLP Divergence Analysis", overall_score, suggestion)
    }

    fn optimize_based_on_metrics(&self, config: &mut GpuConfig, metrics: &logging::NsightMetrics) {
        // Apply metrics-based optimizations
        // This would contain sophisticated optimization logic based on Nsight metrics

        // Example optimizations
        if metrics.alu_utilization < 0.7 {
            // Improve ALU utilization
            config.max_regs = (config.max_regs * 2).min(128);
        }

        if metrics.sm_efficiency < 0.7 {
            // Increase parallelism
            config.max_kangaroos = (config.max_kangaroos * 2).min(100000);
        }

        if metrics.achieved_occupancy < 0.7 {
            // Reduce load to prevent thermal throttling
            config.gpu_frac *= 0.8;
        }
    }

    fn tune_ml_predict(&self, config: &mut GpuConfig) {
        // Use machine learning to predict optimal configuration parameters
        // This would involve training models on performance data

        // Example ML-based tuning
        let predicted_optimal_threads = 256; // Would be predicted by ML model
        let predicted_optimal_frac = 0.65;   // Would be predicted by ML model

        config.max_regs = predicted_optimal_threads as i32;
        config.gpu_frac = predicted_optimal_frac;
    }

    fn analyze_ecdlp_bias_efficiency(&self, config: &GpuConfig, metrics: &HashMap<String, f64>) -> f64 {
        // Analyze how effectively bias optimizations are working for ECDLP
        let collision_rate = metrics.get("collision_rate").cloned().unwrap_or(0.001);
        let false_positive_rate = metrics.get("false_positive_rate").cloned().unwrap_or(0.1);

        // Calculate efficiency as collision_rate / (collision_rate + false_positive_rate)
        if collision_rate + false_positive_rate > 0.0 {
            collision_rate / (collision_rate + false_positive_rate)
        } else {
            0.5 // Default neutral efficiency
        }
    }
}

impl Default for PerformanceOperationsImpl {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended GPU configuration with performance tuning
pub struct ExtendedGpuConfig {
    pub base_config: GpuConfig,
    pub performance_history: Vec<HashMap<String, f64>>,
    pub optimization_history: Vec<String>,
}

impl ExtendedGpuConfig {
    /// Create new extended GPU configuration
    pub fn new(base_config: GpuConfig) -> Self {
        ExtendedGpuConfig {
            base_config,
            performance_history: Vec::new(),
            optimization_history: Vec::new(),
        }
    }

    /// Record performance snapshot
    pub fn record_performance(&mut self, metrics: HashMap<String, f64>) {
        self.performance_history.push(metrics);
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }

    /// Record optimization action
    pub fn record_optimization(&mut self, action: String) {
        self.optimization_history.push(action);
        if self.optimization_history.len() > 50 {
            self.optimization_history.remove(0);
        }
    }

    /// Get performance trend analysis
    pub fn get_performance_trend(&self, metric: &str) -> Option<f64> {
        if self.performance_history.len() < 2 {
            return None;
        }

        let recent = &self.performance_history[self.performance_history.len().saturating_sub(10)..];
        let values: Vec<f64> = recent.iter()
            .filter_map(|m| m.get(metric).cloned())
            .collect();

        if values.len() < 2 {
            return None;
        }

        let first_avg = values[..values.len()/2].iter().sum::<f64>() / (values.len()/2) as f64;
        let second_avg = values[values.len()/2..].iter().sum::<f64>() / (values.len() - values.len()/2) as f64;

        if first_avg > 0.0 {
            Some((second_avg - first_avg) / first_avg) // Percentage change
        } else {
            None
        }
    }

    /// Profile relative device performance for load balancing
    pub async fn profile_device_performance(&self) -> (f32, f32) {
        // Profile small batch performance to determine relative speeds
        // Returns (cuda_ratio, vulkan_ratio) where cuda_ratio + vulkan_ratio = 1.0

        #[cfg(all(feature = "rustacuda", feature = "wgpu"))]
        {
            // Implement actual profiling with small test batches
            let cuda_time = self.profile_cuda_performance().await;
            let vulkan_time = self.profile_vulkan_performance().await;

            if cuda_time > 0.0 && vulkan_time > 0.0 {
                // Calculate relative performance ratios
                let total_time = cuda_time + vulkan_time;
                let cuda_ratio = vulkan_time / total_time; // Faster device gets higher ratio
                let vulkan_ratio = cuda_time / total_time;
                (cuda_ratio, vulkan_ratio)
            } else {
                // Fallback to equal split if profiling fails
                (0.5, 0.5)
            }
        }

        #[cfg(not(all(feature = "rustacuda", feature = "wgpu")))]
        {
            // Single backend - use it fully
            (0.0, 1.0)
        }
    }

    /// Profile hashrates for GPU/CPU comparison
    pub fn profile_hashrates(config: &crate::config::GpuConfig) -> (f64, f64) {
        // gpu_ops_sec, cpu_ops_sec
        let test_steps = 10000;
        let test_states = vec![crate::types::RhoState::default(); config.max_kangaroos.min(512)]; // Small for quick
        let jumps = vec![crate::math::bigint::BigInt256::one(); 256];

        // GPU profile (simplified)
        let gpu_start = std::time::Instant::now();
        // In real implementation: dispatch_and_update(device, kernel, test_states.clone(), jumps.clone(), bias, test_steps)
        let gpu_time = gpu_start.elapsed().as_secs_f64();
        let gpu_hr = if gpu_time > 0.0 {
            (test_steps as f64 * test_states.len() as f64) / gpu_time
        } else {
            1000000.0 // Fallback estimate
        };

        // CPU profile (simplified)
        let cpu_start = std::time::Instant::now();
        // In real implementation: CPU stepping simulation
        let cpu_time = cpu_start.elapsed().as_secs_f64();
        let cpu_hr = if cpu_time > 0.0 {
            (test_steps as f64 * test_states.len() as f64) / cpu_time
        } else {
            10000.0 // Fallback estimate
        };

        (gpu_hr, cpu_hr)
    }

    /// Adjust GPU fraction based on utilization and thermal state
    pub fn adjust_gpu_frac(config: &mut crate::config::GpuConfig, util: f64, temp: u32) {
        // util from Nsight [0-1], temp from log
        let (gpu_hr, cpu_hr) = Self::profile_hashrates(config);
        let target_ratio = gpu_hr / (gpu_hr + cpu_hr);
        let util_norm = util; // 0.8 ideal =1.0
        let temp_norm = if temp > 80 {
            0.0
        } else if temp < 65 {
            1.0
        } else {
            (80.0 - temp as f64) / 15.0
        };
        let delta = 0.05 * (util_norm - (1.0 - temp_norm)); // Positive if high util/low temp
        config.gpu_frac = (config.gpu_frac + delta).clamp(0.5, 0.9); // Laptop bounds
        if config.gpu_frac > target_ratio {
            config.gpu_frac = target_ratio;
        }
    }

    /// Update bottleneck analysis for pipeline optimization
    pub fn update_bottleneck_analysis(
        &self,
        monitor: &mut super::execution::PipelinePerformanceMonitor,
        stage_name: &str,
        duration: std::time::Duration,
    ) {
        // Update stage latency tracking
        monitor.stage_latencies
            .entry(stage_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration.as_secs_f64());

        // Keep only recent measurements
        if let Some(latencies) = monitor.stage_latencies.get_mut(stage_name) {
            if latencies.len() > 100 {
                latencies.remove(0);
            }
        }

        // Update bottleneck detection
        if let Some(latencies) = monitor.stage_latencies.get(stage_name) {
            if let Some(avg_latency) = latencies.iter().sum::<f64>().checked_div(latencies.len() as f64) {
                if monitor.bottleneck_detection.slowest_stage.is_none()
                    || avg_latency > monitor.bottleneck_detection.slowest_stage_avg {
                    monitor.bottleneck_detection.slowest_stage = Some(stage_name.to_string());
                    monitor.bottleneck_detection.slowest_stage_avg = avg_latency;
                }
            }
        }
    }

    /// Predict optimal GPU fraction using historical data
    pub fn predict_frac(&self, history: &Vec<(f64, f64, f64, f64)>) -> f64 {
        // History format: (sm_eff, mem_pct, alu_util, past_frac)
        if history.len() < 5 {
            return 0.7; // Default if insufficient data
        }

        // Simplified linear regression for now
        // Use simple averaging with weighted recent history
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, &(_, _, _, frac)) in history.iter().enumerate() {
            let weight = (i + 1) as f64 / history.len() as f64; // Weight recent data more
            weighted_sum += frac * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.5, 0.9)
        } else {
            0.7
        }
    }

    /// Apply rule-based configuration adjustments
    pub fn apply_rule_based_adjustments(config: &mut crate::config::GpuConfig) {
        // Load rule suggestions and apply automatic adjustments
        if let Ok(json_str) = std::fs::read_to_string("suggestions.json") {
            if let Ok(suggestions) =
                serde_json::from_str::<std::collections::HashMap<String, String>>(&json_str)
            {
                let mut adjustments_made = Vec::new();

                // Apply specific rule-based adjustments
                if suggestions
                    .values()
                    .any(|s| s.contains("Low Coalescing") || s.contains("SoA"))
                {
                    config.max_kangaroos = (config.max_kangaroos * 3 / 4).max(512);
                    adjustments_made.push("Reduced kangaroos for SoA coalescing optimization");
                }

                if suggestions.values().any(|s| s.contains("High Register Pressure")) {
                    config.max_kangaroos = (config.max_kangaroos / 2).max(256);
                    adjustments_made.push("Reduced kangaroos for register pressure relief");
                }

                if suggestions.values().any(|s| s.contains("DRAM Bottleneck")) {
                    config.gpu_frac = (config.gpu_frac * 3 / 4).max(0.3);
                    adjustments_made.push("Reduced GPU fraction for DRAM bottleneck");
                }

                if suggestions.values().any(|s| s.contains("Low Occupancy")) {
                    config.max_kangaroos = (config.max_kangaroos * 4 / 3).min(100000);
                    adjustments_made.push("Increased kangaroos for better occupancy");
                }

                // Log applied adjustments
                if !adjustments_made.is_empty() {
                    log::info!("Applied {} rule-based adjustments:", adjustments_made.len());
                    for adjustment in adjustments_made {
                        log::info!("  - {}", adjustment);
                    }
                }
            }
        }
    }

    /// Apply single rule adjustment
    pub fn apply_rule_adjustment(config: &mut crate::config::GpuConfig, rule: &str, severity: f64) {
        match rule {
            "memory_coalescing" => {
                if severity > 0.7 {
                    config.max_kangaroos = (config.max_kangaroos * 3 / 4).max(512);
                }
            }
            "register_pressure" => {
                if severity > 0.6 {
                    config.max_kangaroos = (config.max_kangaroos / 2).max(256);
                }
            }
            "dram_bandwidth" => {
                if severity > 0.8 {
                    config.gpu_frac = (config.gpu_frac * 3 / 4).max(0.3);
                }
            }
            "occupancy" => {
                if severity < 0.5 {
                    config.max_kangaroos = (config.max_kangaroos * 5 / 4).min(100000);
                }
            }
            _ => {}
        }
    }

    /// Dispatch hybrid scaled with rules and metrics
    pub fn dispatch_hybrid_scaled_with_rules_and_metrics(
        config: &mut crate::config::GpuConfig,
        _target: &crate::math::bigint::BigInt256,
        _range: (crate::math::bigint::BigInt256, crate::math::bigint::BigInt256),
        total_steps: u64,
    ) -> Option<crate::math::bigint::BigInt256> {
        let mut completed = 0;
        let batch_size = 1000000; // 1M steps/batch
        let mut rules_applied = false;
        let mut metrics_checked = false;

        while completed < total_steps {
            let batch = batch_size.min((total_steps - completed) as usize);

            // Apply rule-based adjustments (once per run)
            if !rules_applied {
                Self::apply_rule_based_adjustments(config);
                rules_applied = true;
            }

            // Load and apply metrics-based optimization
            if !metrics_checked {
                // In real implementation, would load Nsight metrics
                // For now, skip metrics-based optimization
                metrics_checked = true;
            }

            // Process batch (placeholder)
            completed += batch as u64;

            // Check for early termination (collision found)
            // In real implementation, would check for solution
        }

        None // No solution found
    }

    /// Profile Vulkan performance
    pub async fn profile_vulkan_performance(&self) -> f32 {
        // Create small test batch for profiling
        let test_batch_size = 1024;
        let start = std::time::Instant::now();

        // Simulate Vulkan shader profiling
        // In real implementation, would run actual Vulkan compute shaders
        tokio::time::sleep(std::time::Duration::from_millis(12)).await;

        let elapsed = start.elapsed().as_secs_f32();
        elapsed
    }

    /// Profile CUDA performance
    pub async fn profile_cuda_performance(&self) -> f32 {
        // Create small test batch for profiling
        let test_batch_size = 1024;
        let start = std::time::Instant::now();

        // Simulate CUDA kernel profiling
        // In real implementation, would run actual CUDA kernels
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let elapsed = start.elapsed().as_secs_f32();
        elapsed
    }

    /// Record performance metrics for analysis
    pub fn record_performance_metrics(&mut self, operation: &str, backend: &str, duration_ms: u128) {
        let metrics = super::monitoring::HybridOperationMetrics {
            operation: operation.to_string(),
            vulkan_time_ms: if backend == "vulkan" { duration_ms } else { 0 },
            cuda_time_ms: if backend == "cuda" { duration_ms } else { 0 },
            staging_time_ms: 0, // Not measured yet
            total_time_ms: duration_ms,
            backend_used: backend.to_string(),
        };

        self.performance_metrics.push(metrics);

        // Keep only recent metrics
        if self.performance_metrics.len() > 1000 {
            self.performance_metrics.remove(0);
        }
    }

    /// Update workload patterns for adaptive scheduling
    pub fn update_workload_patterns(&mut self, operation: &str, data_size: usize, backend: &str) {
        // Update workload pattern analysis for future scheduling decisions
        // This would track which backends work best for different workload types
        let _pattern_key = format!("{}_{}", operation, data_size);
        // In real implementation, would update pattern database
        log::debug!("Updated workload pattern: {} on {} with size {}", operation, backend, data_size);
    }

    /// Load Nsight utilization metrics
    pub fn load_nsight_util(&self, path: &str) -> Option<f64> {
        // Parse Nsight metrics file and extract utilization
        // Implementation would read and parse actual metrics
        match std::fs::read_to_string(path) {
            Ok(_content) => {
                // In real implementation, parse JSON metrics
                Some(0.8) // Placeholder
            }
            Err(_) => {
                log::warn!("Could not load Nsight metrics from {}", path);
                None
            }
        }
    }

    /// Parse rule suggestion from string
    pub fn parse_rule_suggestion(&self, suggestion: &str) -> Option<(&str, f64)> {
        // Parse rule suggestions from Nsight or other tools
        // Format: "rule_name:severity" or just "rule_name"
        if let Some(colon_pos) = suggestion.find(':') {
            let rule = &suggestion[..colon_pos];
            let severity_str = &suggestion[colon_pos + 1..];
            if let Ok(severity) = severity_str.parse::<f64>() {
                Some((rule.trim(), severity))
            } else {
                Some((rule.trim(), 0.5)) // Default severity
            }
        } else {
            Some((suggestion.trim(), 0.5))
        }
    }

    /// Production-ready optimization based on metrics
    pub fn optimize_based_on_metrics_production(
        config: &mut crate::config::GpuConfig,
        metrics: &crate::utils::logging::NsightMetrics,
    ) {
        let mut optimization_applied = false;

        // Memory-bound detection and optimization
        if metrics.dram_utilization > 0.8 && metrics.l2_hit_rate < 0.7 {
            // Memory bottleneck detected - reduce parallelism for better cache locality
            config.max_kangaroos = (config.max_kangaroos * 4 / 5).max(512);
            optimization_applied = true;
            log::info!("Applied memory-bound optimization: reduced kangaroos to {}", config.max_kangaroos);
        }

        // Compute-bound detection and optimization
        if metrics.sm_efficiency > 0.9 && metrics.dram_utilization < 0.6 {
            // Compute bottleneck - increase parallelism if occupancy allows
            if metrics.achieved_occupancy < 0.8 {
                config.max_kangaroos = (config.max_kangaroos * 6 / 5).min(200000);
                optimization_applied = true;
                log::info!("Applied compute-bound optimization: increased kangaroos to {}", config.max_kangaroos);
            }
        }

        // Register pressure optimization
        if metrics.sm_efficiency < 0.7 && metrics.achieved_occupancy < 0.5 {
            // Register spilling detected - reduce work per thread
            config.max_kangaroos = (config.max_kangaroos / 2).max(256);
            optimization_applied = true;
            log::info!("Applied register pressure optimization: reduced kangaroos to {}", config.max_kangaroos);
        }

        if !optimization_applied {
            log::info!("No optimizations applied - current configuration appears optimal");
        }
    }
}
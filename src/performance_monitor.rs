//! Comprehensive performance monitoring and metrics collection for SpeedBitCrack V3
//!
//! Features:
//! - Real-time performance metrics collection
//! - GPU utilization tracking
//! - Memory bandwidth monitoring
//! - Thermal and power consumption analysis
//! - Performance trend analysis and bottleneck detection
//! - Integration with CLI for live updates

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result;

/// Comprehensive performance metrics collector
pub struct PerformanceMonitor {
    metrics_history: Arc<Mutex<Vec<PerformanceSnapshot>>>,
    gpu_metrics: Arc<Mutex<HashMap<usize, GpuMetrics>>>,
    system_metrics: Arc<Mutex<SystemMetrics>>,
    alert_thresholds: PerformanceThresholds,
    start_time: Instant,
    collection_interval: Duration,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub total_ops: u64,
    pub ops_per_second: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub network_bandwidth_mbps: f64,
    pub temperature_celsius: f64,
    pub power_consumption_watts: f64,
    pub active_devices: usize,
    pub bottleneck_type: Option<BottleneckType>,
}

#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub device_id: usize,
    pub name: String,
    pub utilization: f64,
    pub memory_used_mb: f64,
    pub memory_total_mb: f64,
    pub temperature: f64,
    pub power_draw: f64,
    pub fan_speed_percent: f64,
    pub clock_speed_mhz: u32,
    pub memory_clock_mhz: u32,
    pub pcie_bandwidth_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
    pub numa_imbalance_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_temperature_celsius: f64,
    pub max_gpu_utilization: f64,
    pub max_memory_usage_percent: f64,
    pub min_ops_per_second: f64,
    pub max_power_consumption_watts: f64,
    pub bottleneck_detection_threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    CpuBound,
    GpuBound,
    MemoryBound,
    NetworkBound,
    ThermalThrottling,
    PowerLimited,
    PcieBandwidth,
    NumAImbalance,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        PerformanceMonitor {
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            gpu_metrics: Arc::new(Mutex::new(HashMap::new())),
            system_metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            alert_thresholds: PerformanceThresholds::default(),
            start_time: Instant::now(),
            collection_interval: Duration::from_secs(1),
        }
    }

    /// Start background monitoring thread
    pub fn start_monitoring(&self) -> Result<()> {
        let metrics_history = Arc::clone(&self.metrics_history);
        let gpu_metrics = Arc::clone(&self.gpu_metrics);
        let system_metrics = Arc::clone(&self.system_metrics);
        let thresholds = self.alert_thresholds.clone();
        let interval = self.collection_interval;

        std::thread::spawn(move || {
            loop {
                if let Err(e) = Self::collect_metrics(
                    &metrics_history,
                    &gpu_metrics,
                    &system_metrics,
                    &thresholds,
                ) {
                    eprintln!("Performance monitoring error: {}", e);
                }
                std::thread::sleep(interval);
            }
        });

        Ok(())
    }

    /// Collect current performance metrics
    fn collect_metrics(
        history: &Arc<Mutex<Vec<PerformanceSnapshot>>>,
        gpu_metrics: &Arc<Mutex<HashMap<usize, GpuMetrics>>>,
        system_metrics: &Arc<Mutex<SystemMetrics>>,
        thresholds: &PerformanceThresholds,
    ) -> Result<()> {
        let timestamp = Instant::now();

        // Collect GPU metrics for all devices
        Self::collect_gpu_metrics(gpu_metrics)?;

        // Collect system-wide metrics
        Self::collect_system_metrics(system_metrics)?;

        // Calculate derived metrics
        let gpu_metrics_locked = gpu_metrics.lock().unwrap();
        let system_metrics_locked = system_metrics.lock().unwrap();

        let avg_gpu_utilization = gpu_metrics_locked.values()
            .map(|m| m.utilization)
            .sum::<f64>() / gpu_metrics_locked.len() as f64;

        let total_memory_used = gpu_metrics_locked.values()
            .map(|m| m.memory_used_mb)
            .sum::<f64>();

        let total_power_consumption = gpu_metrics_locked.values()
            .map(|m| m.power_draw)
            .sum::<f64>();

        let avg_temperature = gpu_metrics_locked.values()
            .map(|m| m.temperature)
            .sum::<f64>() / gpu_metrics_locked.len() as f64;

        // Detect bottlenecks
        let bottleneck = Self::detect_bottleneck(&gpu_metrics_locked, &system_metrics_locked, thresholds);

        // Create snapshot
        let mut snapshot = PerformanceSnapshot {
            timestamp,
            total_ops: 0, // Would be updated by caller
            ops_per_second: 0.0, // Would be calculated from history
            memory_usage_mb: total_memory_used,
            gpu_utilization_percent: avg_gpu_utilization,
            cpu_utilization_percent: system_metrics_locked.cpu_usage_percent,
            network_bandwidth_mbps: system_metrics_locked.network_rx_mbps + system_metrics_locked.network_tx_mbps,
            temperature_celsius: avg_temperature,
            power_consumption_watts: total_power_consumption,
            active_devices: gpu_metrics_locked.len(),
            bottleneck_type: bottleneck,
        };

        // Calculate ops per second from history
        let mut history_locked = history.lock().unwrap();
        if let Some(previous) = history_locked.last() {
            let time_diff = timestamp.duration_since(previous.timestamp).as_secs_f64();
            if time_diff > 0.0 {
                snapshot.ops_per_second = (snapshot.total_ops.saturating_sub(previous.total_ops)) as f64 / time_diff;
            }
        }

        history_locked.push(snapshot);

        // Keep only last 1000 snapshots to prevent unbounded growth
        if history_locked.len() > 1000 {
            history_locked.remove(0);
        }

        Ok(())
    }

    /// Collect metrics for all GPU devices
    fn collect_gpu_metrics(gpu_metrics: &Arc<Mutex<HashMap<usize, GpuMetrics>>>) -> Result<()> {
        let mut metrics = gpu_metrics.lock().unwrap();

        // For RTX 5090 cluster (8 GPUs)
        for device_id in 0..8 {
            let gpu_metric = GpuMetrics {
                device_id,
                name: format!("RTX 5090 #{}", device_id),
                utilization: Self::query_gpu_utilization(device_id)?,
                memory_used_mb: Self::query_gpu_memory_used(device_id)?,
                memory_total_mb: 32768.0, // 32GB for RTX 5090
                temperature: Self::query_gpu_temperature(device_id)?,
                power_draw: Self::query_gpu_power_draw(device_id)?,
                fan_speed_percent: Self::query_gpu_fan_speed(device_id)?,
                clock_speed_mhz: Self::query_gpu_clock_speed(device_id)?,
                memory_clock_mhz: Self::query_gpu_memory_clock(device_id)?,
                pcie_bandwidth_utilization: Self::query_pcie_utilization(device_id)?,
            };

            metrics.insert(device_id, gpu_metric);
        }

        Ok(())
    }

    /// Collect system-wide performance metrics
    fn collect_system_metrics(system_metrics: &Arc<Mutex<SystemMetrics>>) -> Result<()> {
        let mut metrics = system_metrics.lock().unwrap();

        metrics.cpu_usage_percent = Self::query_cpu_usage()?;
        metrics.memory_used_gb = Self::query_memory_used()?;
        metrics.memory_total_gb = Self::query_memory_total()?;
        metrics.network_rx_mbps = Self::query_network_rx()?;
        metrics.network_tx_mbps = Self::query_network_tx()?;
        metrics.disk_read_mbps = Self::query_disk_read()?;
        metrics.disk_write_mbps = Self::query_disk_write()?;
        metrics.numa_imbalance_score = Self::calculate_numa_imbalance()?;

        Ok(())
    }

    /// Detect performance bottlenecks
    fn detect_bottleneck(
        gpu_metrics: &HashMap<usize, GpuMetrics>,
        system_metrics: &SystemMetrics,
        thresholds: &PerformanceThresholds,
    ) -> Option<BottleneckType> {
        // Check thermal throttling
        let avg_temp = gpu_metrics.values().map(|m| m.temperature).sum::<f64>() / gpu_metrics.len() as f64;
        if avg_temp > thresholds.max_temperature_celsius {
            return Some(BottleneckType::ThermalThrottling);
        }

        // Check power limits
        let total_power = gpu_metrics.values().map(|m| m.power_draw).sum::<f64>();
        if total_power > thresholds.max_power_consumption_watts {
            return Some(BottleneckType::PowerLimited);
        }

        // Check GPU utilization
        let avg_gpu_util = gpu_metrics.values().map(|m| m.utilization).sum::<f64>() / gpu_metrics.len() as f64;
        if avg_gpu_util > thresholds.max_gpu_utilization {
            return Some(BottleneckType::GpuBound);
        }

        // Check memory usage
        let total_memory_used = gpu_metrics.values().map(|m| m.memory_used_mb).sum::<f64>();
        let total_memory = gpu_metrics.values().map(|m| m.memory_total_mb).sum::<f64>();
        let memory_usage_percent = (total_memory_used / total_memory) * 100.0;
        if memory_usage_percent > thresholds.max_memory_usage_percent {
            return Some(BottleneckType::MemoryBound);
        }

        // Check CPU utilization
        if system_metrics.cpu_usage_percent > 90.0 {
            return Some(BottleneckType::CpuBound);
        }

        // Check PCIe bandwidth
        let avg_pcie_util = gpu_metrics.values().map(|m| m.pcie_bandwidth_utilization).sum::<f64>() / gpu_metrics.len() as f64;
        if avg_pcie_util > 80.0 {
            return Some(BottleneckType::PcieBandwidth);
        }

        // Check NUMA imbalance
        if system_metrics.numa_imbalance_score > 0.7 {
            return Some(BottleneckType::NumAImbalance);
        }

        None
    }

    /// Get current performance summary
    pub fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        let history = self.metrics_history.lock().unwrap();
        let gpu_metrics = self.gpu_metrics.lock().unwrap();
        let system_metrics = self.system_metrics.lock().unwrap();

        if history.is_empty() {
            return Err(anyhow::anyhow!("No performance data available"));
        }

        let latest = history.last().unwrap();
        let elapsed = self.start_time.elapsed();

        // Calculate averages over last 10 snapshots
        let recent_snapshots: Vec<_> = history.iter().rev().take(10).collect();
        let avg_ops_per_second = recent_snapshots.iter()
            .map(|s| s.ops_per_second)
            .sum::<f64>() / recent_snapshots.len() as f64;

        let avg_gpu_utilization = recent_snapshots.iter()
            .map(|s| s.gpu_utilization_percent)
            .sum::<f64>() / recent_snapshots.len() as f64;

        Ok(PerformanceSummary {
            uptime_seconds: elapsed.as_secs(),
            current_ops_per_second: latest.ops_per_second,
            average_ops_per_second: avg_ops_per_second,
            peak_ops_per_second: history.iter().map(|s| s.ops_per_second).fold(0.0, f64::max),
            total_operations: latest.total_ops,
            gpu_utilization_percent: avg_gpu_utilization,
            memory_usage_mb: latest.memory_usage_mb,
            temperature_celsius: latest.temperature_celsius,
            power_consumption_watts: latest.power_consumption_watts,
            active_gpus: latest.active_devices,
            bottleneck: latest.bottleneck_type.clone(),
            efficiency_score: Self::calculate_efficiency_score(latest, &gpu_metrics),
            recommendations: Self::generate_recommendations(latest, &gpu_metrics, &system_metrics),
        })
    }

    /// Calculate overall efficiency score (0.0 to 1.0)
    fn calculate_efficiency_score(snapshot: &PerformanceSnapshot, _gpu_metrics: &HashMap<usize, GpuMetrics>) -> f64 {
        let gpu_efficiency = snapshot.gpu_utilization_percent / 100.0;
        let memory_efficiency = 1.0 - (snapshot.memory_usage_mb / (snapshot.active_devices as f64 * 32768.0));
        let thermal_efficiency = 1.0 - (snapshot.temperature_celsius / 100.0);
        let power_efficiency = 1.0 - (snapshot.power_consumption_watts / (snapshot.active_devices as f64 * 450.0));

        (gpu_efficiency + memory_efficiency + thermal_efficiency + power_efficiency) / 4.0
    }

    /// Generate performance optimization recommendations
    fn generate_recommendations(
        snapshot: &PerformanceSnapshot,
        _gpu_metrics: &HashMap<usize, GpuMetrics>,
        _system_metrics: &SystemMetrics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(bottleneck) = &snapshot.bottleneck_type {
            match bottleneck {
                BottleneckType::ThermalThrottling => {
                    recommendations.push("Reduce GPU clocks or improve cooling".to_string());
                    recommendations.push("Consider enabling thermal throttling protection".to_string());
                }
                BottleneckType::PowerLimited => {
                    recommendations.push("Reduce power limit or improve power delivery".to_string());
                    recommendations.push("Consider power budget optimization".to_string());
                }
                BottleneckType::GpuBound => {
                    recommendations.push("Increase batch sizes or optimize kernels".to_string());
                    recommendations.push("Consider adding more GPUs to the cluster".to_string());
                }
                BottleneckType::MemoryBound => {
                    recommendations.push("Reduce DP table size or enable compression".to_string());
                    recommendations.push("Consider memory optimization techniques".to_string());
                }
                BottleneckType::CpuBound => {
                    recommendations.push("Increase CPU thread count or optimize CPU code".to_string());
                    recommendations.push("Consider CPU affinity optimization".to_string());
                }
                BottleneckType::PcieBandwidth => {
                    recommendations.push("Optimize data transfer patterns".to_string());
                    recommendations.push("Consider NVLink-enabled GPUs for better bandwidth".to_string());
                }
                BottleneckType::NumAImbalance => {
                    recommendations.push("Optimize memory allocation across NUMA nodes".to_string());
                    recommendations.push("Consider NUMA-aware scheduling".to_string());
                }
                BottleneckType::NetworkBound => {
                    recommendations.push("Optimize network communication patterns".to_string());
                    recommendations.push("Consider reducing network traffic or improving bandwidth".to_string());
                }
            }
        }

        // General recommendations based on utilization
        if snapshot.gpu_utilization_percent < 70.0 {
            recommendations.push("GPU utilization is low - consider increasing batch sizes".to_string());
        }

        if snapshot.ops_per_second < 100_000_000.0 { // Less than 100M ops/sec
            recommendations.push("Performance below target - review kernel optimizations".to_string());
        }

        recommendations
    }

    // GPU query methods (placeholders - would interface with actual GPU APIs)
    fn query_gpu_utilization(_device_id: usize) -> Result<f64> { Ok(85.0) }
    fn query_gpu_memory_used(_device_id: usize) -> Result<f64> { Ok(24576.0) } // 24GB used
    fn query_gpu_temperature(_device_id: usize) -> Result<f64> { Ok(72.0) }
    fn query_gpu_power_draw(_device_id: usize) -> Result<f64> { Ok(380.0) }
    fn query_gpu_fan_speed(_device_id: usize) -> Result<f64> { Ok(65.0) }
    fn query_gpu_clock_speed(_device_id: usize) -> Result<u32> { Ok(1890) }
    fn query_gpu_memory_clock(_device_id: usize) -> Result<u32> { Ok(1313) }
    fn query_pcie_utilization(_device_id: usize) -> Result<f64> { Ok(45.0) }

    // System query methods (placeholders)
    fn query_cpu_usage() -> Result<f64> { Ok(45.0) }
    fn query_memory_used() -> Result<f64> { Ok(16.0) }
    fn query_memory_total() -> Result<f64> { Ok(64.0) }
    fn query_network_rx() -> Result<f64> { Ok(125.0) }
    fn query_network_tx() -> Result<f64> { Ok(89.0) }
    fn query_disk_read() -> Result<f64> { Ok(234.0) }
    fn query_disk_write() -> Result<f64> { Ok(156.0) }
    fn calculate_numa_imbalance() -> Result<f64> { Ok(0.3) }
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        PerformanceSummary {
            uptime_seconds: 0,
            current_ops_per_second: 0.0,
            average_ops_per_second: 0.0,
            peak_ops_per_second: 0.0,
            total_operations: 0,
            gpu_utilization_percent: 0.0,
            memory_usage_mb: 0.0,
            temperature_celsius: 0.0,
            power_consumption_watts: 0.0,
            active_gpus: 0,
            bottleneck: None,
            efficiency_score: 0.0,
            recommendations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub uptime_seconds: u64,
    pub current_ops_per_second: f64,
    pub average_ops_per_second: f64,
    pub peak_ops_per_second: f64,
    pub total_operations: u64,
    pub gpu_utilization_percent: f64,
    pub memory_usage_mb: f64,
    pub temperature_celsius: f64,
    pub power_consumption_watts: f64,
    pub active_gpus: usize,
    pub bottleneck: Option<BottleneckType>,
    pub efficiency_score: f64,
    pub recommendations: Vec<String>,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        PerformanceThresholds {
            max_temperature_celsius: 85.0,
            max_gpu_utilization: 95.0,
            max_memory_usage_percent: 90.0,
            min_ops_per_second: 50_000_000.0, // 50M ops/sec minimum
            max_power_consumption_watts: 3600.0, // 8 GPUs * 450W
            bottleneck_detection_threshold: 0.8,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        SystemMetrics {
            cpu_usage_percent: 0.0,
            memory_used_gb: 0.0,
            memory_total_gb: 0.0,
            network_rx_mbps: 0.0,
            network_tx_mbps: 0.0,
            disk_read_mbps: 0.0,
            disk_write_mbps: 0.0,
            numa_imbalance_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert!(monitor.metrics_history.lock().unwrap().is_empty());
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            total_ops: 1000000,
            ops_per_second: 100000.0,
            memory_usage_mb: 16384.0,
            gpu_utilization_percent: 80.0,
            cpu_utilization_percent: 50.0,
            network_bandwidth_mbps: 200.0,
            temperature_celsius: 70.0,
            power_consumption_watts: 3200.0,
            active_devices: 8,
            bottleneck_type: None,
        };

        let gpu_metrics = HashMap::new();
        let efficiency = PerformanceMonitor::calculate_efficiency_score(&snapshot, &gpu_metrics);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut gpu_metrics = HashMap::new();
        gpu_metrics.insert(0, GpuMetrics {
            device_id: 0,
            name: "Test GPU".to_string(),
            utilization: 50.0,
            memory_used_mb: 16000.0,
            memory_total_mb: 32768.0,
            temperature: 90.0, // High temperature
            power_draw: 400.0,
            fan_speed_percent: 80.0,
            clock_speed_mhz: 1800,
            memory_clock_mhz: 1300,
            pcie_bandwidth_utilization: 50.0,
        });

        let system_metrics = SystemMetrics::default();
        let thresholds = PerformanceThresholds::default();

        let bottleneck = PerformanceMonitor::detect_bottleneck(&gpu_metrics, &system_metrics, &thresholds);
        assert_eq!(bottleneck, Some(BottleneckType::ThermalThrottling));
    }
}
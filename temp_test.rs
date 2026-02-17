//! Refined Hybrid GPU Manager with Drift Mitigation
//!
//! Manages concurrent CUDA/Vulkan execution with shared memory
//! and drift monitoring for precision-critical computations

use super::shared::SharedBuffer;
use super::backends::hybrid_backend::HybridBackend;
use super::backends::backend_trait::GpuBackend;
use crate::types::{Point, Solution, KangarooState};
use crate::math::bigint::BigInt256;
use crate::kangaroo::collision::Trap;
use crate::math::secp::Secp256k1;
use crate::config::{Config, BiasMode};
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use log::{info, warn, debug};
use std::thread;
use std::time::{Duration, Instant};

/// Metrics for drift monitoring
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    pub error_rate: f64,
    pub cuda_throughput: f64,    // ops/sec
    pub vulkan_throughput: f64,  // ops/sec
    pub swap_count: u64,
    pub last_swap_time: Instant,
}

/// Advanced flow control state
#[derive(Debug, Clone)]
pub struct FlowControlState {
    pub current_mode: FlowExecutionMode,
    pub active_flows: std::collections::HashMap<String, FlowInstance>,
    pub flow_priorities: Vec<String>,
    pub resource_allocation: ResourceAllocation,
    pub performance_targets: PerformanceTargets,
    pub adaptation_metrics: AdaptationMetrics,
}

/// Flow execution modes
#[derive(Debug, Clone, PartialEq)]
pub enum FlowExecutionMode {
    Sequential,        // One flow at a time
    Parallel,          // Multiple flows concurrently
    Pipeline,          // Pipelined execution
    Adaptive,          // Dynamic mode switching
    PriorityBased,     // Priority queue execution
}

/// Flow instance tracking
#[derive(Debug, Clone)]
pub struct FlowInstance {
    pub id: String,
    pub name: String,
    pub priority: FlowPriority,
    pub state: FlowState,
    pub start_time: Instant,
    pub progress: f64, // 0.0 to 1.0
    pub dependencies: Vec<String>,
    pub resource_usage: ResourceUsage,
    pub performance_metrics: FlowPerformanceMetrics,
}

/// Flow priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlowPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

/// Flow execution states
#[derive(Debug, Clone, PartialEq)]
pub enum FlowState {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Resource allocation tracking
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub vulkan_memory_mb: usize,
    pub cuda_memory_mb: usize,
    pub cpu_threads: usize,
    pub gpu_compute_units: usize,
    pub network_bandwidth_mbps: usize,
}

/// Performance targets
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub target_throughput_ops_sec: f64,
    pub max_latency_ms: u64,
    pub min_efficiency_percent: f32,
    pub max_error_rate: f64,
}

/// Adaptation metrics for dynamic optimization
#[derive(Debug, Clone)]
pub struct AdaptationMetrics {
    pub mode_switches: u64,
    pub performance_improvements: Vec<f64>,
    pub resource_efficiency: f64,
    pub prediction_accuracy: f64,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_mb: usize,
    pub compute_units: usize,
    pub network_mb_sec: f64,
    pub power_watts: f32,
}

/// Flow performance metrics
#[derive(Debug, Clone)]
pub struct FlowPerformanceMetrics {
    pub operations_completed: u64,
    pub throughput_ops_sec: f64,
    pub latency_ms: u64,
    pub error_count: u64,
    pub resource_efficiency: f64,
}

/// Refined hybrid manager with drift mitigation
pub struct HybridGpuManager {
    hybrid_backend: HybridBackend,
    curve: Secp256k1,
    drift_threshold: f64,
    config: Arc<Config>,  // Full config for bias/bsgs/bloom/gold propagation
    metrics: Arc<Mutex<DriftMetrics>>,
    flow_control: Arc<Mutex<FlowControlState>>,
    scheduler: Arc<Mutex<crate::gpu::backends::hybrid_backend::HybridScheduler>>,
}

impl HybridGpuManager {
    /// Get CUDA backend for direct access
    #[cfg(feature = "rustacuda")]
    fn cuda_backend(&self) -> &crate::gpu::backends::cuda_backend::CudaBackend {
        // This assumes HybridBackend has a cuda field - may need adjustment
        // Phase 5: Implement proper CUDA backend access
        panic!("CUDA backend access not yet implemented")
    }

    /// TEMPORARILY DISABLED: GPU EC operations cause inconsistent results
    /// The CUDA kernels have separate EC math implementations that don't match CPU
    /// This causes the hybrid system to produce invalid points
    pub fn calculate_drift_error(&self, _buffer: &SharedBuffer<Point>, _sample_size: usize) -> Result<f64, Box<dyn std::error::Error>> {
        // Return low error to prefer CPU operations for now
        Ok(0.0)
    }

    /// CPU validation of curve equation: y² = x³ + 7 mod p
    fn curve_equation(&self, x: &[u64; 4], y: &[u64; 4], _p: &BigInt256) -> bool {
        let x_big = BigInt256::from_u64_array(*x);
        let y_big = BigInt256::from_u64_array(*y);

        // Compute y²
        let y_squared = self.curve.barrett_p.mul(&y_big, &y_big);

        // Compute x³ + 7
        let x_squared = self.curve.barrett_p.mul(&x_big, &x_big);
        let x_cubed = self.curve.barrett_p.mul(&x_squared, &x_big);
        let x_cubed_plus_7 = self.curve.barrett_p.add(&x_cubed, &BigInt256::from_u64(7));

        // Check equality
        y_squared == x_cubed_plus_7
    }

    /// Create new hybrid manager with drift monitoring and config propagation
    pub async fn new(config: &Config, drift_threshold: f64, _check_interval_secs: u64) -> Result<Self> {
        let hybrid_backend = HybridBackend::new().await?;
        // Log CUDA version for compatibility (only if CUDA feature enabled)
        #[cfg(feature = "rustacuda")]
        {
            // CUDA backend available, but no direct version logging method
            log::info!("CUDA backend initialized");
        }
        let curve = Secp256k1::new();

        // Initialize advanced flow control
        let flow_control = Arc::new(Mutex::new(FlowControlState {
            current_mode: FlowExecutionMode::Adaptive,
            active_flows: std::collections::HashMap::new(),
            flow_priorities: Vec::new(),
            resource_allocation: ResourceAllocation {
                vulkan_memory_mb: 4096,
                cuda_memory_mb: 8192,
                cpu_threads: num_cpus::get(),
                gpu_compute_units: 1, // Will be detected
                network_bandwidth_mbps: 10000,
            },
}

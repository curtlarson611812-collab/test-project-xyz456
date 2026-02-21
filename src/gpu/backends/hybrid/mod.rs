//! Hybrid GPU backend implementation
//!
//! Intelligent dispatch between Vulkan (bulk) and CUDA (precision) backends
//! with advanced multi-GPU coordination and DP checking integration

// Re-export all hybrid backend modules
pub mod buffers;
pub mod cluster;
pub mod communication;
pub mod dispatch;
pub mod dp_integration;
pub mod execution;
pub mod load_balancer;
pub mod monitoring;
pub mod operations;
pub mod performance;
pub mod power;
pub mod scheduling;
pub mod topology;

/// Hybrid operations that can be dispatched to different GPU backends
#[derive(Debug, Clone)]
pub enum HybridOperation {
    BatchInverse(Vec<[u32; 8]>, [u32; 8]),
    BatchBarrettReduce(Vec<[u32; 16]>, [u32; 9], [u32; 8], bool),
    BatchBigIntMul(Vec<[u32; 8]>, Vec<[u32; 8]>),
    StepBatch(Vec<[[u32; 8]; 3]>, Vec<[u32; 8]>, Vec<u32>),
    SolveCollision(
        Vec<[u32; 8]>,
        Vec<[u32; 8]>,
        Vec<[u32; 8]>,
        Vec<[u32; 8]>,
        Vec<[u32; 8]>,
        [u32; 8],
    ),
    // Additional operations for comprehensive coverage
    Inverse([u32; 8], [u32; 8]), // Single modular inverse
    DpCheck(Vec<[u32; 8]>, u32), // DP checking with mask
    BigIntMul([u32; 8], [u32; 8]), // Single bigint multiplication
    BarrettReduce([u32; 8], [u32; 8], [u32; 8]), // Single Barrett reduction
    BsgsSolve([u32; 8], [u32; 8], [u32; 8]), // BSGS solving
    Custom(Vec<u8>, Vec<u8>), // Custom operations
}

// Re-export key types for API compatibility
pub use buffers::{CpuStagingBuffer, UnifiedGpuBuffer, CommandBufferCache, SharedBuffer, ExternalMemoryHandle};
pub use cluster::{GpuCluster, GpuDevice, GpuTopology, GpuApiType, WorkloadPattern, PerformanceSnapshot};
pub use communication::CrossGpuCommunication;
pub use dispatch::HybridBackend;
pub use dp_integration::DpIntegrationManager;
pub use execution::{FlowPipeline, FlowStage};
pub use load_balancer::AdaptiveLoadBalancer;
pub use monitoring::{HybridOperationMetrics, NsightRuleResult, PipelinePerformanceSummary, StagePerformanceSummary};
pub use scheduling::HybridScheduler;
pub use operations::{HybridOperations, HybridOperationsImpl};
pub use performance::{PerformanceOperations, PerformanceOperationsImpl, ExtendedGpuConfig};
pub use power::PowerManager;
pub use topology::TopologyManager;

// Scheduler types are defined in this module

/// Backend selection strategy for operation execution
#[derive(Debug)]
pub enum BackendSelection {
    Single(String),
    Redundant(Vec<String>),
    Adaptive(Vec<String>),
}

/// Scheduling context for operation execution
#[derive(Debug)]
pub struct SchedulingContext {
    pub vulkan_load: BackendLoad,
    pub cuda_load: BackendLoad,
    pub thermal_state: f64,
    pub power_budget: f64,
    pub system_memory_pressure: f64,
    pub thermal_throttling_active: bool,
}

/// Backend load information
#[derive(Debug, Clone)]
pub struct BackendLoad {
    pub backend_name: String,
    pub active_operations: usize,
    pub queue_depth: usize,
    pub memory_usage_percent: f64,
    pub compute_utilization_percent: f64,
}

// Re-export enums and supporting types
pub use cluster::{BalancingStrategy, SharedMemoryRegion, ResultAggregator, AggregationStrategy, GpuResult, PatternType};

// Re-export hybrid-specific types that were in the original monolithic file
#[derive(Debug)]
pub enum WorkResult {
    BatchInverse(Vec<Option<[u32; 8]>>),
    BatchBarrettReduce(Vec<[u32; 8]>),
    BatchBigIntMul(Vec<[u32; 16]>),
    StepBatch(Vec<super::super::backend_trait::Trap>),
    SolveCollision(Vec<Option<[u32; 8]>>),
    Error(anyhow::Error),
}

#[derive(Clone, Debug)]
pub enum WorkPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Clone, Debug)]
pub enum BackendPreference {
    Auto,
    Any,
    Vulkan,
    Cuda,
    Cpu,
}

/// Scheduling policy for operation distribution
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// Optimize for throughput (batch processing)
    Throughput,
    /// Optimize for latency (real-time responses)
    Latency,
    /// Balance throughput and latency
    Balanced,
    /// Custom policy with specific parameters
    Custom { priority_weight: f64, load_balance_weight: f64 },
}



/// Work item for out-of-order execution
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: u64,
    pub operation: HybridOperation,
    pub priority: WorkPriority,
    pub dependencies: Vec<u64>, // IDs of work items this depends on
    pub backend_preference: BackendPreference,
    pub estimated_duration: std::time::Duration,
    pub submitted_at: std::time::Instant,
}

/// OOO execution queue with dependency tracking
#[derive(Debug)]
pub struct OooExecutionQueue {
    work_queue: std::collections::VecDeque<WorkItem>,
    active_work: std::collections::HashMap<u64, tokio::task::JoinHandle<WorkResult>>,
    completed_work: std::collections::HashMap<u64, WorkResult>,
    dependency_graph: std::collections::HashMap<u64, Vec<u64>>, // work_id -> dependent_work_ids
}


impl WorkResult {
    pub fn device_id(&self) -> usize {
        // Placeholder - would extract device ID from result
        0
    }
}
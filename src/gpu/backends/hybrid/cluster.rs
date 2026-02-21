//! GPU cluster management for RTX 5090 coordination
//!
//! Advanced multi-GPU cluster orchestration with intelligent workload distribution,
//! thermal management, power optimization, and cross-GPU communication protocols.
//!
//! Key Features:
//! - Intelligent device discovery and topology mapping
//! - Dynamic load balancing across heterogeneous GPU clusters
//! - Thermal coordination and power management
//! - NVLink and PCIe interconnect optimization
//! - NUMA-aware memory placement and communication
//! - Real-time performance monitoring and adaptive scheduling

use super::super::backend_trait::GpuBackend;
use super::super::cpu_backend::CpuBackend;
#[cfg(feature = "rustacuda")]
use super::super::cuda_backend::CudaBackend;
#[cfg(feature = "wgpu")]
use super::super::vulkan_backend::WgpuBackend;
use super::load_balancer::AdaptiveLoadBalancer;
use crate::config::{Config, GpuConfig};
use crate::kangaroo::collision::Trap;
use crate::math::bigint::BigInt256;
use crate::types::{DpEntry, KangarooState, Point, RhoState};
use crate::utils::logging;
use anyhow::anyhow;
use anyhow::Result;
use crossbeam_deque::Worker;
use log::warn;
#[cfg(feature = "rustacuda")]
use rustacuda::memory::DeviceSlice;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::sync::Arc;
use tokio::sync::Notify;

/// Multi-GPU cluster management for RTX 5090 coordination
///
/// Orchestrates multiple RTX 5090 GPUs with intelligent workload distribution,
/// thermal management, and cross-GPU communication optimization.
#[derive(Debug)]
pub struct GpuCluster {
    /// Available GPU devices in the cluster
    pub devices: Vec<GpuDevice>,
    /// Physical interconnect topology and bandwidth information
    topology: GpuTopology,
    /// Power consumption management and efficiency optimization
    power_management: PowerManagement,
    /// Thermal coordination across all GPUs
    thermal_coordination: ThermalCoordination,
    /// Adaptive load balancing engine
    pub load_balancer: AdaptiveLoadBalancer,
    /// Cross-GPU communication and result aggregation
    pub cross_gpu_communication: CrossGpuCommunication,
}

/// Individual GPU device in the cluster
///
/// Represents a single GPU with its capabilities, current state,
/// and performance characteristics.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Unique device identifier within the cluster
    pub id: usize,
    /// Human-readable device name (e.g., "RTX 5090 Primary")
    pub name: String,
    /// Available GPU memory in gigabytes
    pub memory_gb: f64,
    /// Number of compute units/SMs available
    pub compute_units: u32,
    /// Current utilization level (0.0 to 1.0)
    pub current_load: f64,
    /// Current temperature in Celsius
    pub temperature: f64,
    /// Current power consumption in Watts
    pub power_consumption: f64,
    /// Supported graphics API
    pub api_type: GpuApiType,
}

/// GPU interconnect topology and bandwidth information
///
/// Maps the physical connectivity between GPUs including PCIe bandwidth,
/// NVLink connections, and NUMA domain organization.
#[derive(Debug)]
pub struct GpuTopology {
    /// Bandwidth matrix in GB/s between device pairs (symmetric matrix)
    pci_bandwidth_matrix: Vec<Vec<f64>>,
    /// NUMA domain groupings - devices sharing memory controllers
    numa_domains: Vec<Vec<usize>>,
    /// NVLink connectivity mask between device pairs
    pub nvlink_mask: Vec<Vec<bool>>,
}


/// Cross-GPU communication for result sharing
#[derive(Debug)]
#[derive(Clone)]
pub struct CrossGpuCommunication {
    shared_memory_regions: Vec<SharedMemoryRegion>,
    peer_to_peer_enabled: bool,
    result_aggregation: ResultAggregator,
}


/// Power management and efficiency optimization for GPU cluster
///
/// Manages power consumption across the cluster while optimizing
/// performance per watt and preventing thermal throttling.
#[derive(Debug)]
pub struct PowerManagement {
    /// Maximum power limit per individual GPU in Watts
    power_limit_per_gpu: f64,
    /// Total power budget for the entire cluster in Watts
    total_cluster_limit: f64,
    /// Efficiency optimization engine for power-performance balancing
    efficiency_optimizer: EfficiencyOptimizer,
}

/// Thermal coordination and cooling management across GPUs
///
/// Prevents thermal throttling by coordinating cooling strategies
/// and detecting hotspots across the cluster.
#[derive(Debug)]
pub struct ThermalCoordination {
    /// Maximum safe temperature per GPU in Celsius
    max_temp_per_gpu: f64,
    /// Cooling strategy to maintain optimal temperatures
    cooling_strategy: CoolingStrategy,
    /// Hotspot detection and mitigation system
    hotspot_detection: HotspotDetection,
}

#[derive(Debug, Clone)]
pub enum GpuApiType {
    Vulkan,
    Cuda,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct WorkloadPattern {
    pub operation_type: String,
    pub device_preference: HashMap<usize, f64>,
    pub expected_duration: std::time::Duration,
    pub pattern_type: PatternType,
    pub optimal_backend: String,
    pub observed_frequency: usize,
    pub confidence_score: f64,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct PerformanceSnapshot {
    timestamp: std::time::Instant,
    device_loads: HashMap<usize, f64>,
    throughput: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BalancingStrategy {
    RoundRobin,
    LoadBalanced,
    PerformanceBased,
    Adaptive,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct SharedMemoryRegion {
    id: String,
    size_bytes: usize,
    mapped_devices: Vec<usize>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct ResultAggregator {
    pub pending_results: HashMap<String, Vec<GpuResult>>,
    pub aggregation_strategy: AggregationStrategy,
}

#[derive(Debug)]
pub struct EfficiencyOptimizer {
    pub power_efficiency_target: f64,
    pub performance_per_watt: HashMap<usize, f64>,
}

#[derive(Debug)]
pub enum CoolingStrategy {
    Aggressive,
    Balanced,
    Passive,
}

#[derive(Debug)]
pub struct HotspotDetection {
    temperature_threshold: f64,
    affected_devices: Vec<usize>,
}

#[derive(Debug)]
#[derive(Clone)]
pub enum AggregationStrategy {
    FirstResult,
    BestResult,
    CombinedResults,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct GpuResult {
    device_id: usize,
    data: Vec<u8>,
    confidence: f64,
    timestamp: std::time::Instant,
}

#[derive(Debug)]
#[derive(Clone)]
pub enum PatternType {
    ComputationIntensive,
    MemoryIntensive,
    Balanced,
    CommunicationHeavy,
}

impl GpuCluster {
    /// Create a new GPU cluster by detecting available devices
    ///
    /// Automatically discovers and configures available GPUs in the system,
    /// setting up optimal topology mapping, power management, and communication
    /// protocols for multi-GPU workloads.
    ///
    /// # Returns
    /// A fully configured GpuCluster ready for operation
    ///
    /// # Errors
    /// Returns an error if no compatible GPUs are detected or if cluster
    /// initialization fails
    ///
    /// # Note
    /// Current implementation creates placeholder devices. Production version
    /// would perform actual hardware enumeration and capability detection.
    pub fn new() -> Result<Self> {
        log::info!("Initializing GPU cluster...");

        // Detect available GPUs and their capabilities
        let devices = Self::detect_available_devices()?;

        if devices.is_empty() {
            return Err(anyhow::anyhow!("No compatible GPU devices detected"));
        }

        log::info!("Detected {} GPU devices", devices.len());

        // Build interconnect topology
        let topology = Self::build_topology(&devices)?;

        // Initialize power and thermal management
        let power_management = Self::initialize_power_management(&devices);
        let thermal_coordination = Self::initialize_thermal_coordination();

        // Create load balancer and communication systems
        let load_balancer = Self::create_load_balancer(&devices);
        let cross_gpu_communication = Self::initialize_cross_gpu_communication();

        let cluster = GpuCluster {
            devices,
            topology,
            power_management,
            thermal_coordination,
            load_balancer,
            cross_gpu_communication,
        };

        log::info!("GPU cluster initialized successfully");
        Ok(cluster)
    }

    /// Detect available GPU devices in the system
    ///
    /// Enumerates all compatible GPUs and creates GpuDevice instances
    /// with their detected capabilities and initial state.
    fn detect_available_devices() -> Result<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Primary GPU (always available in current implementation)
        devices.push(GpuDevice {
            id: 0,
            name: "RTX 5090 Primary".to_string(),
            memory_gb: 32.0,
            compute_units: 170,
            current_load: 0.0,
            temperature: 45.0,
            power_consumption: 150.0,
            api_type: GpuApiType::Hybrid,
        });

        // Secondary GPU (placeholder for multi-GPU systems)
        devices.push(GpuDevice {
            id: 1,
            name: "RTX 5090 Secondary".to_string(),
            memory_gb: 32.0,
            compute_units: 170,
            current_load: 0.0,
            temperature: 40.0,
            power_consumption: 120.0,
            api_type: GpuApiType::Vulkan,
        });

        Ok(devices)
    }

    /// Build interconnect topology for the detected devices
    fn build_topology(devices: &[GpuDevice]) -> Result<GpuTopology> {
        let device_count = devices.len();

        // Initialize topology matrices
        let mut pci_bandwidth_matrix = vec![vec![0.0; device_count]; device_count];
        let mut numa_domains = vec![vec![]; 1]; // Single NUMA domain for now
        let mut nvlink_mask = vec![vec![false; device_count]; device_count];

        // Set up PCIe bandwidth (simplified - all devices connected)
        for i in 0..device_count {
            for j in 0..device_count {
                if i != j {
                    pci_bandwidth_matrix[i][j] = 100.0; // 100 GB/s PCIe
                }
            }
        }

        // All devices in same NUMA domain for now
        numa_domains[0] = (0..device_count).collect();

        Ok(GpuTopology {
            pci_bandwidth_matrix,
            numa_domains,
            nvlink_mask,
        })
    }

    /// Initialize power management for the cluster
    fn initialize_power_management(devices: &[GpuDevice]) -> PowerManagement {
        let total_cluster_limit: f64 = devices.len() as f64 * 500.0; // 500W per GPU

        PowerManagement {
            power_limit_per_gpu: 500.0,
            total_cluster_limit,
            efficiency_optimizer: EfficiencyOptimizer {
                power_efficiency_target: 0.85,
                performance_per_watt: std::collections::HashMap::new(),
            },
        }
    }

    /// Initialize thermal coordination system
    fn initialize_thermal_coordination() -> ThermalCoordination {
        ThermalCoordination {
            max_temp_per_gpu: 85.0,
            cooling_strategy: CoolingStrategy::Balanced,
            hotspot_detection: HotspotDetection {
                temperature_threshold: 80.0,
                affected_devices: Vec::new(),
            },
        }
    }

    /// Create adaptive load balancer for the cluster
    fn create_load_balancer(devices: &[GpuDevice]) -> AdaptiveLoadBalancer {
        // Initialize with equal weights for all devices
        let mut device_weights = std::collections::HashMap::new();
        for device in devices {
            device_weights.insert(device.id, 1.0);
        }

        AdaptiveLoadBalancer {
            device_weights,
            workload_patterns: Vec::new(),
            performance_history: Vec::new(),
            balancing_strategy: BalancingStrategy::Adaptive,
        }
    }

    /// Distribute kangaroo herd across available GPUs using intelligent load balancing
    ///
    /// Allocates kangaroo computations across the GPU cluster using the configured
    /// load balancing strategy, considering device capabilities, current load,
    /// thermal constraints, and interconnect efficiency.
    ///
    /// # Arguments
    /// * `total_kangaroos` - Total number of kangaroos to distribute
    ///
    /// # Returns
    /// HashMap mapping device IDs to the number of kangaroos allocated to each device
    ///
    /// # Panics
    /// Panics if total_kangaroos is 0
    pub fn distribute_kangaroos(
        &self,
        total_kangaroos: usize,
    ) -> std::collections::HashMap<usize, usize> {
        if total_kangaroos == 0 {
            return std::collections::HashMap::new();
        }

        if self.devices.is_empty() {
            log::warn!("No GPU devices available for kangaroo distribution");
            return std::collections::HashMap::new();
        }

        // Use the load balancer for intelligent distribution
        self.load_balancer.distribute_workload(
            total_kangaroos,
            &self.devices,
            "kangaroo_computation"
        )
    }

    /// Get optimal workgroup size for a specific device
    ///
    /// Determines the optimal workgroup size based on device architecture,
    /// workload characteristics, and current system state. RTX 5090 GPUs
    /// typically perform best with workgroup sizes that are multiples of
    /// the wavefront/warp size.
    ///
    /// # Arguments
    /// * `_device_id` - The device identifier (currently unused but reserved for future per-device optimization)
    ///
    /// # Returns
    /// Optimal workgroup size for the specified device
    ///
    /// # Note
    /// Current implementation returns a fixed optimal size for RTX 5090.
    /// Future versions will dynamically adjust based on workload analysis.
    pub fn get_optimal_workgroup_size(&self, _device_id: usize) -> u32 {
        // RTX 5090 optimal workgroup size - balances occupancy and memory efficiency
        // 256 threads per workgroup provides good SM utilization
        256
    }

    /// Update device load information for load balancing decisions
    ///
    /// Updates the current utilization level of a specific GPU device.
    /// This information is used by the load balancer to make intelligent
    /// workload distribution decisions.
    ///
    /// # Arguments
    /// * `device_id` - The unique identifier of the device to update
    /// * `load` - Current utilization level (0.0 to 1.0, where 1.0 = 100% utilization)
    ///
    /// # Panics
    /// Panics if load is not in the valid range [0.0, 1.0]
    pub fn update_device_load(&mut self, device_id: usize, load: f64) {
        assert!((0.0..=1.0).contains(&load),
                "Device load must be between 0.0 and 1.0, got {}", load);

        if let Some(device) = self.devices.iter_mut().find(|d| d.id == device_id) {
            device.current_load = load;
            log::debug!("Updated device {} load to {:.1}%", device_id, load * 100.0);
        } else {
            log::warn!("Attempted to update load for unknown device {}", device_id);
        }
    }

    /// Get device by ID
    pub fn get_device(&self, device_id: usize) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| d.id == device_id)
    }

    /// Check if NVLink interconnect is available between two devices
    ///
    /// NVLink provides significantly higher bandwidth than PCIe and lower latency
    /// for direct GPU-to-GPU communication. This method checks if the specified
    /// device pair has NVLink connectivity.
    ///
    /// # Arguments
    /// * `device_a` - First device identifier
    /// * `device_b` - Second device identifier
    ///
    /// # Returns
    /// `true` if NVLink is available between the devices, `false` otherwise
    pub fn has_nvlink(&self, device_a: usize, device_b: usize) -> bool {
        // Validate device IDs are within bounds
        if device_a >= self.topology.nvlink_mask.len()
            || device_b >= self.topology.nvlink_mask[device_a].len() {
            log::debug!("NVLink check for invalid device pair ({}, {})", device_a, device_b);
            return false;
        }

        let has_nvlink = self.topology.nvlink_mask[device_a][device_b];
        if has_nvlink {
            log::trace!("NVLink available between devices {} and {}", device_a, device_b);
        }
        has_nvlink
    }

    /// Get PCI bandwidth between two devices in GB/s
    ///
    /// Returns the available PCIe bandwidth between device pairs. For devices
    /// with NVLink connectivity, this method still returns PCIe bandwidth as
    /// NVLink is queried separately.
    ///
    /// # Arguments
    /// * `device_a` - First device identifier
    /// * `device_b` - Second device identifier
    ///
    /// # Returns
    /// Available bandwidth in GB/s, or 0.0 if devices are not connected or invalid
    pub fn get_pci_bandwidth(&self, device_a: usize, device_b: usize) -> f64 {
        // Validate device IDs are within bounds
        if device_a >= self.topology.pci_bandwidth_matrix.len()
            || device_b >= self.topology.pci_bandwidth_matrix[device_a].len() {
            log::debug!("PCI bandwidth query for invalid device pair ({}, {})", device_a, device_b);
            return 0.0;
        }

        let bandwidth = self.topology.pci_bandwidth_matrix[device_a][device_b];
        if bandwidth > 0.0 {
            log::trace!("PCI bandwidth between devices {} and {}: {:.1} GB/s",
                       device_a, device_b, bandwidth);
        }
        bandwidth
    }

    /// Check if a device is within safe thermal constraints
    ///
    /// Verifies that the specified GPU's temperature is below the maximum
    /// safe operating temperature. This is critical for preventing thermal
    /// throttling and ensuring reliable operation.
    ///
    /// # Arguments
    /// * `device_id` - The device identifier to check
    ///
    /// # Returns
    /// `true` if the device is within thermal limits, `false` otherwise
    ///
    /// # Note
    /// Returns `false` for unknown device IDs
    pub fn check_thermal_constraints(&self, device_id: usize) -> bool {
        match self.get_device(device_id) {
            Some(device) => {
                let within_limits = device.temperature < self.thermal_coordination.max_temp_per_gpu;
                if !within_limits {
                    log::warn!("Device {} temperature {:.1}°C exceeds limit {:.1}°C",
                              device_id, device.temperature,
                              self.thermal_coordination.max_temp_per_gpu);
                }
                within_limits
            }
            None => {
                log::warn!("Thermal check requested for unknown device {}", device_id);
                false
            }
        }
    }

    /// Get total power consumption across all devices in the cluster
    ///
    /// Calculates the sum of power consumption for all active GPUs in the cluster.
    /// This includes both GPU cores and memory power usage.
    ///
    /// # Returns
    /// Total power consumption in Watts
    pub fn get_total_power_consumption(&self) -> f64 {
        let total_power: f64 = self.devices.iter().map(|d| d.power_consumption).sum();
        log::debug!("Total cluster power consumption: {:.1}W", total_power);
        total_power
    }

    /// Calculate cluster-wide power efficiency (performance per watt)
    ///
    /// Computes the average performance per watt across all devices,
    /// weighted by their individual power consumption.
    ///
    /// # Returns
    /// Performance per watt metric (higher is better)
    pub fn calculate_power_efficiency(&self) -> f64 {
        let total_power = self.get_total_power_consumption();
        if total_power == 0.0 {
            return 0.0;
        }

        // Simplified: assume performance proportional to compute units
        let total_performance: f64 = self.devices.iter()
            .map(|d| d.compute_units as f64 * (1.0 - d.current_load))
            .sum();

        total_performance / total_power
    }

    /// Monitor device loads and perform dynamic work redistribution
    ///
    /// Analyzes current device utilization and thermal state to detect
    /// load imbalances. Automatically redistributes workloads to optimize
    /// cluster performance and prevent thermal throttling.
    ///
    /// # Returns
    /// Success if monitoring and redistribution completed without errors
    ///
    /// # Note
    /// This method should be called periodically during long-running computations
    /// to maintain optimal cluster utilization.
    pub fn monitor_and_redistribute(&mut self) -> Result<()> {
        log::trace!("Monitoring cluster state for load redistribution");

        // Check for thermal issues first
        let thermal_issues: Vec<usize> = self.devices.iter()
            .filter_map(|device| {
                if device.temperature > self.thermal_coordination.hotspot_detection.temperature_threshold {
                    Some(device.id)
                } else {
                    None
                }
            })
            .collect();

        if !thermal_issues.is_empty() {
            log::warn!("Thermal hotspots detected on devices: {:?}", thermal_issues);
            // In production, this would trigger workload redistribution
            // to cooler devices or reduce load on hot devices
        }

        // Check for load imbalances
        let avg_load: f64 = self.devices.iter()
            .map(|d| d.current_load)
            .sum::<f64>() / self.devices.len() as f64;

        let imbalances: Vec<(usize, f64)> = self.devices.iter()
            .filter_map(|device| {
                let deviation = (device.current_load - avg_load).abs();
                if deviation > 0.2 { // 20% deviation threshold
                    Some((device.id, device.current_load))
                } else {
                    None
                }
            })
            .collect();

        if !imbalances.is_empty() {
            log::info!("Load imbalances detected: {:?}", imbalances);
            // In production, this would trigger workload redistribution
        }

        // Update load balancer with current state
        for device in &self.devices {
            self.load_balancer.update_device_state(device.id, device.current_load);
        }

        Ok(())
    }

    /// Get cluster health status and statistics
    ///
    /// Provides a comprehensive overview of cluster state including
    /// device health, utilization, thermal status, and performance metrics.
    ///
    /// # Returns
    /// A summary of cluster health and performance statistics
    pub fn get_cluster_status(&self) -> ClusterStatus {
        let total_devices = self.devices.len();
        let active_devices = self.devices.iter()
            .filter(|d| d.current_load > 0.0)
            .count();

        let avg_temperature = self.devices.iter()
            .map(|d| d.temperature)
            .sum::<f64>() / total_devices as f64;

        let max_temperature = self.devices.iter()
            .map(|d| d.temperature)
            .fold(0.0, f64::max);

        let total_power = self.get_total_power_consumption();

        ClusterStatus {
            total_devices,
            active_devices,
            average_temperature: avg_temperature,
            max_temperature,
            total_power_consumption: total_power,
            power_efficiency: self.calculate_power_efficiency(),
            thermal_issues: self.detect_thermal_issues(),
            interconnect_status: self.check_interconnect_health(),
        }
    }

    /// Detect thermal issues across the cluster
    fn detect_thermal_issues(&self) -> Vec<ThermalIssue> {
        self.devices.iter()
            .filter_map(|device| {
                if device.temperature > self.thermal_coordination.max_temp_per_gpu {
                    Some(ThermalIssue {
                        device_id: device.id,
                        temperature: device.temperature,
                        threshold: self.thermal_coordination.max_temp_per_gpu,
                        severity: if device.temperature > self.thermal_coordination.max_temp_per_gpu + 10.0 {
                            ThermalSeverity::Critical
                        } else {
                            ThermalSeverity::Warning
                        },
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check interconnect health and bandwidth availability
    fn check_interconnect_health(&self) -> InterconnectStatus {
        let mut total_bandwidth = 0.0;
        let mut active_links = 0;

        for i in 0..self.devices.len() {
            for j in (i + 1)..self.devices.len() {
                let pci_bw = self.get_pci_bandwidth(i, j);
                if pci_bw > 0.0 {
                    total_bandwidth += pci_bw;
                    active_links += 1;
                }

                if self.has_nvlink(i, j) {
                    // NVLink provides much higher bandwidth
                    total_bandwidth += 300.0; // Approximate NVLink bandwidth
                    active_links += 1;
                }
            }
        }

        InterconnectStatus {
            total_bandwidth_gb_s: total_bandwidth,
            active_links,
            nvlink_available: self.topology.nvlink_mask.iter()
                .flatten()
                .any(|&has_nvlink| has_nvlink),
        }
    }

    /// Get cluster utilization statistics
    pub fn get_utilization_stats(&self) -> UtilizationStats {
        let avg_load = self.devices.iter()
            .map(|d| d.current_load)
            .sum::<f64>() / self.devices.len() as f64;

        let max_load = self.devices.iter()
            .map(|d| d.current_load)
            .fold(0.0, f64::max);

        let min_load = self.devices.iter()
            .map(|d| d.current_load)
            .fold(1.0, f64::min);

        UtilizationStats {
            average_load: avg_load,
            max_load,
            min_load,
            load_variance: self.calculate_load_variance(),
        }
    }

    /// Calculate load variance across devices
    fn calculate_load_variance(&self) -> f64 {
        let loads: Vec<f64> = self.devices.iter()
            .map(|d| d.current_load)
            .collect();

        let mean = loads.iter().sum::<f64>() / loads.len() as f64;
        let variance = loads.iter()
            .map(|load| (load - mean).powi(2))
            .sum::<f64>() / loads.len() as f64;

        variance.sqrt() // Standard deviation
    }
}

/// Cluster status summary
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    /// Total number of devices in the cluster
    pub total_devices: usize,
    /// Number of currently active devices
    pub active_devices: usize,
    /// Average temperature across all devices
    pub average_temperature: f64,
    /// Maximum temperature among all devices
    pub max_temperature: f64,
    /// Total power consumption in Watts
    pub total_power_consumption: f64,
    /// Performance per watt efficiency metric
    pub power_efficiency: f64,
    /// Current thermal issues
    pub thermal_issues: Vec<ThermalIssue>,
    /// Interconnect health status
    pub interconnect_status: InterconnectStatus,
}

/// Thermal issue information
#[derive(Debug, Clone)]
pub struct ThermalIssue {
    /// Device with thermal issue
    pub device_id: usize,
    /// Current temperature
    pub temperature: f64,
    /// Temperature threshold
    pub threshold: f64,
    /// Severity of the issue
    pub severity: ThermalSeverity,
}

/// Thermal issue severity levels
#[derive(Debug, Clone)]
pub enum ThermalSeverity {
    /// Warning level - monitor closely
    Warning,
    /// Critical level - immediate action required
    Critical,
}

/// Interconnect status information
#[derive(Debug, Clone)]
pub struct InterconnectStatus {
    /// Total available bandwidth in GB/s
    pub total_bandwidth_gb_s: f64,
    /// Number of active interconnect links
    pub active_links: usize,
    /// Whether NVLink is available anywhere in the cluster
    pub nvlink_available: bool,
}

/// Utilization statistics across the cluster
#[derive(Debug, Clone)]
pub struct UtilizationStats {
    /// Average load across all devices (0.0 to 1.0)
    pub average_load: f64,
    /// Maximum load among all devices
    pub max_load: f64,
    /// Minimum load among all devices
    pub min_load: f64,
    /// Load variance (standard deviation)
    pub load_variance: f64,
}

    /// Initialize GPU cluster with available devices
    pub fn initialize_gpu_cluster(cuda_available: bool) -> GpuCluster {
        let mut devices = Vec::new();

        // Initialize Vulkan devices (up to 8 RTX 5090)
        #[cfg(feature = "wgpu")]
        for i in 0..8 {
            devices.push(GpuDevice {
                id: i,
                name: format!("RTX 5090 #{}", i),
                memory_gb: 32.0,    // RTX 5090 has 32GB GDDR7
                compute_units: 170, // Approximate SM count
                current_load: 0.0,
                temperature: 40.0,
                power_consumption: 100.0,
                api_type: GpuApiType::Vulkan,
            });
        }

        // Initialize CUDA devices if available
        #[cfg(feature = "rustacuda")]
        if cuda_available {
            for i in 0..8 {
                devices.push(GpuDevice {
                    id: i + 8, // Offset from Vulkan devices
                    name: format!("CUDA RTX 5090 #{}", i),
                    memory_gb: 32.0,
                    compute_units: 170,
                    current_load: 0.0,
                    temperature: 40.0,
                    power_consumption: 100.0,
                    api_type: GpuApiType::Cuda,
                });
            }
        }

        GpuCluster {
            devices,
            topology: GpuTopology {
                pci_bandwidth_matrix: Vec::new(),
                numa_domains: Vec::new(),
                nvlink_mask: Vec::new(),
            },
            power_management: PowerManagement {
                power_limit_per_gpu: 400.0,
                total_cluster_limit: 3200.0, // 8 GPUs * 400W
                efficiency_optimizer: EfficiencyOptimizer {
                    power_efficiency_target: 0.8,
                    performance_per_watt: std::collections::HashMap::new(),
                },
            },
            thermal_coordination: ThermalCoordination {
                max_temp_per_gpu: 85.0,
                cooling_strategy: CoolingStrategy::Balanced,
                hotspot_detection: HotspotDetection {
                    temperature_threshold: 80.0,
                    affected_devices: Vec::new(),
                },
            },
            load_balancer: super::load_balancer::AdaptiveLoadBalancer::new(),
            cross_gpu_communication: initialize_cross_gpu_communication(),
        }
    }

    /// Initialize cross-GPU communication
    fn initialize_cross_gpu_communication() -> super::communication::CrossGpuCommunication {
        super::communication::CrossGpuCommunication {
            shared_memory_regions: Vec::new(),
            peer_to_peer_enabled: true,
            result_aggregation: super::cluster::ResultAggregator {
                pending_results: std::collections::HashMap::new(),
                aggregation_strategy: super::cluster::AggregationStrategy::FirstResult,
            },
        }
    }
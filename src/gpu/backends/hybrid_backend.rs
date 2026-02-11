//! Hybrid Backend Implementation
//!
//! Intelligent dispatch between Vulkan (bulk) and CUDA (precision) backends

use super::backend_trait::GpuBackend;
use super::cpu_backend::CpuBackend;
use crate::types::RhoState;
use crate::kangaroo::collision::Trap;
use crate::config::GpuConfig;
use crate::math::bigint::BigInt256;
use crate::utils::logging;
use anyhow::Result;
use crossbeam_deque::Worker;
use std::sync::Arc;
use tokio::sync::Notify;
use std::collections::HashMap;
use std::fs::read_to_string;
use anyhow::anyhow;

/// Nsight rule scoring and color coding for dynamic optimization
#[derive(Debug, Clone)]
pub struct NsightRuleResult {
    pub rule_name: String,
    pub score: f64,
    pub color: String,
    pub suggestion: String,
}

impl NsightRuleResult {
    pub fn new(rule_name: &str, score: f64, suggestion: &str) -> Self {
        let color = if score > 80.0 {
            "\u{1F7E2}" // Green circle
        } else if score > 60.0 {
            "\u{1F7E1}" // Yellow circle
        } else {
            "\u{1F534}" // Red circle
        };

        Self {
            rule_name: rule_name.to_string(),
            score,
            color: color.to_string(),
            suggestion: suggestion.to_string(),
        }
    }
}

/// Hybrid backend that dispatches operations to appropriate GPUs
/// Uses Vulkan for bulk operations (step_batch) and CUDA for precision math
pub struct HybridBackend {
    #[cfg(feature = "wgpu")]
    vulkan: WgpuBackend,
    #[cfg(feature = "rustacuda")]
    cuda: CudaBackend,
    cpu: CpuBackend,
}

impl HybridBackend {
    /// Create new hybrid backend with all available backends
    pub async fn new() -> Result<Self> {
        let cpu = CpuBackend::new()?;

        #[cfg(feature = "wgpu")]
        let vulkan = WgpuBackend::new().await?;

        #[cfg(feature = "rustacuda")]
        let cuda = CudaBackend::new()?;

        Ok(Self {
            #[cfg(feature = "wgpu")]
            vulkan,
            #[cfg(feature = "rustacuda")]
            cuda,
            cpu,
        })
    }
}

impl HybridBackend {
    // Chunk: Hybrid Shared Init (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: crossbeam_deque::Worker, std::sync::Arc, types::RhoState
    pub fn init_shared_buffer(_capacity: usize) -> Arc<Worker<RhoState>> {
        Arc::new(Worker::new_fifo())  // Lock-free deque
    }
    // Test: Init, push RhoState, pop check

    // Chunk: Hybrid Await Sync (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: tokio::sync::Notify, init_shared_buffer
    pub async fn hybrid_sync(gpu_notify: Notify, shared: Arc<Worker<RhoState>>) -> Vec<RhoState> {
        gpu_notify.notified().await;
        let stealer = shared.stealer();
        let mut collected = Vec::new();
        while let crossbeam_deque::Steal::Success(state) = stealer.steal() {  // Lock-free pop
            collected.push(state);
        }
        collected
    }
    // Test: Notify after mock GPU push, await collect

    /// Profile device performance for dynamic load balancing
    async fn profile_device_performance(&self) -> (f32, f32) {
        // Profile small batch performance to determine relative speeds
        // Returns (cuda_ratio, vulkan_ratio) where cuda_ratio + vulkan_ratio = 1.0

        #[cfg(all(feature = "rustacuda", feature = "wgpu"))]
        {
            // TODO: Implement actual profiling with small test batches
            // For now, assume CUDA is 1.5x faster than Vulkan for mixed workloads
            (0.6, 0.4) // CUDA gets 60%, Vulkan gets 40%
        }

        #[cfg(all(feature = "rustacuda", not(feature = "wgpu")))]
        {
            (1.0, 0.0) // CUDA only
        }

        #[cfg(all(not(feature = "rustacuda"), feature = "wgpu"))]
        {
            (0.0, 1.0) // Vulkan only
        }

        #[cfg(not(any(feature = "rustacuda", feature = "wgpu")))]
        {
            (0.0, 0.0) // CPU only
        }
    }

    // Chunk: Profile Hashrates for Ratio (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: std::time::Instant, cpu_backend::cpu_batch_step, cuda_backend::dispatch_and_update
    pub fn profile_hashrates(config: &GpuConfig) -> (f64, f64) {  // gpu_ops_sec, cpu_ops_sec
        let test_steps = 10000;
        let test_states = vec![RhoState::default(); config.max_kangaroos.min(512)];  // Small for quick
        let jumps = vec![BigInt256::one(); 256];

        // GPU profile
        let gpu_start = std::time::Instant::now();
        // dispatch_and_update(/* device, kernel, test_states.clone(), jumps.clone(), bias, test_steps */).ok();
        let gpu_time = gpu_start.elapsed().as_secs_f64();
        let gpu_hr = (test_steps as f64 * test_states.len() as f64) / gpu_time;

        // CPU profile
        let mut cpu_states = test_states.clone();
        let cpu_start = std::time::Instant::now();
        CpuBackend::cpu_batch_step(&mut cpu_states, test_steps, &jumps);
        let cpu_time = cpu_start.elapsed().as_secs_f64();
        let cpu_hr = (test_steps as f64 * test_states.len() as f64) / cpu_time;

        (gpu_hr, cpu_hr)
    }

    // Chunk: Adjust Frac on Metrics (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: adjust_gpu_frac, scale_kangaroos, profile_hashrates
    pub fn adjust_gpu_frac(config: &mut GpuConfig, util: f64, temp: u32) {  // util from Nsight [0-1], temp from log
        let (gpu_hr, cpu_hr) = Self::profile_hashrates(config);
        let target_ratio = gpu_hr / (gpu_hr + cpu_hr);
        let util_norm = util;  // 0.8 ideal =1.0
        let temp_norm = if temp > 80 { 0.0 } else if temp < 65 { 1.0 } else { (80.0 - temp as f64) / 15.0 };
        let delta = 0.05 * (util_norm - (1.0 - temp_norm));  // Positive if high util/low temp
        config.gpu_frac = (config.gpu_frac + delta).clamp(0.5, 0.9);  // Laptop bounds
        if config.gpu_frac > target_ratio { config.gpu_frac = target_ratio; }  // Cap on profiled
    }

    // Chunk: Dynamic Kangaroo Scaling (src/config.rs)
    // Dependencies: std::process::Command for nvidia-smi
    pub fn scale_kangaroos(config: &mut GpuConfig, util: f64, temp: u32) {
        let output = std::process::Command::new("nvidia-smi").arg("-q").arg("-d").arg("memory").output().ok();
        let mem_str = output.map(|o| String::from_utf8(o.stdout).unwrap_or_default()).unwrap_or_default();
        let used_mem = mem_str.lines().find(|l| l.contains("Used")).and_then(|l| l.split_whitespace().nth(2).and_then(|s| s.parse::<u32>().ok())).unwrap_or(0);
        let avail_mem = 8192 - used_mem;  // 8GB total

        let target_t = (avail_mem as usize * 1024 / 128) * (util as usize / 10 * 6);  // Mem / state_size * occ_factor
        if temp < 65 && util > 0.9 && target_t > config.max_kangaroos {
            config.max_kangaroos = (config.max_kangaroos * 3 / 2).min(4096);
        } else if temp > 75 || used_mem > 6144 {
            config.max_kangaroos /= 2;
        }
    }

    /// Optimized dispatch with dynamic load balancing
    pub async fn dispatch_step_batch(
        &self,
        positions: &mut Vec<[[u32; 8]; 3]>,
        distances: &mut Vec<[u32; 8]>,
        types: &Vec<u32>,
        _batch_size: usize,
    ) -> Result<Vec<Trap>> {
        let (_cuda_ratio, _vulkan_ratio) = self.profile_device_performance().await;

        #[allow(unused_assignments)]
        let mut all_traps = Vec::new();

        #[cfg(feature = "rustacuda")]
        if _cuda_ratio > 0.0 {
            let cuda_batch = ((batch_size as f32) * cuda_ratio) as usize;
            if cuda_batch > 0 {
                // Split data for CUDA processing
                let mut cuda_positions_vec: Vec<[[u32; 8]; 3]> = positions[0..cuda_batch].to_vec();
                let mut cuda_distances_vec: Vec<[u32; 8]> = distances[0..cuda_batch].to_vec();
                let cuda_types_vec: Vec<u32> = types[0..cuda_batch].to_vec();

                match self.cuda.step_batch(&mut cuda_positions_vec, &mut cuda_distances_vec, &cuda_types_vec) {
                    Ok(cuda_traps) => all_traps.extend(cuda_traps),
                    Err(e) => {
                        log::warn!("CUDA batch failed, falling back to CPU: {}", e);
                        // Fallback to CPU for this portion
                        let cpu_traps = self.cpu.step_batch(&mut cuda_positions_vec, &mut cuda_distances_vec, &cuda_types_vec)?;
                        all_traps.extend(cpu_traps);
                    }
                }
            }
        }

        #[cfg(feature = "wgpu")]
        if vulkan_ratio > 0.0 {
            let vulkan_start = ((batch_size as f32) * cuda_ratio) as usize;
            let vulkan_batch = ((batch_size as f32) * vulkan_ratio) as usize;
            let vulkan_end = (vulkan_start + vulkan_batch).min(batch_size);

            if vulkan_end > vulkan_start {
                // Split data for Vulkan processing
                let mut vulkan_positions_vec: Vec<[[u32; 8]; 3]> = positions[vulkan_start..vulkan_end].to_vec();
                let mut vulkan_distances_vec: Vec<[u32; 8]> = distances[vulkan_start..vulkan_end].to_vec();
                let vulkan_types_vec: Vec<u32> = types[vulkan_start..vulkan_end].to_vec();

                match self.vulkan.step_batch(&mut vulkan_positions_vec, &mut vulkan_distances_vec, &vulkan_types_vec) {
                    Ok(vulkan_traps) => all_traps.extend(vulkan_traps),
                    Err(e) => {
                        log::warn!("Vulkan batch failed, falling back to CPU: {}", e);
                        // Fallback to CPU for this portion
                        let cpu_traps = self.cpu.step_batch(&mut vulkan_positions_vec, &mut vulkan_distances_vec, &vulkan_types_vec)?;
                        all_traps.extend(cpu_traps);
                    }
                }
            }
        }

        // If no GPU backends available, use CPU for everything
        #[cfg(not(any(feature = "rustacuda", feature = "wgpu")))]
        {
            all_traps = self.cpu.step_batch(positions, distances, types)?;
        }

        Ok(all_traps)
    }

    // Chunk: Metrics-Based Dynamic Optimization (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: NsightMetrics, generate_metric_based_recommendations
    pub fn optimize_based_on_metrics_placeholder(config: &mut GpuConfig, metrics: &logging::NsightMetrics) {
        Self::optimize_based_on_metrics(config, metrics);
    }

    pub fn optimize_based_on_metrics(config: &mut GpuConfig, metrics: &logging::NsightMetrics) {
        // Apply metric-based optimizations
        let mut optimization_applied = false;

        // Memory-bound optimizations
        if metrics.dram_utilization > 0.8 || metrics.l2_hit_rate < 0.7 {
            // Reduce kangaroo count to fit in cache better
            config.max_kangaroos = (config.max_kangaroos * 3 / 4).max(512);
            log::info!("Applied memory optimization: reduced kangaroos to {} due to high DRAM utilization", config.max_kangaroos);
            optimization_applied = true;
        }

        // Occupancy optimizations
        if metrics.sm_efficiency < 0.7 || metrics.achieved_occupancy < 0.6 {
            // Reduce kangaroo count to improve occupancy
            config.max_kangaroos = (config.max_kangaroos * 4 / 5).max(256);
            log::info!("Applied occupancy optimization: reduced kangaroos to {} due to low SM efficiency", config.max_kangaroos);
            optimization_applied = true;
        }

        // Compute-bound optimizations
        if metrics.alu_utilization > 0.9 && metrics.sm_efficiency > 0.8 {
            // Can handle more parallelism
            config.max_kangaroos = (config.max_kangaroos * 5 / 4).min(4096);
            log::info!("Applied compute optimization: increased kangaroos to {} due to high ALU utilization", config.max_kangaroos);
            optimization_applied = true;
        }

        // Register pressure optimizations
        if metrics.register_usage > 64 {
            config.max_kangaroos = (config.max_kangaroos * 2 / 3).max(256);
            log::info!("Applied register optimization: reduced kangaroos to {} due to high register usage", config.max_kangaroos);
            optimization_applied = true;
        }

        if !optimization_applied && metrics.sm_efficiency > 0.8 && metrics.l2_hit_rate > 0.8 {
            log::info!("GPU performing well - no optimization needed");
        }

        // Log recommendations
        if !metrics.optimization_recommendations.is_empty() {
            log::info!("Nsight recommendations:");
            for rec in &metrics.optimization_recommendations {
                log::info!("  - {}", rec);
            }
        }
    }

    // Chunk: Rule-Based Configuration Adjustment (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: serde_json, std::fs::read_to_string
    pub fn apply_rule_based_adjustments_placeholder(config: &mut GpuConfig) {
        Self::apply_rule_based_adjustments(config);
    }

    pub fn apply_rule_based_adjustments(config: &mut GpuConfig) {
        // Load rule suggestions and apply automatic adjustments
        if let Ok(json_str) = std::fs::read_to_string("suggestions.json") {
            if let Ok(suggestions) = serde_json::from_str::<std::collections::HashMap<String, String>>(&json_str) {
                let mut adjustments_made = Vec::new();

                // Apply specific rule-based adjustments
                if suggestions.values().any(|s| s.contains("Low Coalescing") || s.contains("SoA")) {
                    config.max_kangaroos = (config.max_kangaroos * 3 / 4).max(512);
                    adjustments_made.push("Reduced kangaroos for SoA coalescing optimization");
                }

                if suggestions.values().any(|s| s.contains("High Registers") || s.contains("reduce locals")) {
                    config.max_regs = config.max_regs.min(48);
                    adjustments_made.push("Reduced max registers for occupancy optimization");
                }

                if suggestions.values().any(|s| s.contains("High Divergence") || s.contains("subgroup")) {
                    config.max_kangaroos = (config.max_kangaroos * 4 / 5).max(256);
                    adjustments_made.push("Reduced kangaroos to mitigate divergence impact");
                }

                if suggestions.values().any(|s| s.contains("modular") || s.contains("Barrett")) {
                    config.gpu_frac = (config.gpu_frac * 0.9).max(0.5);
                    adjustments_made.push("Adjusted GPU fraction for modular arithmetic optimization");
                }

                // Log applied adjustments
                for adjustment in &adjustments_made {
                    log::info!("Rule-based adjustment: {}", adjustment);
                }

                if adjustments_made.is_empty() {
                    log::info!("No rule-based adjustments needed - performance looks good");
                }
            }
        }
    }

    // Chunk: Enhanced Scaled Dispatch with Rules and Metrics (src/gpu/backends/hybrid_backend.rs)
    // Dependencies: apply_rule_based_adjustments, optimize_based_on_metrics, load_comprehensive_nsight_metrics
    pub fn dispatch_hybrid_scaled_with_rules_and_metrics(config: &mut GpuConfig, _target: &BigInt256, _range: (BigInt256, BigInt256), total_steps: u64) -> Option<BigInt256> {
        let mut completed = 0;
        let batch_size = 1000000;  // 1M steps/batch
        let mut rules_applied = false;
        let mut metrics_checked = false;

        while completed < total_steps {
            let batch = batch_size.min((total_steps - completed) as usize);

            // Apply rule-based adjustments (once per run)
            if !rules_applied {
                log::info!("Applying Nsight rule-based configuration adjustments...");
                Self::apply_rule_based_adjustments(config);
                rules_applied = true;
            }

            // Load and apply metrics-based optimization
            if !metrics_checked {
                if let Some(metrics) = logging::load_comprehensive_nsight_metrics("ci_metrics.json") {
                    log::info!("Loaded Nsight metrics: SM eff={:.1}%, Occ={:.1}%, L2 hit={:.1}%, DRAM util={:.1}%",
                              metrics.sm_efficiency * 100.0,
                              metrics.achieved_occupancy * 100.0,
                              metrics.l2_hit_rate * 100.0,
                              metrics.dram_utilization * 100.0);

                    Self::optimize_based_on_metrics(config, &metrics);
                    metrics_checked = true;
                }
            }

            // Note: dispatch_hybrid function needs to be implemented or imported
            let result = None; // Placeholder - need to implement dispatch_hybrid
            if let Some(key) = result { return Some(key); }
            completed += batch as u64;

            // Legacy thermal scaling (still useful as fallback)
            let temp = logging::get_avg_temp("temp.log").unwrap_or(70);
            if temp > 75 {
                config.max_kangaroos /= 2;
                log::warn!("Thermal throttling: reduced kangaroos to {} due to high temp ({}°C)", config.max_kangaroos, temp);
            }
        }
        None
    }

    /// Check if this backend supports precision operations (true for CUDA, false for CPU)
    pub fn supports_precision_ops(&self) -> bool {
        #[cfg(feature = "rustacuda")]
        {
            true
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            false
        }
    }

    /// Create shared buffer for Vulkan-CUDA interop (if available)
    /// Falls back to separate allocations if interop not supported
    #[cfg(any(feature = "wgpu", feature = "rustacuda"))]
    pub fn create_shared_buffer(&self, size: usize) -> anyhow::Result<SharedBuffer> {
        #[cfg(feature = "rustacuda")]
        {
            // CUDA buffer allocation
            use crate::gpu::backends::cuda_backend::CudaBackend;
            // For now, return a placeholder - would need actual CUDA buffer
            Err(anyhow::anyhow!("CUDA shared buffers not yet implemented"))
        }
        #[cfg(all(not(feature = "rustacuda"), feature = "wgpu"))]
        {
            // Vulkan-only buffer
            use wgpu::util::DeviceExt;
            let buffer = self.vulkan.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shared_buffer"),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            Ok(SharedBuffer::Vulkan(buffer))
        }
        #[cfg(not(any(feature = "wgpu", feature = "rustacuda")))]
        {
            Err(anyhow::anyhow!("No GPU backends available for shared buffers"))
        }
    }
}

/// Shared buffer enum for Vulkan-CUDA interop
#[cfg(any(feature = "wgpu", feature = "rustacuda"))]
pub enum SharedBuffer {
    #[cfg(feature = "rustacuda")]
    Cuda(()), // Placeholder - would be actual CUDA buffer type
    #[cfg(feature = "wgpu")]
    Vulkan(wgpu::Buffer),
}

#[async_trait::async_trait]
impl GpuBackend for HybridBackend {
    async fn new() -> Result<Self> {
        Self::new().await
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        // Dispatch to Vulkan for bulk precomputation
        #[cfg(feature = "wgpu")]
        {
            self.vulkan.precomp_table(primes, base)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            self.cpu.precomp_table(primes, base)
        }
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        // Dispatch to Vulkan for bulk stepping operations
        #[cfg(feature = "wgpu")]
        {
            self.vulkan.step_batch(positions, distances, types)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            self.cpu.step_batch(positions, distances, types)
        }
    }

    fn step_batch_bias(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>, config: &crate::config::Config) -> Result<Vec<Trap>> {
        // Dispatch to Vulkan for bias-enhanced bulk stepping operations
        #[cfg(feature = "wgpu")]
        {
            self.vulkan.step_batch_bias(positions, distances, types, config)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            self.cpu.step_batch_bias(positions, distances, types, config)
        }
    }

    fn batch_bsgs_solve(&self, deltas: Vec<[[u32;8];3]>, alphas: Vec<[u32;8]>, distances: Vec<[u32;8]>, config: &crate::config::Config) -> Result<Vec<Option<[u32;8]>>> {
        // Dispatch to CUDA for BSGS solving (most efficient for this operation)
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_bsgs_solve(deltas, alphas, distances, config)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            // Fallback to CPU implementation
            self.cpu.batch_bsgs_solve(deltas, alphas, distances, config)
        }
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Dispatch to CUDA for precision inverse operations
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_inverse(inputs, modulus)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_inverse(inputs, modulus)
        }
    }

    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // Dispatch to CUDA for precision solve operations
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_solve(alphas, betas)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_solve(alphas, betas)
        }
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Dispatch to CUDA for complex collision solving
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n)
        }
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // Dispatch to CUDA for modular reduction operations
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_barrett_reduce(x, mu, modulus, use_montgomery)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_barrett_reduce(x, mu, modulus, use_montgomery)
        }
    }

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Dispatch to CUDA for precision multiplication
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_mul(a, b)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_mul(a, b)
        }
    }

    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        // Dispatch to CUDA for affine conversion
        #[cfg(feature = "rustacuda")]
        {
            self.cuda.batch_to_affine(positions, modulus)
        }
        #[cfg(not(feature = "rustacuda"))]
        {
            self.cpu.batch_to_affine(positions, modulus)
        }
    }
}

impl HybridBackend {
    /// Parse Nsight Compute suggestions and apply dynamic tuning
    pub fn apply_nsight_rules(&self, config: &mut GpuConfig) -> Result<Vec<NsightRuleResult>> {
        let suggestions_path = "suggestions.json";

        // Read suggestions from Nsight analysis
        let json_str = read_to_string(suggestions_path)
            .map_err(|e| anyhow!("Failed to read suggestions.json: {}", e))?;

        let sugg_map: HashMap<String, String> = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse suggestions JSON: {}", e))?;

        let mut results = Vec::new();

        // Apply rules based on Nsight suggestions
        for (rule_name, suggestion) in sugg_map.iter() {
            let (score, suggestion_text) = self.parse_rule_suggestion(rule_name, suggestion);
            let result = NsightRuleResult::new(rule_name, score, &suggestion_text);

            // Apply dynamic adjustments based on rule results
            self.apply_rule_adjustment(config, &result)?;

            results.push(result);
        }

        Ok(results)
    }

    /// Parse individual rule suggestion and extract score
    fn parse_rule_suggestion(&self, _rule_name: &str, suggestion: &str) -> (f64, String) {
        // Extract score from suggestion text (assumes format like "85.2% efficient")
        let score = if let Some(pct_pos) = suggestion.find('%') {
            if let Some(start) = suggestion[..pct_pos].rfind(|c: char| !c.is_ascii_digit() && c != '.') {
                suggestion[start + 1..pct_pos].parse::<f64>().unwrap_or(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        (score, suggestion.to_string())
    }

    /// Apply configuration adjustments based on rule results
    fn apply_rule_adjustment(&self, config: &mut GpuConfig, rule: &NsightRuleResult) -> Result<()> {
        match rule.rule_name.as_str() {
            "Low Coalescing" => {
                if rule.score < 80.0 {
                    // Reduce kangaroo count to improve coalescing
                    config.max_kangaroos = (config.max_kangaroos / 2).max(512);
                    log::info!("Reduced kangaroo count to {} for better coalescing", config.max_kangaroos);
                }
            }
            "High Registers" => {
                if rule.score < 70.0 {
                    // Reduce register pressure
                    config.max_regs = 48;
                    log::info!("Limited registers to {} for better occupancy", config.max_regs);
                }
            }
            "DRAM Utilization" => {
                if rule.score > 80.0 {
                    // High DRAM usage - reduce memory pressure
                    config.max_kangaroos = (config.max_kangaroos * 3 / 4).max(512);
                    log::info!("Reduced kangaroo count to {} due to high DRAM utilization", config.max_kangaroos);
                }
            }
            _ => {
                // Unknown rule - log for analysis
                log::debug!("Unknown Nsight rule '{}': {}", rule.rule_name, rule.suggestion);
            }
        }

        Ok(())
    }

    /// Enhanced custom ECDLP rule for bias efficiency analysis
    pub fn analyze_ecdlp_bias_efficiency(&self, metrics: &HashMap<String, f64>) -> NsightRuleResult {
        let alu_pct = metrics.get("sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active")
            .copied().unwrap_or(0.0);
        let ipc = metrics.get("sm__inst_executed.avg.pct_of_peak_sustained_active")
            .copied().unwrap_or(1.0);

        let score = if alu_pct > 80.0 && ipc < 70.0 {
            60.0 // Needs optimization
        } else if alu_pct > 60.0 && ipc < 80.0 {
            75.0 // Moderate
        } else {
            90.0 // Good
        };

        let suggestion = if score < 80.0 {
            "Fuse Barrett reduction in bias_check_kernel.cu for ALU efficiency"
        } else {
            "Bias efficiency is optimal"
        };

        NsightRuleResult::new("EcdlpBiasEfficiency", score, suggestion)
    }

    /// Custom rule for analyzing DP divergence patterns
    pub fn analyze_ecdlp_divergence(&self, metrics: &HashMap<String, f64>) -> NsightRuleResult {
        let warp_eff = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
            .copied().unwrap_or(100.0);
        let branch_eff = metrics.get("sm__inst_executed.avg.pct_of_peak_sustained_elapsed")
            .copied().unwrap_or(1.0);

        let score = if warp_eff < 90.0 || branch_eff < 0.8 {
            65.0 // High divergence
        } else if warp_eff < 95.0 || branch_eff < 0.9 {
            80.0 // Moderate divergence
        } else {
            95.0 // Low divergence
        };

        let suggestion = if score < 80.0 {
            "Consider subgroupAny for DP trailing_zeros check to reduce divergence"
        } else {
            "Divergence is well-controlled"
        };

        NsightRuleResult::new("EcdlpDivergenceAnalysis", score, suggestion)
    }

    /// ML-based predictive optimization using linear regression on historical profiling data
    pub fn predict_frac(&self, history: &Vec<(f64, f64, f64, f64)>) -> f64 {
        // History format: (sm_eff, mem_pct, alu_util, past_frac)
        if history.len() < 5 {
            return 0.7; // Default if insufficient data
        }

        // Simplified linear regression for now (full ndarray matrix inversion is complex)
        // Use simple averaging with weighted recent history
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, (_, _, _, frac)) in history.iter().enumerate() {
            let weight = (i + 1) as f64; // Weight recent samples more
            weighted_sum += frac * weight;
            total_weight += weight;
        }

        let avg_frac = weighted_sum / total_weight;

        // Load current metrics for adjustment
        let current_eff = self.load_nsight_util("ci_metrics.json").unwrap_or(0.8);

        // Simple adjustment based on current efficiency
        let adjustment = if current_eff > 0.85 {
            0.05 // Increase fraction if GPU is efficient
        } else if current_eff < 0.7 {
            -0.05 // Decrease fraction if GPU is struggling
        } else {
            0.0 // Keep similar
        };

        // Clamp to reasonable bounds
        (avg_frac + adjustment).clamp(0.5, 0.9)
    }

    /// Load current Nsight utilization metrics
    fn load_nsight_util(&self, _path: &str) -> Option<f64> {
        // Simplified implementation - would parse actual metrics file
        // For now return a default value
        Some(0.8)
    }

    /// Apply ML-based predictive tuning to GPU configuration
    pub fn tune_ml_predict(&self, config: &mut GpuConfig) {
        use crate::utils::logging::load_history;

        let hist = load_history("history.json").unwrap_or_default();
        config.gpu_frac = self.predict_frac(&hist).clamp(0.5, 0.9);

        log::info!("ML prediction adjusted GPU fraction to {:.2}", config.gpu_frac);
    }

    /// Hybrid async dispatch with overlapping compute and memory operations
    pub async fn hybrid_overlap(&self, _config: &GpuConfig, _target: &BigInt256,
                               _range: (BigInt256, BigInt256), _batch_steps: u64)
                               -> Result<Option<BigInt256>, Box<dyn std::error::Error>> {
        #[cfg(feature = "rustacuda")]
        {
            use crate::gpu::backends::cuda_backend::CudaBackend;

            let cuda = CudaBackend::new()?;
            let device = cuda.device()?;

            // Create separate streams for compute and memory operations
            let compute_stream = device.create_stream(cudarc::driver::CudaStreamFlags::NON_BLOCKING)?;
            let memory_stream = device.create_stream(cudarc::driver::CudaStreamFlags::NON_BLOCKING)?;

            // Allocate states with prefetching
            let mut states = cuda.alloc_and_copy_pinned_async(&vec![RhoState::default(); 1000],
                                                             true, true).await?;

            // Prefetch states to GPU
            device.mem_prefetch_async(
                states.as_ptr() as *const std::ffi::c_void,
                states.len() * std::mem::size_of::<RhoState>(),
                0, // device ordinal
                Some(&compute_stream),
            )?;

            // Launch compute kernel on compute stream
            let compute_event = cuda.dispatch_async(
                &self.get_rho_kernel()?, // Placeholder - would need actual kernel
                &mut states,
                self.get_jump_table()?, // Placeholder
                self.get_bias_table()?, // Placeholder
                batch_steps as u32
            ).await?;

            // Copy results back on memory stream (overlaps with compute)
            let host_states = states.copy_to_vec_async(Some(&memory_stream))?;

            // Wait for compute to complete
            compute_event.synchronize()?;

            // Check for collisions on CPU
            if let Some(key) = self.check_and_resolve_collisions(&self.dp_table, &host_states).await {
                return Ok(Some(key));
            }
        }

        Ok(None)
    }

    /// Placeholder functions for kernel and data access (would need proper implementation)
    #[cfg(feature = "rustacuda")]
    fn get_rho_kernel(&self) -> Result<cudarc::driver::CudaFunction, Box<dyn std::error::Error>> {
        // Placeholder - actual implementation would load/compile the kernel
        Err("Kernel loading not implemented".into())
    }

    #[cfg(feature = "rustacuda")]
    fn get_jump_table(&self) -> Result<cudarc::driver::CudaSlice<BigInt256>, Box<dyn std::error::Error>> {
        // Placeholder
        Err("Jump table not implemented".into())
    }

    #[cfg(feature = "rustacuda")]
    fn get_bias_table(&self) -> Result<cudarc::driver::CudaSlice<f32>, Box<dyn std::error::Error>> {
        // Placeholder
        Err("Bias table not implemented".into())
    }

    #[allow(dead_code)]
    async fn check_and_resolve_collisions(&self, _dp_table: &crate::dp::DpTable, _states: &[RhoState])
                                         -> Option<BigInt256> {
        // Placeholder collision detection
        None
    }

    /// Prefetch memory for optimal kangaroo state access patterns
    #[cfg(feature = "rustacuda")]
    pub async fn prefetch_states_batch(&self, states: &CudaSlice<RhoState>,
                                      batch_start: usize, batch_size: usize)
                                      -> Result<(), Box<dyn std::error::Error>> {
        use crate::gpu::backends::cuda_backend::CudaBackend;

        let cuda = CudaBackend::new()?;
        cuda.prefetch_batch(states, batch_start, batch_size).await?;
        Ok(())
    }

    /// Unified memory prefetching for optimal access patterns
    #[cfg(feature = "rustacuda")]
    pub async fn prefetch_unified_memory(&self, ptr: *mut RhoState, size_bytes: usize,
                                        to_gpu: bool) -> Result<(), Box<dyn std::error::Error>> {
        use crate::gpu::backends::cuda_backend::CudaBackend;

        let cuda = CudaBackend::new()?;
        let device = cuda.device()?;

        let flags = if to_gpu {
            cudarc::driver::sys::cudaMemoryAdvise::cudaMemAdviseSetPreferredLocation
        } else {
            cudarc::driver::sys::cudaMemoryAdvise::cudaMemAdviseUnsetPreferredLocation
        };

        // Set memory advice for optimal access pattern
        device.mem_advise(ptr as *mut std::ffi::c_void, size_bytes, flags, 0)?;

        // Prefetch to target location
        if to_gpu {
            device.mem_prefetch_async(
                ptr as *const std::ffi::c_void,
                size_bytes,
                0, // device ordinal
                None, // default stream
            )?;
        } else {
            // Prefetch to host
            device.mem_prefetch_async(
                ptr as *const std::ffi::c_void,
                size_bytes,
                cudarc::driver::sys::cudaCpuDeviceId,
                None,
            )?;
        }

        Ok(())
    }

    fn safe_diff_mod_n(&self, tame_dist: &[u32;8], wild_dist: &[u32;8], n: &[u32;8]) -> Result<[u32;8]> {
        #[cfg(feature = "rustacuda")]
        if self.cuda_available {
            return self.cuda.safe_diff_mod_n(tame_dist, wild_dist, n);
        }
        #[cfg(feature = "wgpu")]
        if self.vulkan_available {
            return self.vulkan.safe_diff_mod_n(tame_dist, wild_dist, n);
        }
        self.cpu.safe_diff_mod_n(tame_dist, wild_dist, n)
    }

    fn barrett_reduce(&self, x: &[u32;16], modulus: &[u32;8], mu: &[u32;16]) -> Result<[u32;8]> {
        #[cfg(feature = "rustacuda")]
        if self.cuda_available {
            return self.cuda.barrett_reduce(x, modulus, mu);
        }
        #[cfg(feature = "wgpu")]
        if self.vulkan_available {
            return self.vulkan.barrett_reduce(x, modulus, mu);
        }
        self.cpu.barrett_reduce(x, modulus, mu)
    }

    fn mul_glv_opt(&self, p: &[[u32;8];3], k: &[u32;8]) -> Result<[[u32;8];3]> {
        #[cfg(feature = "rustacuda")]
        if self.cuda_available {
            return self.cuda.mul_glv_opt(p, k);
        }
        #[cfg(feature = "wgpu")]
        if self.vulkan_available {
            return self.vulkan.mul_glv_opt(p, k);
        }
        self.cpu.mul_glv_opt(p, k)
    }

    fn mod_inverse(&self, a: &[u32;8], modulus: &[u32;8]) -> Result<[u32;8]> {
        #[cfg(feature = "rustacuda")]
        if self.cuda_available {
            return self.cuda.mod_inverse(a, modulus);
        }
        #[cfg(feature = "wgpu")]
        if self.vulkan_available {
            return self.vulkan.mod_inverse(a, modulus);
        }
        self.cpu.mod_inverse(a, modulus)
    }
}
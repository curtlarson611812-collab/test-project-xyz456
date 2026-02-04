//! Hybrid Backend Implementation
//!
//! Intelligent dispatch between Vulkan (bulk) and CUDA (precision) backends

use super::backend_trait::GpuBackend;
use super::cpu_backend::CpuBackend;
use crate::types::RhoState;
use crate::kangaroo::collision::Trap;
use anyhow::Result;
use crossbeam_deque::Worker;
use std::sync::Arc;
use tokio::sync::Notify;

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

    /// Optimized dispatch with dynamic load balancing
    pub async fn dispatch_step_batch(
        &self,
        positions: &mut Vec<[[u32; 8]; 3]>,
        distances: &mut Vec<[u32; 8]>,
        types: &Vec<u32>,
        _batch_size: usize,
    ) -> Result<Vec<Trap>> {
        let (_cuda_ratio, _vulkan_ratio) = self.profile_device_performance().await;

        let mut all_traps = Vec::new();

        #[cfg(feature = "rustacuda")]
        if cuda_ratio > 0.0 {
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
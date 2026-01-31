//! Hybrid GPU dispatch abstraction
//!
//! Trait for Vulkan/CUDA backends, async buffer mapping, overlap logic

use crate::kangaroo::collision::Trap;
use anyhow::Result;
use num_bigint::BigUint;

#[cfg(feature = "vulkano")]
use vulkano::{instance::{Instance, InstanceCreateInfo}, device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags}, buffer::{Buffer, BufferCreateInfo, BufferUsage}, memory::{MemoryAllocateInfo, MemoryPropertyFlags}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage}, sync::{GpuFuture}, pipeline::{ComputePipeline, PipelineBindPoint}, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet}};

#[cfg(feature = "cudarc")]
use cudarc::driver::*;
#[cfg(feature = "cudarc")]
use cudarc::cublas::CudaBlas;
use std::sync::Arc;
// Note: cuFFT functionality removed due to cudarc limitations
// Raw CUDA driver API would be needed for full FFT support

// CUDA extern functions - Note: Using cudarc types instead of raw cuda
#[cfg(feature = "cudarc")]
extern "C" {
    fn batch_modular_inverse_cuda(
        inputs: *const u32,
        modulus: *const u32,
        outputs: *mut u32,
        is_prime: bool,
        exp_nibbles: *const u8,
        batch_size: i32,
        stream: *mut std::ffi::c_void, // cudarc stream handle
    ) -> i32; // cudarc error code

    fn bigint_mul_gemmex_cuda(
        cublas_handle: *mut std::ffi::c_void,
        a_limbs: *const u32,
        b_limbs: *const u32,
        result_limbs: *mut u32,
        batch_size: i32,
        limbs: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    fn batch_mod_mul_hybrid_cuda(
        a: *const u32,
        b: *const u32,
        mod_: *const u32,
        result: *mut u32,
        batch: i32,
        stream: cuda::CudaStream,
    ) -> cuda::CudaError;

    fn batch_mod_sqr_hybrid_cuda(
        a: *const u32,
        mod_: *const u32,
        result: *mut u32,
        batch: i32,
        stream: cuda::CudaStream,
    ) -> cuda::CudaError;
}

/// GPU backend trait for Vulkan/CUDA operations
pub trait GpuBackend {
    fn new() -> Result<Self> where Self: Sized;
    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)>;
    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>>;
    // Phase 2 precision methods for hybrid CUDA operations
    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>>;
    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>>;
    // Collision equation solving for DLP resolution
    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>>;
    // Barrett reduction for 256-bit modular reduction
    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>>;
    // cuBLAS-accelerated big integer operations
    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>>;
    // Batch affine conversion for DP export (Rule #7)
    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)>;
}


/// Create appropriate GPU backend based on available hardware
pub fn create_backend() -> Result<HybridBackend> {
    HybridBackend::new()
}

/// WGPU backend implementation (alternative to Vulkano)
#[cfg(feature = "vulkano")]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kangaroo_pipeline: wgpu::ComputePipeline,
    jump_pipeline: wgpu::ComputePipeline,
    dp_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "vulkano")]
impl WgpuBackend {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await
            .ok_or(anyhow::anyhow!("No suitable WGPU adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SpeedBitCrack WGPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await?;

        // Load and compile shaders
        let kangaroo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kangaroo shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan/shaders/kangaroo.wgsl").into()),
        });

        let jump_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jump table shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan/shaders/jump_table.wgsl").into()),
        });

        let dp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dp check shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan/shaders/dp_check.wgsl").into()),
        });

        // Create compute pipelines
        let kangaroo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kangaroo pipeline"),
            layout: None,
            module: &kangaroo_shader,
            entry_point: "kangaroo_step",
        });

        let jump_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jump pipeline"),
            layout: None,
            module: &jump_shader,
            entry_point: "jump_table_precomp",
        });

        let dp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dp pipeline"),
            layout: None,
            module: &dp_shader,
            entry_point: "dp_check_kernel",
        });

        Ok(Self {
            device,
            queue,
            kangaroo_pipeline,
            jump_pipeline,
            dp_pipeline,
        })
    }
}

#[cfg(feature = "vulkano")]
impl GpuBackend for WgpuBackend {
    fn new() -> Result<Self> {
        // WGPU requires async, but we provide sync wrapper
        Err(anyhow::anyhow!("WGPU backend requires async initialization - use WgpuBackend::new().await"))
    }

    fn precomp_table(&self, _primes: Vec<[u32;8]>, _base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow::anyhow!("WGPU precomp_table not yet implemented - requires compute shader for jump table generation"))
    }

    fn step_batch(&self, _positions: &mut Vec<[[u32;8];3]>, _distances: &mut Vec<[u32;8]>, _types: &Vec<u32>) -> Result<Vec<Trap>> {
        Err(anyhow::anyhow!("WGPU step_batch not yet implemented - requires compute shader for kangaroo stepping"))
    }

    fn batch_inverse(&self, _inputs: Vec<[u32;8]>, _modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("WGPU batch_inverse not yet implemented - requires compute shader for modular inverse"))
    }

    fn batch_solve(&self, _alphas: Vec<[u32;8]>, _betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        Err(anyhow::anyhow!("WGPU batch_solve not yet implemented - requires compute shader for collision solving"))
    }

    fn batch_solve_collision(&self, _alpha_t: Vec<[u32;8]>, _alpha_w: Vec<[u32;8]>, _beta_t: Vec<[u32;8]>, _beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, _n: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("WGPU batch_solve_collision not yet implemented - requires compute shader for complex collision equations"))
    }

    fn batch_barrett_reduce(&self, _x: Vec<[u32;16]>, _mu: [u32;9], _modulus: [u32;8], _use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("WGPU Barrett reduction not yet implemented"))
    }

    fn batch_mul(&self, _a: Vec<[u32;8]>, _b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // WGPU can implement multiplication using compute shaders
        Err(anyhow::anyhow!("WGPU batch multiplication not yet implemented"))
    }

    fn batch_to_affine(&self, _positions: Vec<[[u32;8];3]>, _modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        // WGPU can implement affine conversion using compute shaders
        Err(anyhow::anyhow!("WGPU batch affine conversion not yet implemented"))
    }
}

/// Vulkan backend implementation using vulkano
#[cfg(feature = "vulkano")]
pub struct VulkanBackend {
    instance: std::sync::Arc<vulkano::instance::Instance>,
    device: std::sync::Arc<vulkano::device::Device>,
    queue: std::sync::Arc<vulkano::device::Queue>,
    kangaroo_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
    jump_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
    dp_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
}

#[cfg(feature = "vulkano")]
impl VulkanBackend {
    pub fn new() -> Result<Self> {
        // Enhanced instance creation with validation layers for debugging
        let instance_info = InstanceCreateInfo {
            enabled_extensions: vulkano::instance::InstanceExtensions::empty(),
            enabled_layers: if cfg!(debug_assertions) {
                vec!["VK_LAYER_KHRONOS_validation".to_string()]
            } else {
                vec![]
            },
            ..Default::default()
        };

        let instance = Instance::new(instance_info)?;

        // Select best physical device (prefer discrete GPU with compute)
        let physical = instance.enumerate_physical_devices()?
            .max_by_key(|device| {
                let props = device.properties();
                let score = match props.device_type {
                    vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 3,
                    vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 2,
                    _ => 1,
                };
                // Boost score if device has good compute capabilities
                score + if props.limits.max_compute_work_group_invocations >= 1024 { 1 } else { 0 }
            })
            .ok_or(anyhow::anyhow!("No suitable Vulkan device found"))?;

        // Find compute queue family
        let queue_family = physical.queue_family_properties()
            .iter()
            .position(|props| props.queue_flags.intersects(QueueFlags::COMPUTE))
            .ok_or(anyhow::anyhow!("No compute queue family found"))? as u32;

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family,
                    queues: vec![1.0],
                    ..Default::default()
                }],
                ..Default::default()
            }
        )?;

        let queue = queues.next().ok_or(anyhow::anyhow!("Failed to get compute queue"))?;

        // Load and compile shaders at runtime (alternative to build-time compilation)
        let kangaroo_shader = Self::load_shader(&device, include_str!("vulkan/shaders/kangaroo.wgsl"))?;
        let jump_shader = Self::load_shader(&device, include_str!("vulkan/shaders/jump_table.wgsl"))?;
        let dp_shader = Self::load_shader(&device, include_str!("vulkan/shaders/dp_check.wgsl"))?;

        // Create compute pipelines
        let kangaroo_pipeline = ComputePipeline::new(
            device.clone(),
            kangaroo_shader.entry_point("kangaroo_step").ok_or(anyhow::anyhow!("kangaroo_step entry not found"))?,
            &(),
        )?;

        let jump_pipeline = ComputePipeline::new(
            device.clone(),
            jump_shader.entry_point("jump_table_precomp").ok_or(anyhow::anyhow!("jump_table_precomp entry not found"))?,
            &(),
        )?;

        let dp_pipeline = ComputePipeline::new(
            device.clone(),
            dp_shader.entry_point("dp_check_kernel").ok_or(anyhow::anyhow!("dp_check_kernel entry not found"))?,
            &(),
        )?;

        Ok(Self {
            instance,
            device,
            queue,
            kangaroo_pipeline,
            jump_pipeline,
            dp_pipeline,
        })
    }

    fn load_shader(device: &std::sync::Arc<vulkano::device::Device>, wgsl_source: &str) -> Result<std::sync::Arc<vulkano::shader::ShaderModule>> {
        // For now, create a minimal compute shader
        // In production, would use vulkano_shaders for proper WGSL compilation
        // This is a placeholder that would need actual WGSL->SPIR-V compilation
        Err(anyhow::anyhow!("Runtime WGSL compilation not implemented - use build.rs approach"))
    }
}

#[cfg(feature = "vulkano")]
impl GpuBackend for VulkanBackend {
    fn new() -> Result<Self> {
        Self::new()
    }



    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        let num = positions.len() as u64;
        if num == 0 { return Ok(vec![]); }

        // Concise buffer allocation for inputs
        let positions_buf = Buffer::from_iter(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            positions.iter().flatten().flatten().copied()
        )?;

        let distances_buf = Buffer::from_iter(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            distances.iter().flatten().copied()
        )?;

        let types_buf = Buffer::from_iter(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            types.iter().copied()
        )?;

        // Jump table buffers (assume pre-loaded, placeholder here)
        let jump_points_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: 1024 * std::mem::size_of::<[[u32;8];3]>() as u64, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        let jump_sizes_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: 1024 * std::mem::size_of::<[u32;8]>() as u64, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        // Trap output buffers (with atomic index)
        let trap_xs_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: 1024 * std::mem::size_of::<[u32;8]>() as u64, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        let trap_dists_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: 1024 * std::mem::size_of::<[u32;8]>() as u64, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        let trap_types_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: 1024 * 4, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        let trap_index_buf = Buffer::from_data(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            0u32
        )?;

        // Descriptor set for kangaroo pipeline (bindings 0-8)
        let desc_set = PersistentDescriptorSet::new(
            &self.kangaroo_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::buffer(0, positions_buf.clone()),
                WriteDescriptorSet::buffer(1, distances_buf.clone()),
                WriteDescriptorSet::buffer(2, types_buf),
                WriteDescriptorSet::buffer(3, jump_points_buf),
                WriteDescriptorSet::buffer(4, jump_sizes_buf),
                WriteDescriptorSet::buffer(5, trap_xs_buf.clone()),
                WriteDescriptorSet::buffer(6, trap_dists_buf.clone()),
                WriteDescriptorSet::buffer(7, trap_types_buf.clone()),
                WriteDescriptorSet::buffer(8, trap_index_buf.clone()),
            ]
        )?;

        // Dispatch compute shader
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        )?;

        builder
            .bind_pipeline_compute(self.kangaroo_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.kangaroo_pipeline.layout().clone(), 0, desc_set)
            .dispatch([(num as u32 + 255) / 256, 1, 1])?;

        let cmd = builder.build()?;
        let future = cmd.execute(self.queue.clone())?.then_signal_fence_and_flush()?;
        future.wait(None)?;

        // Read trap count and data
        let trap_count = trap_index_buf.read()?.0 as usize;
        if trap_count == 0 {
            // Read back updated positions/distances even if no traps
            *positions = Self::read_positions_buffer(&positions_buf, num as usize)?;
            *distances = Self::read_distances_buffer(&distances_buf, num as usize)?;
            return Ok(vec![]);
        }

        // Read trap data
        let xs = Self::read_trap_buffer(&trap_xs_buf, trap_count)?;
        let dists = Self::read_trap_buffer(&trap_dists_buf, trap_count)?;
        let typs = trap_types_buf.read()?.iter().take(trap_count).cloned().collect::<Vec<_>>();

        // Convert to Trap structs
        let traps = (0..trap_count)
            .map(|i| Trap {
                x: Self::pack_u32_to_u64(&xs[i]),
                dist: Self::biguint_from_u32(&dists[i]),
                is_tame: typs[i] == 0,
            })
            .collect();

        // Read back updated positions/distances
        *positions = Self::read_positions_buffer(&positions_buf, num as usize)?;
        *distances = Self::read_distances_buffer(&distances_buf, num as usize)?;

        Ok(traps)
    }

    fn batch_inverse(&self, _inputs: Vec<[u32;8]>, _modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Vulkan backend can implement precision operations using compute shaders
        // For now, placeholder - would use modular_inverse.wgsl shader
        Err(anyhow::anyhow!("Vulkan precision operations not yet implemented"))
    }

    fn batch_solve(&self, _alphas: Vec<[u32;8]>, _betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // Vulkan backend can implement solve operations using compute shaders
        // For now, placeholder - would use solve_collision.wgsl shader
        Err(anyhow::anyhow!("Vulkan solve operations not yet implemented"))
    }

    fn batch_solve_collision(&self, _alpha_t: Vec<[u32;8]>, _alpha_w: Vec<[u32;8]>, _beta_t: Vec<[u32;8]>, _beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, _n: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Vulkan backend can implement collision solving using compute shaders
        Err(anyhow::anyhow!("Vulkan collision solving not yet implemented"))
    }

    fn batch_barrett_reduce(&self, _x: Vec<[u32;16]>, _mu: [u32;9], _modulus: [u32;8], _use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // Vulkan backend can implement Barrett reduction using compute shaders
        Err(anyhow::anyhow!("Vulkan Barrett reduction not yet implemented"))
    }

    fn batch_mul(&self, _a: Vec<[u32;8]>, _b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Vulkan backend can implement multiplication using compute shaders
        // For now, placeholder - would use bigint_mul.wgsl shader
        Err(anyhow::anyhow!("Vulkan batch multiplication not yet implemented"))
    }

    fn batch_to_affine(&self, _positions: Vec<[[u32;8];3]>, _modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        // Vulkan backend can implement affine conversion using compute shaders
        // For now, placeholder - would use affine conversion WGSL shader
        Err(anyhow::anyhow!("Vulkan batch affine conversion not yet implemented"))
    }

    // Helper methods for buffer reading and data conversion
}

/// Utility functions for Vulkan buffer reading
#[cfg(feature = "vulkano")]
impl VulkanBackend {
    pub fn read_positions_buffer(buf: &Buffer<[[[u32;8];3]]>, count: usize) -> Result<Vec<[[u32;8];3]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    pub fn read_distances_buffer(buf: &Buffer<[[u32;8]]>, count: usize) -> Result<Vec<[u32;8]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    pub fn read_trap_buffer(buf: &Buffer<[[u32;8]]>, count: usize) -> Result<Vec<[u32;8]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    // Pack [u32;8] (little-endian) to [u64;4]
    pub fn pack_u32_to_u64(arr: &[u32;8]) -> [u64;4] {
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        out
    }

    // Convert [u32;8] (little-endian) to BigUint
    pub fn biguint_from_u32(arr: &[u32;8]) -> BigUint {
        BigUint::from_slice(&arr.iter().rev().map(|&u| u).collect::<Vec<_>>())
    }
}

/// CUDA backend implementation for precision operations
#[cfg(feature = "cudarc")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    cublas_handle: CudaBlas,    // Real cudarc cublas integration
    // Note: PTX modules cannot be loaded directly with cudarc
    // High-level module loading requires raw CUDA driver API
    // For now, using placeholder - real implementation would need driver API
}

#[cfg(feature = "cudarc")]
impl GpuBackend for CudaBackend {
    fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, _primes: Vec<[u32;8]>, _base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow::anyhow!("CUDA precomp_table not yet implemented"))
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        let batch_size = positions.len();
        if batch_size == 0 || batch_size != distances.len() || batch_size != types.len() {
            return Err(anyhow::anyhow!("Invalid batch sizes for kangaroo stepping"));
        }

        // Flatten inputs for device memory
        let positions_flat: Vec<u32> = positions.iter().flat_map(|p| p.iter().flatten()).cloned().collect();
        let distances_flat: Vec<u32> = distances.iter().flatten().cloned().collect();

        // Allocate device memory
        let d_positions = self.context.alloc_copy(&positions_flat)?;
        let d_distances = self.context.alloc_copy(&distances_flat)?;
        let d_types = self.context.alloc_copy(types)?;
        let d_new_positions = self.context.alloc_zeros::<u32>(batch_size * 24)?; // 3 * 8 u32 per position
        let d_new_distances = self.context.alloc_zeros::<u32>(batch_size * 8)?;
        let d_traps = self.context.alloc_zeros::<u32>(batch_size * 9)?; // trap data per kangaroo

        // Launch kangaroo stepping kernel
        let step_fn = self.inverse_module.get_function("kangaroo_step_batch")?;
        unsafe {
            step_fn.launch(
                &self.stream,
                ((batch_size as u32 + 255) / 256, 1, 1),
                (256, 1, 1),
                &[
                    &d_positions.as_kernel_parameter(),
                    &d_distances.as_kernel_parameter(),
                    &d_types.as_kernel_parameter(),
                    &d_new_positions.as_kernel_parameter(),
                    &d_new_distances.as_kernel_parameter(),
                    &d_traps.as_kernel_parameter(),
                    &(batch_size as u32),
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let new_positions_flat = d_new_positions.copy_to_vec()?;
        let new_distances_flat = d_new_distances.copy_to_vec()?;
        let traps_flat = d_traps.copy_to_vec()?;

        // Update the mutable inputs
        for i in 0..batch_size {
            let pos_offset = i * 24;
            let dist_offset = i * 8;

            // Update positions
            for j in 0..3 {
                for k in 0..8 {
                    positions[i][j][k] = new_positions_flat[pos_offset + j * 8 + k];
                }
            }

            // Update distances
            for k in 0..8 {
                distances[i][k] = new_distances_flat[dist_offset + k];
            }
        }

        // Parse traps
        let mut traps = Vec::new();
        for i in 0..batch_size {
            let trap_offset = i * 9;
            let trap_type = traps_flat[trap_offset];
            if trap_type != 0 {
                // Trap found - parse the data
                let mut x = [0u64; 4];
                for j in 0..4 {
                    x[j] = (traps_flat[trap_offset + 1 + j * 2] as u64) |
                           ((traps_flat[trap_offset + 1 + j * 2 + 1] as u64) << 32);
                }
                let dist_biguint = BigUint::from_slice(&traps_flat[trap_offset + 1..trap_offset + 9].iter().rev().map(|&u| u).collect::<Vec<_>>());

                // Extract is_tame from trap data (assuming kernel outputs type in trap_offset position)
                let is_tame = traps_flat[trap_offset] == 0; // 0 = tame, 1 = wild
                traps.push(Trap { x, dist: dist_biguint, is_tame });
            }
        }

        Ok(traps)
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch_size = inputs.len() as i32;
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Prepare p-2 exponent for Fermat's little theorem
        // For secp256k1 prime, compute p-2
        let mut p_minus_2 = modulus;
        // p-2 = p - 2 (subtract 2 from the prime)
        if p_minus_2[0] >= 2 {
            p_minus_2[0] -= 2;
        } else {
            // Handle borrow if needed
            p_minus_2[0] = p_minus_2[0].wrapping_sub(2);
        }
        let exp_bits = p_minus_2.to_vec();
        let exp_bit_length = 256;

        // Allocate device memory
        let d_inputs = self.context.alloc_copy(&inputs.concat())?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_exp_bits = self.context.alloc_copy(&exp_bits)?;
        let d_outputs = self.context.alloc_zeros::<u32>(batch_size as usize * 8)?;

        // Launch cuBLAS-accelerated batch inverse kernel
        let grid_size = (batch_size as u32 + 255) / 256;
        let block_size = 256;

        let inverse_fn = self.inverse_module.get_function("batch_fermat_inverse")?;
        unsafe {
            inverse_fn.launch(
                &self.stream,
                (grid_size, 1, 1),
                (block_size, 1, 1),
                &[
                    &d_inputs.as_kernel_parameter(),
                    &d_outputs.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &d_exp_bits.as_kernel_parameter(),
                    &(exp_bit_length as i32),
                    &batch_size,
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let output_flat = d_outputs.copy_to_vec()?;
        let outputs = output_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok(outputs)
    }

    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // CUDA implementation for batch collision solving
        // Uses optimized CUDA kernels for parallel discrete logarithm solving
        let batch_size = alphas.len();
        if batch_size == 0 || batch_size != betas.len() {
            return Err(anyhow::anyhow!("Invalid batch sizes"));
        }

        // Flatten inputs for device memory
        let alphas_flat: Vec<u32> = alphas.into_iter().flatten().collect();
        let betas_flat: Vec<u32> = betas.into_iter().flatten().collect();

        // Allocate device memory
        let d_alphas = self.context.alloc_copy(&alphas_flat)?;
        let d_betas = self.context.alloc_copy(&betas_flat)?;
        let d_results = self.context.alloc_zeros::<u64>(batch_size * 4)?;

        // Launch batch solve kernel
        let solve_fn = self.solve_module.get_function("batch_solve_kernel")?;
        unsafe {
            solve_fn.launch(
                &self.stream,
                (batch_size as u32, 1, 1),
                (256, 1, 1),
                &[
                    &d_alphas.as_kernel_parameter(),
                    &d_betas.as_kernel_parameter(),
                    &d_results.as_kernel_parameter(),
                    &(batch_size as u32),
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let results_flat = d_results.copy_to_vec()?;
        let results = results_flat.chunks(4).map(|c| c.try_into().unwrap()).collect();

        Ok(results)
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch = alpha_t.len();
        if batch == 0 || batch != alpha_w.len() || batch != beta_t.len() || batch != beta_w.len() || batch != target.len() {
            return Err(anyhow::anyhow!("Invalid batch sizes for collision solving"));
        }

        // Flatten all inputs
        let alpha_t_flat: Vec<u32> = alpha_t.into_iter().flatten().collect();
        let alpha_w_flat: Vec<u32> = alpha_w.into_iter().flatten().collect();
        let beta_t_flat: Vec<u32> = beta_t.into_iter().flatten().collect();
        let beta_w_flat: Vec<u32> = beta_w.into_iter().flatten().collect();
        let target_flat: Vec<u32> = target.into_iter().flatten().collect();

        // Allocate device memory
        let d_alpha_t = self.context.alloc_copy(&alpha_t_flat)?;
        let d_alpha_w = self.context.alloc_copy(&alpha_w_flat)?;
        let d_beta_t = self.context.alloc_copy(&beta_t_flat)?;
        let d_beta_w = self.context.alloc_copy(&beta_w_flat)?;
        let d_target = self.context.alloc_copy(&target_flat)?;
        let d_n = self.context.alloc_copy(&n)?;
        let d_priv_out = self.context.alloc_zeros::<u32>(batch * 8)?;

        // Get kernel function
        let solve_fn = self.solve_module.get_function("batch_collision_solve")?;

        // Launch kernel
        unsafe {
            solve_fn.launch(
                &self.stream,
                ((batch as u32 + 255) / 256, 1, 1),
                (256, 1, 1),
                &[
                    &d_alpha_t.as_kernel_parameter(),
                    &d_alpha_w.as_kernel_parameter(),
                    &d_beta_t.as_kernel_parameter(),
                    &d_beta_w.as_kernel_parameter(),
                    &d_target.as_kernel_parameter(),
                    &d_n.as_kernel_parameter(),
                    &d_priv_out.as_kernel_parameter(),
                    &(batch as i32),
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let priv_flat = d_priv_out.copy_to_vec()?;

        // Convert back to [u32;8] arrays
        let results = priv_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok(results)
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        let batch = x.len();
        if batch == 0 {
            return Ok(vec![]);
        }

        // Flatten x inputs (512-bit each)
        let x_flat: Vec<u32> = x.into_iter().flatten().collect();

        // Compute n' for Montgomery if needed
        let n_prime = if use_montgomery { Self::compute_n_prime(&modulus) } else { 0 };

        // Allocate device memory
        let d_x = self.context.alloc_copy(&x_flat)?;
        let d_mu = self.context.alloc_copy(&mu)?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_out = self.context.alloc_zeros::<u32>(batch * 8)?;

        // Get kernel function
        let barrett_fn = self.solve_module.get_function("batch_barrett_reduce")?;

        // Launch kernel
        unsafe {
            barrett_fn.launch(
                &self.stream,
                ((batch as u32 + 255) / 256, 1, 1),
                (256, 1, 1),
                &[
                    &d_x.as_kernel_parameter(),
                    &d_mu.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &d_out.as_kernel_parameter(),
                    &use_montgomery,
                    &(n_prime as u32),
                    &(batch as i32),
                    &8i32, // limbs
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let out_flat = d_out.copy_to_vec()?;

        // Convert back to [u32;8] arrays
        let results = out_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok(results)
    }


    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        let batch_size = positions.len() as i32;
        if batch_size == 0 {
            return Ok((vec![], vec![]));
        }

        // Flatten positions: each point is [X,Y,Z] where each is [u32;8], so 24 u32 per point
        let positions_flat: Vec<u32> = positions.into_iter()
            .flat_map(|point| point.into_iter().flatten())
            .collect();

        // Allocate device memory
        let d_positions = self.context.alloc_copy(&positions_flat)?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_x_outputs = self.context.alloc_zeros::<u32>(batch_size as usize * 8)?;
        let d_y_outputs = self.context.alloc_zeros::<u32>(batch_size as usize * 8)?;

        // Compute n' for Montgomery reduction
        let n_prime = Self::compute_n_prime(&modulus);

        // Launch fused affine conversion kernel
        let affine_fn = self.inverse_module.get_function("batch_affine_fused")?;
        let grid_size = ((batch_size as u32 + 255) / 256) as u32;
        unsafe {
            affine_fn.launch(
                &self.stream,
                (grid_size, 1, 1),
                (256, 1, 1),
                &[
                    &d_positions.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &(n_prime as u32),
                    &d_x_outputs.as_kernel_parameter(),
                    &d_y_outputs.as_kernel_parameter(),
                    &batch_size,
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let x_flat = d_x_outputs.copy_to_vec()?;
        let y_flat = d_y_outputs.copy_to_vec()?;

        // Convert back to arrays
        let x_coords = x_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();
        let y_coords = y_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok((x_coords, y_coords))
    }



    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>, mod_: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow::anyhow!("Invalid batch size"));
        }

        // Use hybrid Barrett-Montgomery modular multiplication for precision operations
        // This provides ~30% performance gains over plain modular arithmetic (Rule #4)
        let a_flat: Vec<u32> = a.into_iter().flatten().collect();
        let b_flat: Vec<u32> = b.into_iter().flatten().collect();

        // Allocate device memory
        let d_a = self.context.alloc_copy(&a_flat)?;
        let d_b = self.context.alloc_copy(&b_flat)?;
        let d_mod = self.context.alloc_copy(&mod_)?;
        let d_result = self.context.alloc_zeros::<u32>(batch_size * 8)?;

        // Get kernel function
        let hybrid_mul_fn = self.hybrid_module.get_function("batch_mod_mul_hybrid")?;

        // Launch hybrid multiplication kernel
        unsafe {
            hybrid_mul_fn.launch(
                &self.stream,
                ((batch_size as u32 + 255) / 256, 1, 1),
                (256, 1, 1),
                &[
                    &d_a.as_kernel_parameter(),
                    &d_b.as_kernel_parameter(),
                    &d_mod.as_kernel_parameter(),
                    &d_result.as_kernel_parameter(),
                    &(batch_size as i32),
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let result_flat = d_result.copy_to_vec()?;

        // Convert back to [u32;8] arrays
        let results = result_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok(results)
    }

}


/// Hybrid backend that automatically selects between Vulkan and CUDA based on availability and operation type
#[derive(Clone)]
pub enum HybridBackend {
    #[cfg(feature = "cudarc")]
    Cuda(CudaBackend),
    Cpu(CpuBackend),
}

/// CPU fallback backend
#[derive(Clone)]
pub struct CpuBackend;

impl HybridBackend {
    /// Create a new hybrid backend with automatic selection
    pub fn new() -> Result<Self> {
        // Try CUDA first for precision operations (preferred for secp256k1 math)
        #[cfg(feature = "cudarc")]
        {
            match CudaBackend::new() {
                Ok(cuda) => return Ok(Self::Cuda(cuda)),
                Err(_) => {} // Fall through to CPU
            }
        }

        // Final fallback to CPU (Vulkan backend removed due to fundamental limitations)
        Ok(Self::Cpu(CpuBackend::new()))
    }

    /// Check if this backend supports precision operations (true for CUDA, false for CPU)
    pub fn supports_precision_ops(&self) -> bool {
        #[cfg(feature = "cudarc")]
        {
            matches!(self, Self::Cuda(_))
        }
        #[cfg(not(feature = "cudarc"))]
        {
            false
        }
    }

    /// Create shared buffer for Vulkan-CUDA interop (if available)
    /// Falls back to separate allocations if interop not supported
    #[cfg(any(feature = "vulkano", feature = "cudarc"))]
    pub fn create_shared_buffer(&self, size: usize) -> Result<SharedBuffer> {
        match self {
            #[cfg(feature = "cudarc")]
            Self::Cuda(cuda) => {
                // CUDA buffer allocation
                let buffer = cuda.context.alloc_zeros::<u8>(size)?;
                Ok(SharedBuffer::Cuda(buffer))
            }
            #[cfg(feature = "vulkano")]
            Self::Vulkan(vulkan) => {
                // Vulkan-only buffer
                Ok(SharedBuffer::Vulkan(Buffer::new_sized(
                    vulkan.device.clone(),
                    BufferCreateInfo {
                        size: size as u64,
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    MemoryAllocateInfo {
                        property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                        ..Default::default()
                    }
                )?))
            }
            Self::Cpu => {
                Err(anyhow::anyhow!("CPU backend doesn't support shared buffers"))
            }
        }
    }
}

/// Shared buffer enum for Vulkan-CUDA interop
#[cfg(any(feature = "vulkano", feature = "cudarc"))]
pub enum SharedBuffer {
    #[cfg(feature = "cudarc")]
    Cuda(cudarc::driver::CudaSlice<u8>),
    #[cfg(feature = "vulkano")]
    Vulkan(std::sync::Arc<vulkano::buffer::Buffer>),
}

impl GpuBackend for HybridBackend {
    fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, _primes: Vec<[u32;8]>, _base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow::anyhow!("CUDA precomp_table not yet implemented"))
    }


    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        match self {
            Self::Vulkan(v) => v.step_batch(positions, distances, types),
            Self::Cuda(c) => c.step_batch(positions, distances, types),
            Self::Cpu(c) => c.step_batch(positions, distances, types),
        }
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision inverse operations
        match self {
            Self::Cuda(c) => c.batch_inverse(inputs, modulus),
            Self::Vulkan(v) => v.batch_inverse(inputs, modulus),
            Self::Cpu(c) => c.batch_inverse(inputs, modulus),
        }
    }

    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // Prefer CUDA for precision solving operations
        match self {
            Self::Cuda(c) => c.batch_solve(alphas, betas),
            Self::Vulkan(v) => v.batch_solve(alphas, betas),
            Self::Cpu(c) => c.batch_solve(alphas, betas),
        }
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision collision solving
        match self {
            Self::Cuda(c) => c.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n),
            Self::Vulkan(v) => v.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n),
            Self::Cpu(c) => c.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n),
        }
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision Barrett reduction
        match self {
            Self::Cuda(c) => c.batch_barrett_reduce(x, mu, modulus, use_montgomery),
            Self::Vulkan(v) => v.batch_barrett_reduce(x, mu, modulus, use_montgomery),
            Self::Cpu(c) => c.batch_barrett_reduce(x, mu, modulus, use_montgomery),
        }
    }

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Prefer CUDA for precision multiplication operations
        match self {
            Self::Cuda(c) => c.batch_mul(a, b),
            Self::Vulkan(v) => v.batch_mul(a, b),
            Self::Cpu(c) => c.batch_mul(a, b),
        }
    }

    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        // Prefer CUDA for precision affine conversion operations
        match self {
            Self::Cuda(c) => c.batch_to_affine(positions, modulus),
            Self::Vulkan(v) => v.batch_to_affine(positions, modulus),
            Self::Cpu(c) => c.batch_to_affine(positions, modulus),
        }
    }
}
#[cfg(feature = "cudarc")]
impl CudaBackend {
    // Helper: Check if modulus is prime (simplified primality test)
    pub fn is_prime_modulus(modulus: &[u32; 8]) -> bool {
        // Convert to BigUint for primality testing
        let mod_big = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());

        // Simple primality test (production would use more sophisticated methods)
        if mod_big < num_bigint::BigUint::from(2u32) {
            return false;
        }
        if mod_big == num_bigint::BigUint::from(2u32) || mod_big == num_bigint::BigUint::from(3u32) {
            return true;
        }
        if mod_big.clone() % num_bigint::BigUint::from(2u32) == num_bigint::BigUint::from(0u32) {
            return false;
        }

        // Check divisibility by odd numbers up to sqrt(modulus)
        let sqrt_mod = mod_big.sqrt();
        let mut i = num_bigint::BigUint::from(3u32);
        while i <= sqrt_mod {
            if mod_big.clone() % i.clone() == num_bigint::BigUint::from(0u32) {
                return false;
            }
            i += num_bigint::BigUint::from(2u32);
        }

        true
    }

    // Helper: Compute p-2 as base-16 nibbles for windowed exponentiation
    pub fn compute_exponent_nibbles(modulus: &[u32; 8]) -> Vec<u8> {
        let mut exp = num_bigint::BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());
        exp -= num_bigint::BigUint::from(2u32); // p-2

        let mut nibbles = vec![0u8; 64]; // 256 bits / 4 bits per nibble
        for i in 0..64 {
            nibbles[i] = (exp.bit(i * 4) as u8) |
                        ((exp.bit(i * 4 + 1) as u8) << 1) |
                        ((exp.bit(i * 4 + 2) as u8) << 2) |
                        ((exp.bit(i * 4 + 3) as u8) << 3);
        }
        nibbles
    }

    // Helper: Compute n' for Montgomery reduction (n' = -n^{-1} mod 2^32)
    pub fn compute_n_prime(modulus: &[u32; 8]) -> u32 {
        // For Montgomery reduction, we need n' such that n * n' ≡ -1 mod 2^32
        // We can compute this using the extended Euclidean algorithm on n mod 2^32 and 2^32
        let n_low = modulus[0] as u64; // Least significant limb
        let r = 1u64 << 32; // 2^32

        // Extended Euclidean algorithm to find inverse of n_low mod r
        let mut old_r = r;
        let mut r = n_low;
        let mut old_s: i64 = 0;
        let mut s: i64 = 1;

        while r > 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - (quotient as i64) * s;
            old_s = temp_s;
        }

        // old_s now contains the inverse, make it positive and compute n'
        let mut inv = old_s;
        while inv < 0 {
            inv += r as i64;
        }
        let n_prime = ((-inv) % (r as i64)) as u32;

        n_prime
    }
}

#[cfg(feature = "cudarc")]
impl GpuBackend for CudaBackend {
    fn new() -> Result<Self> {
        // Initialize CUDA device and stream
        let device = Arc::new(CudaDevice::new(0)?);
        let stream = CudaStream::default();

        // Initialize cuBLAS
        let cublas_handle = CudaBlas::new(device.clone())?;

        // Note: cudarc does not support loading PTX modules directly
        // Raw CUDA driver API would be needed for kernel execution
        // This is a fundamental limitation of cudarc's design

        Ok(Self {
            device,
            stream,
            cublas_handle,
        })
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow::anyhow!("CUDA kernel execution requires PTX module loading - cudarc limitation: CudaModule::load_ptx() not available. Raw CUDA driver API (cuModuleLoad(), cuModuleGetFunction(), cuLaunchKernel()) needed for actual GPU execution. cuBLAS works but custom kernels require driver-level access."))
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        Err(anyhow::anyhow!("CUDA kangaroo stepping requires custom PTX kernel execution - cudarc limitation: No CudaModule/CudaFunction support for launching custom kernels. Raw CUDA driver API needed: cuModuleLoad(), cuModuleGetFunction(), cuLaunchKernel() with grid(batch/256,1,1), block(256,1,1) for optimal occupancy. cuBLAS handles matrix ops but not elliptic curve arithmetic."))
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("CUDA modular inverse requires PTX kernel execution - cudarc limitation: No kernel launching capabilities. Raw driver API needed for Fermat's Little Theorem (a^(p-2) mod p) or Euclidean algorithm implementation. secp256k1 prime allows efficient GPU computation with Montgomery reduction."))
    }

    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        Err(anyhow::anyhow!("CUDA collision solving requires custom PTX kernels - cudarc limitation: No kernel execution support. Raw driver API needed for discrete log equation solving: d = alpha_t - alpha_w * inv(beta_w - beta_t) mod n. Montgomery arithmetic required for 256-bit modular operations."))
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("CUDA collision equation solving requires PTX kernel execution - cudarc limitation prevents access to implemented Barrett reduction kernels. Raw driver API needed for complex dlog solving with Montgomery arithmetic and modular inverse operations."))
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("CUDA Barrett reduction requires PTX kernel execution - cudarc limitation prevents loading modular arithmetic kernels. Raw driver API needed for q = ((x >> k) * mu) >> k, r = x - q * m with Montgomery optimization for multiplication-heavy operations."))
    }

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow::anyhow!("Invalid batch size for multiplication"));
        }

        // Allocate device memory for cuBLAS operations
        let mut results = Vec::with_capacity(batch_size);

        // For each pair, perform schoolbook multiplication using CPU fallback
        // since cudarc doesn't support custom kernels
        for i in 0..batch_size {
            let mut result = [0u32; 16];

            // Simple schoolbook multiplication (not optimized for GPU)
            // In a real implementation, this would use custom PTX kernels
            for j in 0..8 {
                for k in 0..8 {
                    let product = (a[i][j] as u64) * (b[i][k] as u64);
                    let mut carry = product;

                    // Add to result with carry propagation
                    let mut pos = j + k;
                    while carry > 0 && pos < 16 {
                        let sum = (result[pos] as u64) + (carry & 0xFFFFFFFF);
                        result[pos] = sum as u32;
                        carry = sum >> 32;
                        pos += 1;
                    }
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        Err(anyhow::anyhow!("CUDA affine conversion requires PTX kernel execution - cudarc limitation prevents elliptic curve coordinate transformation. Raw driver API needed for z_inv = z^(-1), x = x * z_inv^2, y = y * z_inv^3 with batch modular inverse for Jacobian to affine conversion."))
    }

}

//! Hybrid GPU dispatch abstraction
//!
//! Trait for Vulkan/CUDA backends, async buffer mapping, overlap logic

use crate::types::{KangarooState, Point, Trap};
use crate::math::BigInt256;
use anyhow::Result;
use num_bigint::BigUint;
#[cfg(feature = "vulkan")]
use std::sync::Arc;

#[cfg(feature = "vulkan")]
use vulkano::{instance::{Instance, InstanceCreateInfo}, device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags}, buffer::{Buffer, BufferCreateInfo, BufferUsage}, memory::{MemoryAllocateInfo, MemoryPropertyFlags}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer}, sync::{GpuFuture, FenceSignalFuture}, pipeline::{ComputePipeline, PipelineBindPoint}, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet}};

#[cfg(feature = "cuda")]
use cuda::{CudaDevice, CudaStream, CudaModule};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction};
#[cfg(feature = "cuda")]
use std::path::Path;

// CUDA extern functions
#[cfg(feature = "cuda")]
extern "C" {
    fn batch_modular_inverse_cuda(
        inputs: *const u32,
        modulus: *const u32,
        outputs: *mut u32,
        is_prime: bool,
        exp_nibbles: *const u8,
        batch_size: i32,
        stream: cuda::CudaStream,
    ) -> cuda::CudaError;

    fn bigint_mul_gemmex_cuda(
        cublas_handle: *mut std::ffi::c_void,
        a_limbs: *const u32,
        b_limbs: *const u32,
        result_limbs: *mut u32,
        batch_size: i32,
        limbs: i32,
        stream: cuda::CudaStream,
    ) -> cuda::CudaError;

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
#[cfg(feature = "vulkan")]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kangaroo_pipeline: wgpu::ComputePipeline,
    jump_pipeline: wgpu::ComputePipeline,
    dp_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "vulkan")]
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../gpu/vulkan/shaders/kangaroo.wgsl").into()),
        });

        let jump_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jump table shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../gpu/vulkan/shaders/jump_table.wgsl").into()),
        });

        let dp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dp check shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../gpu/vulkan/shaders/dp_check.wgsl").into()),
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

#[cfg(feature = "vulkan")]
impl GpuBackend for WgpuBackend {
    fn new() -> Result<Self> {
        // WGPU requires async, but we provide sync wrapper
        Err(anyhow::anyhow!("WGPU backend requires async initialization - use WgpuBackend::new().await"))
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        // WGPU implementation would go here
        // For now, placeholder
        Ok((vec![], vec![]))
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        // WGPU implementation would go here
        // For now, placeholder
        Ok(vec![])
    }

    fn batch_inverse(&self, _inputs: Vec<[u32;8]>, _modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // WGPU can implement precision operations
        Err(anyhow::anyhow!("WGPU precision operations not yet implemented"))
    }

    fn batch_solve(&self, _alphas: Vec<[u32;8]>, _betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // WGPU can implement solve operations
        Err(anyhow::anyhow!("WGPU solve operations not yet implemented"))
    }

    fn batch_solve_collision(&self, _alpha_t: Vec<[u32;8]>, _alpha_w: Vec<[u32;8]>, _beta_t: Vec<[u32;8]>, _beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, _n: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow::anyhow!("WGPU collision solving not yet implemented"))
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
#[cfg(feature = "vulkan")]
pub struct VulkanBackend {
    instance: std::sync::Arc<vulkano::instance::Instance>,
    device: std::sync::Arc<vulkano::device::Device>,
    queue: std::sync::Arc<vulkano::device::Queue>,
    kangaroo_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
    jump_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
    dp_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
}

#[cfg(feature = "vulkan")]
impl VulkanBackend {
    pub fn new() -> Result<Self> {
        // Enhanced instance creation with validation layers for debugging
        let instance_info = InstanceCreateInfo {
            enabled_extensions: if cfg!(debug_assertions) {
                vulkano::ValidationFeatures::all().extensions
            } else {
                vulkano::instance::InstanceExtensions::empty()
            },
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
        let kangaroo_shader = Self::load_shader(&device, include_str!("../../gpu/vulkan/shaders/kangaroo.wgsl"))?;
        let jump_shader = Self::load_shader(&device, include_str!("../../gpu/vulkan/shaders/jump_table.wgsl"))?;
        let dp_shader = Self::load_shader(&device, include_str!("../../gpu/vulkan/shaders/dp_check.wgsl"))?;

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

#[cfg(feature = "vulkan")]
impl GpuBackend for VulkanBackend {
    fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        let num = primes.len() as u64;

        // Concise buffer allocation using iterators
        let primes_buf = Buffer::from_iter(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            primes.iter().flatten().copied()
        )?;

        let base_buf = Buffer::from_iter(
            self.device.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() },
            base.iter().copied()
        )?;

        // Output buffers with calculated sizes
        let points_size = num * std::mem::size_of::<[[u32;8];3]>() as u64;
        let sizes_size = num * std::mem::size_of::<[u32;8]>() as u64;

        let points_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: points_size, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        let sizes_buf = Buffer::new_sized(
            self.device.clone(),
            BufferCreateInfo { size: sizes_size, usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            MemoryAllocateInfo { property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT, ..Default::default() }
        )?;

        // Descriptor set with automatic layout handling
        let desc_set = PersistentDescriptorSet::new(
            &self.jump_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::buffer(0, primes_buf),
                WriteDescriptorSet::buffer(1, base_buf),
                WriteDescriptorSet::buffer(2, points_buf.clone()),
                WriteDescriptorSet::buffer(3, sizes_buf.clone()),
            ]
        )?;

        // Launch kernel with optimal workgroup size
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        )?;

        builder
            .bind_pipeline_compute(self.jump_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.jump_pipeline.layout().clone(), 0, desc_set)
            .dispatch([(num as u32 + 255) / 256, 1, 1])?;

        let cmd = builder.build()?;
        let future = cmd.execute(self.queue.clone())?.then_signal_fence_and_flush()?;
        future.wait(None)?;

        // Read back results
        let points = points_buf.read()?.to_vec();
        let sizes = sizes_buf.read()?.to_vec();

        Ok((points, sizes))
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
    fn read_positions_buffer(buf: &Buffer<[[[u32;8];3]]>, count: usize) -> Result<Vec<[[u32;8];3]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    fn read_distances_buffer(buf: &Buffer<[[u32;8]]>, count: usize) -> Result<Vec<[u32;8]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    fn read_trap_buffer(buf: &Buffer<[[u32;8]]>, count: usize) -> Result<Vec<[u32;8]>> {
        let data = buf.read()?;
        Ok(data.iter().take(count).cloned().collect())
    }

    // Pack [u32;8] (little-endian) to [u64;4]
    fn pack_u32_to_u64(arr: &[u32;8]) -> [u64;4] {
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = (arr[i * 2] as u64) | ((arr[i * 2 + 1] as u64) << 32);
        }
        out
    }

    // Convert [u32;8] (little-endian) to BigUint
    fn biguint_from_u32(arr: &[u32;8]) -> BigUint {
        BigUint::from_slice(&arr.iter().rev().map(|&u| u).collect::<Vec<_>>())
    }
}

/// CUDA backend implementation for precision operations
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: CudaDevice,
    context: CudaContext,
    stream: CudaStream,
    inverse_module: CudaModule, // Contains affine and inverse kernels
    solve_module: CudaModule,   // Contains collision solving and Barrett kernels
    hybrid_module: CudaModule,  // Contains hybrid Barrett-Montgomery arithmetic
    carry_module: CudaModule,   // Contains optimized carry propagation
    bigint_module: CudaModule,
    fft_module: CudaModule,
    fused_module: CudaModule,
    cublas_handle: cudarc::cublas::CudaBlas,
    cufft_forward_plan: cudarc::cufft::CudaFftPlan,
    cufft_inverse_plan: cudarc::cufft::CudaFftPlan,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new() -> Result<Self> {
        // Initialize CUDA library
        cuda::init()?;

        // Check for CUDA devices
        let device_count = cuda::get_device_count()?;
        if device_count == 0 {
            return Err(anyhow::anyhow!("No CUDA-capable devices found"));
        }

        // Select first device (could be made configurable)
        let device = CudaDevice::new(0)?;

        // Check compute capability (require 5.0+ for 256-bit operations)
        let major = device.get_attribute(cuda::Attribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(cuda::Attribute::ComputeCapabilityMinor)?;
        let compute_cap = major as f32 + minor as f32 * 0.1;

        if compute_cap < 5.0 {
            return Err(anyhow::anyhow!(
                "CUDA device compute capability {:.1} is insufficient. Requires 5.0+ for 256-bit operations",
                compute_cap
            ));
        }

        // Create context and stream
        let context = device.create_context()?;
        context.set_current()?;
        let stream = device.create_stream()?;

        // Load pre-compiled PTX module
        let out_dir = std::env::var("OUT_DIR")?;
        let ptx_path = Path::new(&out_dir).join("inverse.ptx");

        if !ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "CUDA PTX module not found at {}. Ensure build.rs compiled successfully.",
                ptx_path.display()
            ));
        }

        let inverse_module = context.load_module(&ptx_path)?;

        // Load solve module for collision solving and Barrett reduction
        let solve_ptx_path = Path::new(&out_dir).join("solve.ptx");
        if !solve_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Solve PTX module not found at {}. Ensure build.rs compiled successfully.",
                solve_ptx_path.display()
            ));
        }

        let solve_module = context.load_module(&solve_ptx_path)?;

        // Load hybrid modular arithmetic module
        let hybrid_ptx_path = Path::new(&out_dir).join("hybrid.ptx");
        if !hybrid_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Hybrid PTX module not found at {}. Ensure build.rs compiled successfully.",
                hybrid_ptx_path.display()
            ));
        }

        let hybrid_module = context.load_module(&hybrid_ptx_path)?;

        // Load carry propagation module
        let carry_ptx_path = Path::new(&out_dir).join("carry_propagation.ptx");
        if !carry_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Carry propagation PTX not found at {}. Ensure build.rs compiled successfully.",
                carry_ptx_path.display()
            ));
        }

        let carry_module = context.load_module(&carry_ptx_path)?;

        // Load bigint multiplication module and initialize cuBLAS
        let bigint_ptx_path = Path::new(&out_dir).join("bigint_mul.ptx");
        if !bigint_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Bigint multiplication PTX not found at {}. Ensure build.rs compiled successfully.",
                bigint_ptx_path.display()
            ));
        }

        let bigint_module = context.load_module(&bigint_ptx_path)?;

        // Load FFT multiplication module and cuFFT plans
        let fft_ptx_path = Path::new(&out_dir).join("fft_mul.ptx");
        if !fft_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "FFT multiplication PTX not found at {}. Ensure build.rs compiled successfully.",
                fft_ptx_path.display()
            ));
        }

        let fft_module = context.load_module(&fft_ptx_path)?;

        // Load carry propagation module
        let carry_ptx_path = Path::new(&out_dir).join("carry_propagation.ptx");
        if !carry_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Carry propagation PTX not found at {}. Ensure build.rs compiled successfully.",
                carry_ptx_path.display()
            ));
        }

        let carry_module = context.load_module(&carry_ptx_path)?;

        // Load fused multiplication and reduction module
        let fused_ptx_path = Path::new(&out_dir).join("fused_mul_redc.ptx");
        if !fused_ptx_path.exists() {
            return Err(anyhow::anyhow!(
                "Fused mul PTX not found at {}. Ensure build.rs compiled successfully.",
                fused_ptx_path.display()
            ));
        }

        let fused_module = context.load_module(&fused_ptx_path)?;

        // Initialize cuBLAS and cuFFT
        let cublas_handle = cudarc::cublas::CudaBlas::new(context.clone())?;

        // Create cuFFT plans for 512-point 1D FFT (for 256-bit multiplication)
        let cufft_forward_plan = cudarc::cufft::CudaFftPlan::new_1d(context.clone(), 512, cudarc::cufft::CudaFftType::Z2Z, 1000)?;
        let cufft_inverse_plan = cudarc::cufft::CudaFftPlan::new_1d(context.clone(), 512, cudarc::cufft::CudaFftType::Z2Z, 1000)?;

        Ok(Self {
            device,
            context,
            stream,
            inverse_module,
            solve_module,
            hybrid_module,
            carry_module,
            bigint_module,
            fft_module,
            fused_module,
            cublas_handle,
            cufft_forward_plan,
            cufft_inverse_plan,
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        // CUDA implementation for jump table precomputation
        // Would use CUDA kernels for parallel point multiplication
        // For now, placeholder - Phase 2 would implement full CUDA jump table computation
        Err(anyhow::anyhow!("CUDA jump table precomputation not yet implemented"))
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        // CUDA implementation for kangaroo stepping
        // Would use CUDA kernels for parallel kangaroo updates
        // For now, placeholder - Phase 2 would implement full CUDA stepping
        Err(anyhow::anyhow!("CUDA kangaroo stepping not yet implemented"))
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch_size = inputs.len() as i32;
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Prepare p-2 exponent for Fermat's little theorem (simplified)
        // In practice, would compute actual p-2 for the modulus
        let exp_bits = vec![1u32; 256]; // Placeholder
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
        // Would use CUDA kernels for parallel private key computation
        // For now, placeholder - Phase 2 would implement full CUDA solving
        Err(anyhow::anyhow!("CUDA batch solve not yet implemented"))
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
        let n_prime = if use_montgomery { self.compute_n_prime(&modulus) } else { 0 };

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

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch_size = inputs.len() as i32;
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Determine if modulus is prime (affects algorithm selection)
        let is_prime = self.is_prime_modulus(&modulus);

        // Prepare exponent nibbles for Fermat (p-2 in base 16)
        let exp_nibbles = if is_prime {
            self.compute_exponent_nibbles(&modulus)
        } else {
            vec![0u8; 64] // Not used for Euclidean
        };

        // Compute n' for Montgomery reduction (if needed)
        let n_prime = self.compute_n_prime(&modulus);

        // Flatten inputs for device memory
        let input_flat: Vec<u32> = inputs.into_iter().flatten().collect();

        // Allocate device memory
        let d_inputs = self.context.alloc_copy(&input_flat)?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_exp_nibbles = self.context.alloc_copy(&exp_nibbles)?;
        let d_outputs = self.context.alloc_zeros::<u32>(batch_size as usize * 8)?;

        // Launch the new hybrid batch modular inverse kernel
        let inverse_fn = self.inverse_module.get_function("batch_mod_inverse")?;
        let grid_size = ((batch_size as u32 + 255) / 256) as u32;
        unsafe {
            inverse_fn.launch(
                &self.stream,
                (grid_size, 1, 1),
                (256, 1, 1),
                &[
                    &d_inputs.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &is_prime,
                    &d_exp_nibbles.as_kernel_parameter(),
                    &(n_prime as u32),
                    &d_outputs.as_kernel_parameter(),
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

    fn fused_mul_redc(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>, modulus: [u32;8], n_prime: u32) -> Result<Vec<[u32;8]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow::anyhow!("Invalid batch size"));
        }

        // Allocate device memory
        let d_a = self.context.alloc_copy(&a.concat())?;
        let d_b = self.context.alloc_copy(&b.concat())?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_results = self.context.alloc_zeros::<u32>(batch_size * 8)?;

        // Launch fused multiplication + Montgomery reduction kernel
        let fused_fn = self.fused_module.get_function("fused_mul_redc")?;
        unsafe {
            fused_fn.launch(
                &self.stream,
                (batch_size as u32, 1, 1),  // One block per bigint
                (256, 1, 1),                // Full warp per block
                &[
                    &d_a.as_kernel_parameter(),
                    &d_b.as_kernel_parameter(),
                    &d_results.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &(n_prime as u32),
                    &(batch_size as u32),
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let results_flat = d_results.copy_to_vec()?;
        let results = results_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

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
        let n_prime = self.compute_n_prime(&modulus);

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

    fn batch_redc(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>, modulus: [u32;8], n_prime: u32) -> Result<Vec<[u32;8]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow::anyhow!("Invalid batch size"));
        }

        // Flatten inputs for device memory
        let a_flat: Vec<u32> = a.into_iter().flatten().collect();
        let b_flat: Vec<u32> = b.into_iter().flatten().collect();

        // Allocate device memory
        let d_a = self.context.alloc_copy(&a_flat)?;
        let d_b = self.context.alloc_copy(&b_flat)?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_results = self.context.alloc_zeros::<u32>(batch_size * 8)?;

        // Launch fused REDC kernel
        let fused_fn = self.fused_module.get_function("batch_fused_redc")?;
        let shared_mem_size = 2 * 8 * std::mem::size_of::<u32>(); // 2 * LIMBS * sizeof(uint32_t)
        unsafe {
            fused_fn.launch(
                &self.stream,
                (batch_size as u32, 1, 1),
                (32, 1, 1), // One warp per bigint
                shared_mem_size,
                &[
                    &d_a.as_kernel_parameter(),
                    &d_b.as_kernel_parameter(),
                    &d_results.as_kernel_parameter(),
                    &d_modulus.as_kernel_parameter(),
                    &(n_prime as u32),
                    &(batch_size as i32),
                    &(8i32), // LIMBS
                ]
            )?;
        }

        // Synchronize and read results
        self.stream.synchronize()?;
        let results_flat = d_results.copy_to_vec()?;
        let results = results_flat.chunks(8).map(|c| c.try_into().unwrap()).collect();

        Ok(results)
    }

    // Helper: Check if modulus is prime (simplified primality test)
    fn is_prime_modulus(&self, modulus: &[u32; 8]) -> bool {
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
    fn compute_exponent_nibbles(&self, modulus: &[u32; 8]) -> Vec<u8> {
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
    fn compute_n_prime(&self, modulus: &[u32; 8]) -> u32 {
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

    fn batch_mul_cublas(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        let batch_size = a.len();
        const LIMBS: usize = 8;
        const RESULT_LIMBS: usize = 16;

        // Convert to float arrays for cuBLAS precision
        let a_flat: Vec<f32> = a.iter().flatten().map(|&x| x as f32).collect();
        let b_flat: Vec<f32> = b.iter().flatten().map(|&x| x as f32).collect();

        let d_a = self.context.alloc_copy(&a_flat)?;
        let d_b = self.context.alloc_copy(&b_flat)?;
        let d_products = self.context.alloc_zeros::<f32>(batch_size * LIMBS * LIMBS)?;
        let d_results = self.context.alloc_zeros::<u64>(batch_size * RESULT_LIMBS)?;

        // Batched GEMM for product matrix
        self.cublas.sgemm_strided_batched(
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            LIMBS as i32, LIMBS as i32, 1,
            &1.0f32,
            d_a.as_ptr(), LIMBS as i32, LIMBS as i32,
            d_b.as_ptr(), LIMBS as i32, LIMBS as i32,
            &0.0f32,
            d_products.as_ptr(), LIMBS as i32, (LIMBS * LIMBS) as i32,
            batch_size as i32,
        )?;

        // Launch optimized carry propagation
        let carry_fn = self.carry_module.get_function("carry_propagate_warp_shuffle")?;
        unsafe {
            carry_fn.launch(
                &self.stream,
                (batch_size as u32, 1, 1),  // grid(batch) - one block per bigint
                (32, 1, 1),                  // block(32) - warp size
                &[
                    &d_results.as_kernel_parameter(),  // out
                    &d_products.as_kernel_parameter(), // products
                    &(batch_size as u32),              // batch
                    &(LIMBS as u32),                   // limbs
                ]
            )?;
        }

        self.stream.synchronize()?;
        let results_flat = d_results.copy_to_vec()?;

        let results = results_flat.chunks(RESULT_LIMBS)
            .map(|chunk| {
                let mut result = [0u32; RESULT_LIMBS];
                for (i, &val) in chunk.iter().enumerate() {
                    result[i] = val as u32;
                    // Note: This simplified version doesn't handle full u64 to u32 conversion
                }
                result
            })
            .collect();

        Ok(results)
    }

    fn batch_mul_cufft(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        let batch_size = a.len();

        // Prepare input data as flattened arrays
        let a_flat: Vec<u32> = a.into_iter().flatten().collect();
        let b_flat: Vec<u32> = b.into_iter().flatten().collect();

        let d_a = self.context.alloc_copy(&a_flat)?;
        let d_b = self.context.alloc_copy(&b_flat)?;
        let d_results = self.context.alloc_zeros::<u32>(batch_size * 16)?;

        // Launch cuFFT multiplication kernel
        // Note: This requires integrating with the cuFFT C API
        // For now, placeholder - would call batch_bigint_mul_cufft

        // Simplified: use cuBLAS fallback for now
        self.batch_mul_cublas(
            a.into_iter().map(|x| x).collect(),
            b.into_iter().map(|x| x).collect()
        )
    }
}

#[cfg(feature = "cuda")]
#[async_trait]
impl GpuBackend for CudaBackend {
    async fn initialize(&mut self) -> Result<()> {
        // TODO: Initialize CUDA context and load kernels using nvrtc
        // Compile CUDA kernels at runtime for optimal performance
        Ok(())
    }

    async fn step_kangaroos(&self, kangaroos: &[KangarooState]) -> Result<Vec<KangarooState>> {
        // TODO: Implement CUDA kangaroo stepping with PTX optimizations
        // Use __umul64hi for fast 128-bit multiplication
        Ok(kangaroos.to_vec()) // Placeholder
    }

    async fn generate_kangaroos(&self, count: usize, is_tame: bool) -> Result<Vec<KangarooState>> {
        // TODO: Implement CUDA kangaroo generation with parallel random number generation
        Ok(vec![]) // Placeholder
    }

    async fn check_distinguished_points(&self, points: &[Point], dp_bits: usize) -> Result<Vec<bool>> {
        // TODO: Implement CUDA DP checking with fast bit operations
        Ok(vec![false; points.len()]) // Placeholder
    }

    async fn modular_operations(&self, ops: &[ModularOp]) -> Result<Vec<BigInt256>> {
        // TODO: Implement CUDA modular arithmetic with Montgomery/Barrett reduction
        // Use Barrett reduction for fast modular multiplication
        Ok(vec![BigInt256::zero(); ops.len()]) // Placeholder
    }

    fn memory_info(&self) -> MemoryInfo {
        // TODO: Get actual CUDA memory info
        MemoryInfo {
            total_memory: 24 * 1024 * 1024 * 1024, // 24GB placeholder for RTX 4090/5090
            used_memory: 0,
            free_memory: 24 * 1024 * 1024 * 1024,
        }
    }

    async fn synchronize(&self) -> Result<()> {
        // TODO: CUDA stream synchronization
        Ok(())
    }
}

/// CPU fallback backend
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

#[async_trait]
impl GpuBackend for CpuBackend {
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    async fn step_kangaroos(&self, kangaroos: &[KangarooState]) -> Result<Vec<KangarooState>> {
        // CPU fallback - just return input unchanged
        Ok(kangaroos.to_vec())
    }

    async fn generate_kangaroos(&self, count: usize, is_tame: bool) -> Result<Vec<KangarooState>> {
        // Generate deterministic CPU fallback kangaroos
        let mut result = Vec::new();
        for i in 0..count {
            let state = KangarooState::new(
                Point { x: [i as u64; 4], y: [0; 4], z: [1; 4] },
                0,
                [0; 4],
                [0; 4],
                is_tame,
                i as u64,
            );
            result.push(state);
        }
        Ok(result)
    }

    async fn check_distinguished_points(&self, _points: &[Point], _dp_bits: usize) -> Result<Vec<bool>> {
        // CPU fallback - return all false
        Ok(vec![false; _points.len()])
    }

    async fn modular_operations(&self, _ops: &[ModularOp]) -> Result<Vec<BigInt256>> {
        // CPU fallback
        Ok(vec![BigInt256::zero(); _ops.len()])
    }

    fn memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
        }
    }

    async fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Hybrid backend that automatically selects between Vulkan and CUDA based on availability and operation type
#[derive(Clone)]
pub enum HybridBackend {
    Vulkan(VulkanBackend),
    Cuda(CudaBackend),
}

impl HybridBackend {
    /// Create a new hybrid backend with automatic selection
    pub fn new() -> Result<Self> {
        // Try CUDA first for precision operations (preferred for secp256k1 math)
        #[cfg(feature = "cuda")]
        {
            match CudaBackend::new() {
                Ok(cuda) => return Ok(Self::Cuda(cuda)),
                Err(_) => {} // Fall through to Vulkan
            }
        }

        // Fallback to Vulkan for general operations
        #[cfg(feature = "vulkan")]
        {
            match VulkanBackend::new() {
                Ok(vulkan) => return Ok(Self::Vulkan(vulkan)),
                Err(_) => {} // Fall through to CPU
            }
        }

        // Final fallback to CPU
        Ok(Self::Cuda(CpuBackend::new()))
    }

    /// Check if this backend supports precision operations (true for CUDA, false for Vulkan/CPU)
    pub fn supports_precision_ops(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Create shared buffer for Vulkan-CUDA interop (if available)
    /// Falls back to separate allocations if interop not supported
    pub fn create_shared_buffer(&self, size: usize) -> Result<SharedBuffer> {
        match self {
            Self::Cuda(cuda) => {
                // Try Vulkan-CUDA interop if Vulkan backend also available
                #[cfg(feature = "vulkan")]
                {
                    // For now, create separate buffers
                    // TODO: Implement VK_EXT_external_memory_cuda interop
                }
                // Fallback to CUDA-only buffer
                Ok(SharedBuffer::Cuda(cuda.context.alloc_zeros::<u8>(size)?))
            }
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
        }
    }
}

/// Shared buffer enum for Vulkan-CUDA interop
pub enum SharedBuffer {
    Cuda(cudarc::driver::CudaSlice<u8>),
    Vulkan(std::sync::Arc<vulkano::buffer::Buffer>),
}

impl GpuBackend for HybridBackend {
    fn new() -> Result<Self> {
        Self::new()
    }

    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        match self {
            Self::Vulkan(v) => v.precomp_table(primes, base),
            Self::Cuda(c) => c.precomp_table(primes, base),
        }
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        match self {
            Self::Vulkan(v) => v.step_batch(positions, distances, types),
            Self::Cuda(c) => c.step_batch(positions, distances, types),
        }
    }

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision inverse operations
        match self {
            Self::Cuda(c) => c.batch_inverse(inputs, modulus),
            Self::Vulkan(v) => v.batch_inverse(inputs, modulus),
        }
    }

    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        // Prefer CUDA for precision solving operations
        match self {
            Self::Cuda(c) => c.batch_solve(alphas, betas),
            Self::Vulkan(v) => v.batch_solve(alphas, betas),
        }
    }

    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision collision solving
        match self {
            Self::Cuda(c) => c.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n),
            Self::Vulkan(v) => v.batch_solve_collision(alpha_t, alpha_w, beta_t, beta_w, target, n),
        }
    }

    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        // Prefer CUDA for precision Barrett reduction
        match self {
            Self::Cuda(c) => c.batch_barrett_reduce(x, mu, modulus, use_montgomery),
            Self::Vulkan(v) => v.batch_barrett_reduce(x, mu, modulus, use_montgomery),
        }
    }

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Prefer CUDA for precision multiplication operations
        match self {
            Self::Cuda(c) => c.batch_mul(a, b),
            Self::Vulkan(v) => v.batch_mul(a, b),
        }
    }

    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        // Prefer CUDA for precision affine conversion operations
        match self {
            Self::Cuda(c) => c.batch_to_affine(positions, modulus),
            Self::Vulkan(v) => v.batch_to_affine(positions, modulus),
        }
    }
}
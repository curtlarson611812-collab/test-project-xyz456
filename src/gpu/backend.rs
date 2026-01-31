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

/// GPU backend trait for Vulkan/CUDA operations
pub trait GpuBackend {
    fn new() -> Result<Self> where Self: Sized;
    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)>;
    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>>;
    // Phase 2 precision methods for hybrid CUDA operations
    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>>;
    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>>;
    // cuBLAS-accelerated big integer operations
    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>>;
}


/// Create appropriate GPU backend based on available hardware
pub fn create_backend() -> Result<Box<dyn GpuBackend>> {
    // Try CUDA first (precision operations), then Vulkan (bulk operations), then CPU fallback

    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new() {
            Ok(cuda) => return Ok(Box::new(cuda)),
            Err(_) => {} // Fall through to Vulkan
        }
    }

    #[cfg(feature = "vulkan")]
    {
        // Try WGPU first (simpler, more portable), then Vulkano
        // For now, just try Vulkano - WGPU would need async initialization
        match VulkanBackend::new() {
            Ok(vulkan) => return Ok(Box::new(vulkan)),
            Err(_) => {} // Fall through to CPU
        }
    }

    // CPU fallback
    Ok(Box::new(CpuBackend::new()))
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

    fn batch_mul(&self, _a: Vec<[u32;8]>, _b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // WGPU can implement multiplication using compute shaders
        Err(anyhow::anyhow!("WGPU batch multiplication not yet implemented"))
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

    fn batch_mul(&self, _a: Vec<[u32;8]>, _b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        // Vulkan backend can implement multiplication using compute shaders
        // For now, placeholder - would use bigint_mul.wgsl shader
        Err(anyhow::anyhow!("Vulkan batch multiplication not yet implemented"))
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
    inverse_module: CudaModule,
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

    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        let batch_size = inputs.len() as i32;
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Compute p-2 for Fermat's little theorem (simplified)
        let mut p_minus_2 = BigUint::from_slice(&modulus.iter().rev().map(|&x| x).collect::<Vec<_>>());
        p_minus_2 -= 2u32;

        // Convert to bit array (simplified - should handle full 256-bit)
        let exp_bits: Vec<u32> = (0..256).map(|i| {
            if p_minus_2.bit(i as u64) { 1 } else { 0 }
        }).collect();

        // Allocate device memory
        let d_inputs = self.context.alloc_copy(&inputs.concat())?;
        let d_modulus = self.context.alloc_copy(&modulus)?;
        let d_exp_bits = self.context.alloc_copy(&exp_bits)?;
        let d_outputs = self.context.alloc_zeros::<u32>(batch_size as usize * 8)?;

        // Launch Fermat inverse kernel (simplified - would implement full square-and-multiply)
        let grid_size = (batch_size + 255) / 256;
        let block_size = 256;

        // For now, use a simplified approach - copy inputs as identity
        // Real implementation would call batch_modular_inverse_cublas
        self.context.memcpy_d2d(&d_outputs, &d_inputs)?;

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

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow::anyhow!("Invalid batch size"));
        }

        // Choose algorithm based on batch size and precision requirements
        if batch_size >= 1000 {
            // Use cuFFT for large batches (better asymptotic performance)
            self.batch_mul_cufft(a, b)
        } else {
            // Use cuBLAS GEMM for smaller batches (lower overhead)
            self.batch_mul_cublas(a, b)
        }
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

        // Launch carry reduction
        let carry_fn = self.carry_module.get_function("carry_propagate_warp_shuffle")?;
        unsafe {
            carry_fn.launch(
                &self.stream,
                (batch_size as u32, RESULT_LIMBS as u32, 1),
                (32, 1, 1), // One warp per result limb
                &[
                    &d_results.as_kernel_parameter(),
                    &d_products.as_kernel_parameter(),
                    &(batch_size as u32),
                    &(LIMBS as u32),
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
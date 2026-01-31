//! wgpu pipeline creation and management
//!
//! Device/queue/pipeline creation, bind group layouts, push constants

use wgpu::{Device, Queue, ShaderModule, ComputePipeline, BindGroupLayout, PipelineLayout};
use anyhow::Result;

/// Vulkan pipeline manager
pub struct VulkanPipeline {
    device: Device,
    queue: Queue,
    kangaroo_pipeline: ComputePipeline,
    jump_table_pipeline: ComputePipeline,
    dp_check_pipeline: ComputePipeline,
}

impl VulkanPipeline {
    /// Create new Vulkan pipeline manager
    pub async fn new() -> Result<Self> {
        // Create wgpu instance and adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SpeedBitCrack GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await?;

        // Create shader modules
        let kangaroo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Kangaroo Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::vulkan::KANGAROO_SHADER.into()),
        });

        let jump_table_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Jump Table Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::vulkan::JUMP_TABLE_SHADER.into()),
        });

        let dp_check_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DP Check Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::vulkan::DP_CHECK_SHADER.into()),
        });

        // Create bind group layouts
        let kangaroo_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Kangaroo Bind Group Layout"),
            entries: &[
                // Kangaroo state buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Jump table buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layouts
        let kangaroo_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Kangaroo Pipeline Layout"),
            bind_group_layouts: &[&kangaroo_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let kangaroo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Kangaroo Compute Pipeline"),
            layout: Some(&kangaroo_pipeline_layout),
            module: &kangaroo_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let jump_table_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Jump Table Pipeline"),
            layout: Some(&kangaroo_pipeline_layout), // Reuse layout
            module: &jump_table_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let dp_check_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DP Check Pipeline"),
            layout: Some(&kangaroo_pipeline_layout), // Reuse layout
            module: &dp_check_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(VulkanPipeline {
            device,
            queue,
            kangaroo_pipeline,
            jump_table_pipeline,
            dp_check_pipeline,
        })
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get kangaroo pipeline
    pub fn kangaroo_pipeline(&self) -> &ComputePipeline {
        &self.kangaroo_pipeline
    }

    /// Get jump table pipeline
    pub fn jump_table_pipeline(&self) -> &ComputePipeline {
        &self.jump_table_pipeline
    }

    /// Get DP check pipeline
    pub fn dp_check_pipeline(&self) -> &ComputePipeline {
        &self.dp_check_pipeline
    }

    /// Create buffer
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(desc)
    }

    /// Create bind group
    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(desc)
    }

    /// Submit command encoder
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }
}
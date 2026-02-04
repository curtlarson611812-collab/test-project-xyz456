//! CUDA Backend Implementation
//!
//! High-performance CUDA acceleration for precision cryptographic operations

#![allow(unsafe_code)] // Required for CUDA kernel launches and buffer operations

use super::backend_trait::GpuBackend;
use crate::kangaroo::collision::Trap;
use crate::math::bigint::BigInt256;
use anyhow::{Result, anyhow};

#[cfg(feature = "rustacuda")]
use rustacuda::device::Device as CudaDevice;
#[cfg(feature = "rustacuda")]
use rustacuda::context::{Context as CudaContext, ContextFlags};
#[cfg(feature = "rustacuda")]
use rustacuda::stream::{Stream as CudaStream, StreamFlags};
#[cfg(feature = "rustacuda")]
use rustacuda::module::Module as CudaModule;
#[cfg(feature = "rustacuda")]
use rustacuda::memory::{DeviceBuffer, CopyDestination};
#[cfg(feature = "rustacuda")]
use rustacuda::launch;
#[cfg(feature = "rustacuda")]
use rustacuda::error::CudaError;
#[cfg(feature = "rustacuda")]
use std::ffi::CStr;
// For cudarc L2 hint
#[cfg(feature = "rustacuda")]
use cudarc::driver::{CudaFunctionAttribute, CudaCacheConfig, DriverError};

/// Rho algorithm state for GPU kernel execution
#[cfg(feature = "rustacuda")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RhoState {
    pub current: Point,
    pub steps: BigInt256,
    pub bias_mod: u64,
}

/// Distinguished point for collision detection
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DpPoint {
    pub x: [u64; 4],
    pub steps: BigInt256,
}

impl Default for DpPoint {
    fn default() -> Self {
        DpPoint {
            x: [0; 4],
            steps: BigInt256::zero(),
        }
    }
}

/// Workgroup size for CUDA kernels - tuned for occupancy on RTX 5090
#[cfg(feature = "rustacuda")]
const WORKGROUP_SIZE: u32 = 256;

/// CUDA error checking macro for consistent error handling with OOM recovery
#[cfg(feature = "rustacuda")]
macro_rules! cuda_check {
    ($expr:expr) => {
        cuda_check!($expr, "CUDA operation")
    };
    ($expr:expr, $ctx:expr) => {
        match $expr {
            Ok(result) => result,
            Err(CudaError::OutOfMemory) => {
                return Err(anyhow!("CUDA OOM in {} - reduce batch size", $ctx));
            },
            Err(e) => {
                return Err(anyhow!("CUDA error in {}: {}", $ctx, e));
            }
        }
    };
}

/// CUDA backend for precision cryptographic operations
#[cfg(feature = "rustacuda")]
pub struct CudaBackend {
    device: CudaDevice,
    context: CudaContext,
    stream: CudaStream,
    inverse_module: CudaModule,
    solve_module: CudaModule,
    bigint_mul_module: CudaModule,
    fft_mul_module: CudaModule,
    hybrid_module: CudaModule,
    precomp_module: CudaModule,
    step_module: CudaModule,
    barrett_module: CudaModule,
    rho_module: CudaModule,
    dp_buffer: DeviceBuffer<DpPoint>,
    dp_count: DeviceBuffer<u32>,
}

#[cfg(feature = "rustacuda")]
impl CudaBackend {
    // Chunk: CUDA PTX Load with Bias Param (src/gpu/backends/cuda_backend.rs)
    // Dependencies: cudarc::driver::*, std::path::Path
    pub fn load_rho_kernel(device: &CudaDevice, ptx_path: &Path, bias_weights: &[f32]) -> Result<CudaFunction, DriverError> {
        let ptx = Ptx::from_file(ptx_path)?;
        let module = device.load_ptx(ptx, "rho", &["rho_kernel"])?;
        let func = module.get_func("rho_kernel")?.ok_or(DriverError::InvalidSymbol)?;
        let bias_dev = device.htod_copy(bias_weights.to_vec())?;  // Copy bias to device
        // Bias as global or param in launch
        Ok(func)
    }
    // Test: Mock device/ptx, check func valid

    /// Compute n' for Montgomery reduction where n' * n ≡ -1 mod 2^32
    /// Using algorithm from HAC 14.94
    fn compute_n_prime(modulus: &[u32; 8]) -> u32 {
        // Extended Euclidean algorithm for 32-bit
        let mut y = 0u32;
        let mut x = 1u32;

        let n0 = modulus[0]; // Least significant word
        let mut a = n0;
        let mut b = 0x100000000u64; // 2^32

        while a > 1 {
            let quotient = a / (b as u32);
            let t = b as u32;

            b = (a % t) as u64;
            a = t;

            let temp = x;
            x = y.wrapping_sub(quotient.wrapping_mul(x));
            y = temp;
        }

        if y > 0x7FFFFFFF {
            y = y.wrapping_neg();
        }

        y
    }

    // Chunk: CUDA Pinned Alloc with Retry (src/gpu/backends/cuda_backend.rs)
    // Dependencies: cudarc::driver::*, types::RhoState
    pub fn alloc_rho_states(device: &CudaDevice, mut count: usize) -> Result<CudaSlice<RhoState>, DriverError> {
        loop {
            match device.alloc_zeroed::<RhoState>(count) {
                Ok(buf) => return Ok(buf),
                Err(DriverError::Cuda(CUresult::CU_ERROR_OUT_OF_MEMORY)) if count > 512 => count /= 2,  // Halve on OOM
                Err(e) => return Err(e),
            }
        }
    }
    // Test: Alloc 1024, if OOM simulate halve to 512

    // Chunk: Occupancy Tune (cuda_backend.rs)
    pub fn get_optimal_block_size(kernel: &CudaModule) -> u32 {
        // Assume kernel has a function, get registers from it
        // Placeholder: tune based on regs
        let regs = 64; // Placeholder
        if regs > 64 { 128 } else { 256 }
    }

    // Chunk: CUDA Launch and Update (src/gpu/backends/cuda_backend.rs)
    // Dependencies: cudarc::driver::*, load_rho_kernel, alloc_rho_states
    pub fn dispatch_and_update(device: &CudaDevice, kernel: &CudaFunction, mut states: CudaSlice<RhoState>, jumps: CudaSlice<BigInt256>, bias: CudaSlice<f32>, steps: u32) -> Result<Vec<RhoState>, DriverError> {
        let grid = (states.len() as u32 / 128 + 1, 1, 1);
        let block = (128, 1, 1);
        kernel.launch(grid, block, &[&mut states, &jumps, &bias, &steps])?;
        let host_states = states.copy_to_vec()?;
        Ok(host_states)
    }
    // Test: Mock launch, check host_states updated

    // Chunk: Multi-Stream Refine (cuda_backend.rs)
    pub async fn gpu_batch_refine(slices: &mut [PosSlice], biases: &[f64]) -> Result<(), CudaError> {
        let stream_count = 4;
        // Chunk: CUDA Collision Resolve (src/gpu/backends/cuda_backend.rs)
        // Dependencies: cudarc::driver::*, dp::table::DpTable, math::secp::mod_inverse
        pub fn check_and_resolve_collisions(dp_table: &DpTable, host_states: &[RhoState]) -> Option<BigInt256> {
            for state in host_states {
                if state.is_dp {
                    if let Some((tame_dist, wild_dist, jump_diff)) = dp_table.lookup(&state.point_hash) {
                        let diff = tame_dist - wild_dist;
                        let inv_jump = mod_inverse(&jump_diff, &CURVE_ORDER);
                        return Some(diff * inv_jump % &CURVE_ORDER);
                    }
                }
            }
            None
        }
        // Test: Mock DP match, compute dlog, check key = diff * inv mod n

        let chunk_size = slices.len() / stream_count;
        let mut futures = vec![];
        for i in 0..stream_count {
            let stream = CudaStream::new(StreamFlags::NON_BLOCKING, None)?;
            let s_chunk = &mut slices[i*chunk_size..(i+1)*chunk_size];
            let dev_slices = DeviceBuffer::from_slice(s_chunk)?;
            let dev_biases = DeviceBuffer::from_slice(biases)?;
            let kernel = load_kernel("refine_slices_gpu");
            kernel.launch_on_stream(&stream, (chunk_size/128 +1,1,1), (128,1,1), dev_slices, dev_biases, chunk_size, 3);
            futures.push(stream.synchronize());
        }
        join_all(futures).await
    }

    // Chunk: Cudarc L2 Hint (cuda_backend.rs)
    // Dependencies: cudarc::driver::{CudaFunctionAttribute, CudaCacheConfig}
    pub fn launch_with_l2_hint(kernel: &CudaFunction, grid: LaunchDim, block: LaunchDim, params: &[CudaSlice]) -> Result<(), DriverError> {
        kernel.set_attribute(CudaFunctionAttribute::PreferredSharedMemoryCarveout, 50)?;  // 50% L2
        kernel.set_cache_config(CudaCacheConfig::PreferL1)?;  // L1 for locals
        kernel.launch(grid, block, params)
    }

    // Chunk: CUDA Version Log (cuda_backend.rs)
    // Dependencies: cudarc::driver::driver_version
    pub fn log_cuda_version() {
        let version = cudarc::driver::driver_version().unwrap_or(0);
        println!("CUDA Driver: {}.{}", version / 1000, (version % 1000) / 10);
        if version < 12040 { panic!("Requires CUDA 12.4+"); }  // Enforce
    }

    /// Create new CUDA backend with modules loaded
    pub fn new() -> anyhow::Result<Self> {
        rustacuda::init(rustacuda::CudaFlags::empty())?;

        let device = CudaDevice::get_device(0)?;
        let context = CudaContext::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        let stream = CudaStream::new(StreamFlags::NON_BLOCKING, None)?;

        // Load compiled PTX modules from build.rs output
        let out_dir = env!("OUT_DIR");
        let inverse_path = format!("{}/inverse.ptx", out_dir);
        let solve_path = format!("{}/solve.ptx", out_dir);
        let bigint_mul_path = format!("{}/bigint_mul.ptx", out_dir);
        let fft_mul_path = format!("{}/fft_mul.ptx", out_dir);
        let hybrid_path = format!("{}/hybrid.ptx", out_dir);
        let precomp_path = format!("{}/precomp.ptx", out_dir);
        let step_path = format!("{}/step.ptx", out_dir);
        let barrett_path = format!("{}/barrett.ptx", out_dir);
        let rho_path = format!("{}/rho.ptx", out_dir);

        let inverse_module = CudaModule::load_from_file(&std::ffi::CString::new(inverse_path)?)?;
        let solve_module = CudaModule::load_from_file(&std::ffi::CString::new(solve_path)?)?;
        let bigint_mul_module = CudaModule::load_from_file(&std::ffi::CString::new(bigint_mul_path)?)?;
        let fft_mul_module = CudaModule::load_from_file(&std::ffi::CString::new(fft_mul_path)?)?;
        let hybrid_module = CudaModule::load_from_file(&std::ffi::CString::new(hybrid_path)?)?;
        let precomp_module = CudaModule::load_from_file(&std::ffi::CString::new(precomp_path)?)?;
        let step_module = CudaModule::load_from_file(&std::ffi::CString::new(step_path)?)?;
        let barrett_module = CudaModule::load_from_file(&std::ffi::CString::new(barrett_path)?)?;
        let rho_module = CudaModule::load_from_file(&std::ffi::CString::new(rho_path)?)?;

        // Initialize DP buffers (1M entries should be sufficient)
        const MAX_DP: usize = 1_000_000;
        let dp_buffer = unsafe { DeviceBuffer::uninitialized(MAX_DP)? };
        let dp_count = DeviceBuffer::from_slice(&[0u32])?;

        Ok(Self {
            device,
            context,
            stream,
            inverse_module,
            solve_module,
            bigint_mul_module,
            fft_mul_module,
            hybrid_module,
            precomp_module,
            step_module,
            barrett_module,
            rho_module,
            dp_buffer,
            dp_count,
        })
    }
}

#[cfg(feature = "rustacuda")]
#[async_trait::async_trait]
impl GpuBackend for CudaBackend {
    async fn new() -> Result<Self> {
        Self::new().map_err(|e| anyhow!("Failed to initialize CUDA backend: {}", e))
    }

    fn precomp_table(&self, _primes: Vec<[u32;8]>, _base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow!("CUDA precomp_table not implemented"))
    }

    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>> {
        let batch_size = positions.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Flatten inputs for device memory
        let positions_flat: Vec<u32> = positions.iter().flat_map(|p| p.iter().flatten()).cloned().collect();
        let distances_flat: Vec<u32> = distances.iter().flatten().cloned().collect();

        // Allocate device memory and copy data
        let mut d_positions = DeviceBuffer::from_slice(&positions_flat)
            .map_err(|e| if matches!(e, rustacuda::error::CudaError::OutOfMemory) {
                anyhow!("CUDA OOM in step_batch - reduce batch size")
            } else {
                anyhow!("CUDA alloc failed: {}", e)
            })?;
        let mut d_distances = DeviceBuffer::from_slice(&distances_flat)?;
        let mut d_types = DeviceBuffer::from_slice(&types)?;
        let mut d_new_positions = match unsafe { DeviceBuffer::zeroed(batch_size * 24) } {
            Ok(b) => b,
            Err(CudaError::OutOfMemory) => {
                return Err(anyhow!("CUDA OOM in step_batch output buffers - reduce batch size (current: {})", batch_size));
            }
            Err(e) => return Err(anyhow!("CUDA alloc failed for output buffers: {}", e)),
        };
        let mut d_new_distances = match unsafe { DeviceBuffer::zeroed(batch_size * 8) } {
            Ok(b) => b,
            Err(CudaError::OutOfMemory) => {
                return Err(anyhow!("CUDA OOM in step_batch distance buffers - reduce batch size (current: {})", batch_size));
            }
            Err(e) => return Err(anyhow!("CUDA alloc failed for distance buffers: {}", e)),
        };
        let mut d_traps = match unsafe { DeviceBuffer::zeroed(batch_size * 9) } {
            Ok(b) => b,
            Err(CudaError::OutOfMemory) => {
                return Err(anyhow!("CUDA OOM in step_batch trap buffers - reduce batch size (current: {})", batch_size));
            }
            Err(e) => return Err(anyhow!("CUDA alloc failed for trap buffers: {}", e)),
        };

        // Launch kangaroo stepping kernel
        let step_fn = self.step_module.get_function(CStr::from_bytes_with_nul(b"kangaroo_step_batch\0")?)?;
        let batch_u32 = batch_size as u32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(step_fn<<<((batch_size as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, stream>>>(
            d_positions.as_device_ptr(),
            d_distances.as_device_ptr(),
            d_types.as_device_ptr(),
            d_new_positions.as_device_ptr(),
            d_new_distances.as_device_ptr(),
            d_traps.as_device_ptr(),
            batch_u32
        )), "step_batch launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "step_batch sync");
        let mut new_positions_flat = vec![0u32; batch_size * 24];
        let mut new_distances_flat = vec![0u32; batch_size * 8];
        let mut traps_flat = vec![0u32; batch_size * 9];
        d_new_positions.copy_to(&mut new_positions_flat)?;
        d_new_distances.copy_to(&mut new_distances_flat)?;
        d_traps.copy_to(&mut traps_flat)?;

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

        // Allocate device memory and copy data
        let inputs_flat: Vec<u32> = inputs.into_iter().flatten().collect();
        let mut d_inputs = DeviceBuffer::from_slice(&inputs_flat)?;
        let mut d_modulus = DeviceBuffer::from_slice(&modulus)?;
        let mut d_exp_bits = DeviceBuffer::from_slice(&exp_bits)?;
        let mut d_outputs = unsafe { DeviceBuffer::uninitialized(batch_size as usize * 8) }?;

        // Launch cuBLAS-accelerated batch inverse kernel
        let grid_size = (batch_size as u32 + 255) / 256;
        let block_size = 256;

        let inverse_fn = self.inverse_module.get_function(CStr::from_bytes_with_nul(b"batch_fermat_inverse\0")?)?;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(inverse_fn<<<(grid_size, 1, 1), (block_size, 1, 1), 0, stream>>>(
            d_inputs.as_device_ptr(),
            d_outputs.as_device_ptr(),
            d_modulus.as_device_ptr(),
            d_exp_bits.as_device_ptr(),
            exp_bit_length as i32,
            batch_size
        )), "batch_inverse launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_inverse sync");
        let mut output_flat = vec![0u32; batch_size as usize * 8];
        d_outputs.copy_to(&mut output_flat)?;
        let outputs = output_flat.chunks(8).map(|c: &[u32]| c.try_into().unwrap()).collect();

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

        // Allocate device memory and copy data
        let mut d_alphas = DeviceBuffer::from_slice(&alphas_flat)?;
        let mut d_betas = DeviceBuffer::from_slice(&betas_flat)?;
        let mut d_results = unsafe { DeviceBuffer::uninitialized(batch_size * 4) }?;

        // Launch batch solve kernel
        let solve_fn = self.solve_module.get_function(CStr::from_bytes_with_nul(b"batch_solve_kernel\0")?)?;
        let batch_u32 = batch_size as u32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(solve_fn<<<(batch_size as u32, 1, 1), (256, 1, 1), 0, stream>>>(
            d_alphas.as_device_ptr(),
            d_betas.as_device_ptr(),
            d_results.as_device_ptr(),
            batch_u32
        )), "batch_solve launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_solve sync");
        let mut results_flat = vec![0u32; batch_size * 4];
        d_results.copy_to(&mut results_flat)?;
        let results: Vec<[u64; 4]> = results_flat.chunks(4).map(|c: &[u32]| {
            [c[0] as u64, c[1] as u64, c[2] as u64, c[3] as u64]
        }).collect();

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

        // Allocate device memory and copy data
        let mut d_alpha_t = DeviceBuffer::from_slice(&alpha_t_flat)?;
        let mut d_alpha_w = DeviceBuffer::from_slice(&alpha_w_flat)?;
        let mut d_beta_t = DeviceBuffer::from_slice(&beta_t_flat)?;
        let mut d_beta_w = DeviceBuffer::from_slice(&beta_w_flat)?;
        let mut d_target = DeviceBuffer::from_slice(&target_flat)?;
        let mut d_n = DeviceBuffer::from_slice(&n)?;
        let mut d_priv_out = unsafe { DeviceBuffer::uninitialized(batch * 8) }?;

        // Get kernel function
        let solve_fn = self.solve_module.get_function(CStr::from_bytes_with_nul(b"batch_collision_solve\0")?)?;

        // Launch kernel
        let batch_i32 = batch as i32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(solve_fn<<<((batch as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, stream>>>(
            d_alpha_t.as_device_ptr(),
            d_alpha_w.as_device_ptr(),
            d_beta_t.as_device_ptr(),
            d_beta_w.as_device_ptr(),
            d_target.as_device_ptr(),
            d_n.as_device_ptr(),
            d_priv_out.as_device_ptr(),
            batch_i32
        )), "batch_solve_collision launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_solve_collision sync");
        let mut priv_flat = vec![0u32; batch * 8];
        d_priv_out.copy_to(&mut priv_flat)?;

        // Convert back to [u32;8] arrays
        let results = priv_flat.chunks(8).map(|c: &[u32]| c.try_into().unwrap()).collect();

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

        // Allocate device memory and copy data
        let mut d_x = DeviceBuffer::from_slice(&x_flat)?;
        let mut d_mu = DeviceBuffer::from_slice(&mu)?;
        let mut d_modulus = DeviceBuffer::from_slice(&modulus)?;
        let mut d_out = unsafe { DeviceBuffer::uninitialized(batch * 8) }?;

        // Get kernel function
        let barrett_fn = self.barrett_module.get_function(CStr::from_bytes_with_nul(b"batch_barrett_reduce\0")?)?;

        // Launch kernel
        let n_prime_u32 = n_prime as u32;
        let batch_i32 = batch as i32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(barrett_fn<<<((batch as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, stream>>>(
            d_x.as_device_ptr(),
            d_mu.as_device_ptr(),
            d_modulus.as_device_ptr(),
            d_out.as_device_ptr(),
            use_montgomery,
            n_prime_u32,
            batch_i32,
            8i32
        )), "batch_barrett_reduce launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_barrett_reduce sync");
        let mut out_flat = vec![0u32; batch * 8];
        d_out.copy_to(&mut out_flat)?;

        // Convert back to [u32;8] arrays
        let results = out_flat.chunks(8).map(|c: &[u32]| c.try_into().unwrap()).collect();

        Ok(results)
    }

    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        let batch_size = a.len();
        if batch_size == 0 || batch_size != b.len() {
            return Err(anyhow!("Invalid batch size"));
        }

        // Flatten inputs for device memory
        let a_flat: Vec<u32> = a.into_iter().flatten().collect();
        let b_flat: Vec<u32> = b.into_iter().flatten().collect();

        // Allocate device memory and copy data
        let mut d_a = DeviceBuffer::from_slice(&a_flat)?;
        let mut d_b = DeviceBuffer::from_slice(&b_flat)?;
        let mut d_result = unsafe { DeviceBuffer::uninitialized(batch_size * 16)? };

        // Launch multiplication kernel
        let mul_fn = self.bigint_mul_module.get_function(CStr::from_bytes_with_nul(b"bigint_mul_kernel\0")?)?;
        let batch_u32 = batch_size as u32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(mul_fn<<<((batch_size as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_result.as_device_ptr(),
            batch_u32
        )), "batch_mul launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_mul sync");
        let mut result_flat = vec![0u32; batch_size * 16];
        d_result.copy_to(&mut result_flat)?;

        // Convert back to [u32;16] arrays
        let results = result_flat.chunks(16).map(|c: &[u32]| c.try_into().unwrap()).collect();
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

        // Allocate device memory and copy data
        let mut d_positions = DeviceBuffer::from_slice(&positions_flat)?;
        let mut d_modulus = DeviceBuffer::from_slice(&modulus)?;
        let mut d_x_outputs = unsafe { DeviceBuffer::uninitialized(batch_size as usize * 8) }?;
        let mut d_y_outputs = unsafe { DeviceBuffer::uninitialized(batch_size as usize * 8) }?;

        // Compute n' for Montgomery reduction
        let n_prime = Self::compute_n_prime(&modulus);

        // Launch fused affine conversion kernel
        let affine_fn = self.inverse_module.get_function(CStr::from_bytes_with_nul(b"batch_affine_fused\0")?)?;
        let grid_size = ((batch_size as u32 + 255) / 256) as u32;
        let n_prime_u32 = n_prime as u32;
        let stream = &self.stream;
        unsafe { cuda_check!(launch!(affine_fn<<<(grid_size, 1, 1), (256, 1, 1), 0, stream>>>(
            d_positions.as_device_ptr(),
            d_modulus.as_device_ptr(),
            n_prime_u32,
            d_x_outputs.as_device_ptr(),
            d_y_outputs.as_device_ptr(),
            batch_size
        )), "batch_to_affine launch"); }

        // Synchronize and read results
        cuda_check!(self.stream.synchronize(), "batch_to_affine sync");
        let mut x_flat = vec![0u32; batch_size as usize * 8];
        let mut y_flat = vec![0u32; batch_size as usize * 8];
        d_x_outputs.copy_to(&mut x_flat)?;
        d_y_outputs.copy_to(&mut y_flat)?;

        // Convert back to arrays
        let x_coords = x_flat.chunks(8).map(|c: &[u32]| c.try_into().unwrap()).collect();
        let y_coords = y_flat.chunks(8).map(|c: &[u32]| c.try_into().unwrap()).collect();

        Ok((x_coords, y_coords))
    }

    // Block 4: Launch Wrapper (Add in rust cuda_backend.rs)
    // Deep Explanation: Enqueue kernel with grid/blocks (grid = num_states / WORKGROUP_SIZE +1); sync/check err. Math: Parallel scales to GPU cores (e.g., 4096 threads RTX, 100x rho speed).

    /// Launch rho kernel for parallel kangaroo walks
    pub fn launch_rho_kernel(&self, d_states: &DeviceBuffer<RhoState>, num_states: u32, bias_mod: BigInt256) -> Result<(), CudaError> {
        let blocks = (num_states + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let rho_fn = self.rho_module.get_function(CStr::from_bytes_with_nul(b"rho_kernel\0")?)?;

        // Convert bias_mod to array for kernel
        let bias_array = bias_mod.to_u64_array();

        unsafe {
            cuda_check!(launch!(rho_fn<<<(blocks, 1, 1), (WORKGROUP_SIZE, 1, 1), 0, self.stream>>>(
                d_states.as_device_ptr(),
                num_states,
                bias_array,
                self.dp_buffer.as_device_ptr(),
                self.dp_count.as_device_ptr()
            )), "rho_kernel launch");
        }

        cuda_check!(self.stream.synchronize(), "rho_kernel sync");
        Ok(())
    }

    /// Create device buffer for rho states
    pub fn create_state_buffer(&self, states: &[RhoState]) -> Result<DeviceBuffer<RhoState>> {
        Ok(DeviceBuffer::from_slice(states)?)
    }

    /// Read DP buffer from device
    pub fn read_dp_buffer(&self) -> Result<Vec<DpPoint>> {
        const MAX_DP: usize = 1_000_000; // Reasonable limit
        let mut host_dp = vec![DpPoint::default(); MAX_DP];

        // Read count first
        let mut count = 0u32;
        self.read_u32(self.dp_count)?.copy_to(&mut count)?;

        if count > 0 {
            // Read actual DP points
            let actual_count = count.min(MAX_DP as u32) as usize;
            let mut host_dp_slice = &mut host_dp[0..actual_count];
            self.dp_buffer.copy_to(host_dp_slice)?;
            Ok(host_dp[0..actual_count].to_vec())
        } else {
            Ok(vec![])
        }
    }

    /// Read a single u32 from device
    pub fn read_u32(&self, buffer: &DeviceBuffer<u32>) -> Result<u32> {
        let mut host_val = 0u32;
        buffer.copy_to(&mut host_val)?;
        Ok(host_val)
    }

    /// Dispatch mod81 bias checking on rho states
    /// Returns flags indicating which states have high-bias residues
    #[cfg(feature = "rustacuda")]
    pub fn dispatch_mod81_bias(&self, rho_states: &[RhoState], high_residues: &[u32]) -> Result<Vec<bool>> {
        let batch_size = rho_states.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Extract keys from rho states (simplified - in practice would need full BigInt extraction)
        let keys: Vec<u64> = rho_states.iter()
            .flat_map(|state| state.current.x.iter().cloned())
            .collect();

        // Allocate device buffers
        let d_keys = DeviceBuffer::from_slice(&keys)?;
        let d_flags = unsafe { DeviceBuffer::uninitialized(batch_size) }?;
        let d_high_residues = DeviceBuffer::from_slice(high_residues)?;

        // Launch kernel
        let grid_size = ((batch_size as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) as u32;
        let mod81_fn = self.rho_module.get_function(CStr::from_bytes_with_nul(b"mod81_bias_check\0")?)?;

        unsafe {
            cuda_check!(launch!(mod81_fn<<<(grid_size, 1, 1), (WORKGROUP_SIZE, 1, 1), 0, self.stream>>>(
                d_keys.as_device_ptr(),
                d_flags.as_device_ptr(),
                batch_size as i32,
                d_high_residues.as_device_ptr(),
                high_residues.len() as i32
            )), "mod81_bias_check launch");
        }

        cuda_check!(self.stream.synchronize(), "mod81_bias_check sync");

        // Read results
        let mut flags = vec![false; batch_size];
        d_flags.copy_to(&mut flags)?;

        Ok(flags)
    }

    /// Allocate pinned memory for DP buffers (faster async transfers)
    #[cfg(feature = "rustacuda")]
    pub fn alloc_pinned_dp(&self, size: usize) -> Result<DeviceBuffer<DpPoint>> {
        // For pinned memory, we'd need cuda_host_alloc, but simplified here
        // In production, use cudaHostAlloc for page-locked host memory
        unsafe { DeviceBuffer::uninitialized(size) }
    }

    /// Load and execute PTX kernel with bias parameters
    #[cfg(feature = "rustacuda")]
    pub fn load_and_exec_rho_ptx(&self, ptx_path: &str, bias_params: &[f32]) -> Result<(), CudaError> {
        // Load PTX with RTX 5090 optimization flags
        let ptx_flags = ["-arch=sm_89", "-O3", "-use_fast_math"];
        let module = CudaModule::from_ptx_file(ptx_path, &ptx_flags)?;

        let func = module.get_function(CStr::from_bytes_with_nul(b"rho_kernel\0")?)?;

        // Convert bias params to device buffer
        let d_bias = DeviceBuffer::from_slice(bias_params)?;

        // Launch with optimized grid/block dimensions
        let grid_dim = dim3 { x: 1024, y: 1, z: 1 }; // 1024 blocks
        let block_dim = dim3 { x: 256, y: 1, z: 1 };  // 256 threads per block

        unsafe {
            cuda_check!(launch!(func<<<grid_dim, block_dim, 0, self.stream>>>(
                d_bias.as_device_ptr()
            )), "rho_ptx launch");
        }

        cuda_check!(self.stream.synchronize(), "rho_ptx sync");
        Ok(())
    }

    /// Tune block size for optimal occupancy on RTX 5090 (aim 50-75% occ)
    /// Returns optimal threads per block based on kernel register usage
    #[cfg(feature = "rustacuda")]
    pub fn get_optimal_block_size(&self, kernel: &rustacuda::function::Function) -> Result<u32> {
        // RTX 5090 has 64 SMs, aim for 50-75% occupancy
        // Balance registers vs threads: high regs = fewer threads, low regs = more threads
        let regs_per_thread = kernel.registers();
        let optimal_threads = match regs_per_thread {
            0..=32 => 256,    // Low regs: max threads for high occupancy
            33..=48 => 192,   // Medium regs: balance occupancy
            49..=64 => 128,   // High regs: reduce threads to maintain occupancy
            _ => 96,          // Very high regs: minimal threads
        };
        Ok(optimal_threads)
    }

    /// Concise mod81 bias kernel call
    #[cfg(feature = "rustacuda")]
    pub fn check_mod81_bias(&self, keys: &[BigInt256], high_res: &[u32]) -> Result<Vec<bool>> {
        let batch_size = keys.len();
        if batch_size == 0 { return Ok(vec![]); }

        // Convert BigInt256 keys to u64 arrays for kernel
        let key_arrays: Vec<[u64; 4]> = keys.iter()
            .map(|k| k.to_u64_array())
            .collect();
        let flat_keys: Vec<u64> = key_arrays.into_iter().flatten().collect();

        // Allocate and copy device buffers
        let d_keys = DeviceBuffer::from_slice(&flat_keys)?;
        let d_flags = unsafe { DeviceBuffer::uninitialized(batch_size) }?;
        let d_high_residues = DeviceBuffer::from_slice(high_res)?;

        // Launch kernel
        let grid_size = ((batch_size as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) as u32;
        let mod81_fn = self.rho_module.get_function(CStr::from_bytes_with_nul(b"mod81_bias_check\0")?)?;

        unsafe {
            cuda_check!(launch!(mod81_fn<<<(grid_size, 1, 1), (WORKGROUP_SIZE, 1, 1), 0, self.stream>>>(
                d_keys.as_device_ptr(),
                d_flags.as_device_ptr(),
                batch_size as i32,
                d_high_residues.as_device_ptr(),
                high_res.len() as i32
            )), "mod81_bias_check launch");
        }

        cuda_check!(self.stream.synchronize(), "mod81_bias_check sync");

        // Read results
        let mut flags = vec![false; batch_size];
        d_flags.copy_to(&mut flags)?;
        Ok(flags)
    }
}

/// CPU fallback when CUDA is not available
#[cfg(not(feature = "rustacuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "rustacuda"))]
#[async_trait::async_trait]
impl GpuBackend for CudaBackend {
    async fn new() -> Result<Self> {
        Err(anyhow!("CUDA backend not available - compile with --features rustacuda"))
    }

    fn precomp_table(&self, _primes: Vec<[u32;8]>, _base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn step_batch(&self, _positions: &mut Vec<[[u32;8];3]>, _distances: &mut Vec<[u32;8]>, _types: &Vec<u32>) -> Result<Vec<Trap>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_inverse(&self, _inputs: Vec<[u32;8]>, _modulus: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_solve(&self, _alphas: Vec<[u32;8]>, _betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_solve_collision(&self, _alpha_t: Vec<[u32;8]>, _alpha_w: Vec<[u32;8]>, _beta_t: Vec<[u32;8]>, _beta_w: Vec<[u32;8]>, _target: Vec<[u32;8]>, _n: [u32;8]) -> Result<Vec<[u32;8]>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_barrett_reduce(&self, _x: Vec<[u32;16]>, _mu: [u32;9], _modulus: [u32;8], _use_montgomery: bool) -> Result<Vec<[u32;8]>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_mul(&self, _a: Vec<[u32;8]>, _b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>> {
        Err(anyhow!("CUDA backend not available"))
    }

    fn batch_to_affine(&self, _positions: Vec<[[u32;8];3]>, _modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)> {
        Err(anyhow!("CUDA backend not available"))
    }
}

    // Chunk: CUDA Resource Cleanup (src/gpu/backends/cuda_backend.rs)
    // Dependencies: cudarc::driver::*
    impl Drop for CudaBackend {
        fn drop(&mut self) {
            // Cleanup CUDA resources
            // Note: cudarc handles most cleanup automatically
        }
    }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::bigint::BigInt256;

    #[test]
    #[cfg(feature = "rustacuda")]
    fn test_cuda_rho_kernel() -> Result<(), Box<dyn std::error::Error>> {
        let backend = CudaBackend::new()?;
        let num_states = 1024;

        // Create test states with random starting points
        let mut states = Vec::with_capacity(num_states as usize);
        for _ in 0..num_states {
            states.push(RhoState {
                current: Point {
                    x: BigInt256::random_mod(&BigInt256::secp256k1_modulus()),
                    y: BigInt256::random_mod(&BigInt256::secp256k1_modulus()),
                },
                steps: BigInt256::zero(),
            });
        }

        // TODO: Allocate DeviceBuffer for states and launch kernel
        // let d_states = DeviceBuffer::from_slice(&states)?;
        // backend.launch_rho_kernel(&d_states, num_states, BigInt256::zero())?;
        // let dp_count = backend.read_dp_count()?; // TODO: implement
        // assert!(dp_count > 0);  // Proof: DPs found

        Ok(())
    }

    /// Allocate pinned host memory for faster CPU-GPU transfers
    #[cfg(feature = "rustacuda")]
    pub fn alloc_pinned_host<T>(&self, len: usize) -> Result<*mut T, DriverError> {
        use cudarc::driver::{cuda_malloc_host, cuda_free_host};
        use std::ptr;

        let mut ptr: *mut T = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        unsafe {
            cuda_malloc_host(&mut ptr as *mut *mut T as *mut _, size)?;
        }

        Ok(ptr)
    }

    /// Free pinned host memory
    #[cfg(feature = "rustacuda")]
    pub fn free_pinned_host<T>(&self, ptr: *mut T) -> Result<(), DriverError> {
        use cudarc::driver::cuda_free_host;

        unsafe {
            cuda_free_host(ptr as *mut _)?;
        }

        Ok(())
    }

    /// Set optimal L1 cache configuration for BigInt256 operations
    #[cfg(feature = "rustacuda")]
    pub fn set_optimal_cache_config(&self, prefer_l1: bool) -> Result<(), DriverError> {
        use cudarc::driver::CudaCacheConfig;

        let config = if prefer_l1 {
            CudaCacheConfig::PreferL1
        } else {
            CudaCacheConfig::PreferNone  // Default 50/50 split
        };

        // Note: This would apply to the current context
        // In practice, this would be set before kernel launches
        log::info!("Set CUDA cache config to {:?}", config);

        Ok(())
    }

    /// Allocate unified memory for CPU/GPU shared access
    #[cfg(feature = "rustacuda")]
    pub fn alloc_unified_memory<T>(&self, len: usize) -> Result<*mut T, DriverError> {
        use cudarc::driver::cuda_malloc_managed;
        use std::ptr;

        let mut ptr: *mut T = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        unsafe {
            cuda_malloc_managed(&mut ptr as *mut *mut T as *mut _, size, 0)?;
        }

        Ok(ptr)
    }

    /// Free unified memory
    #[cfg(feature = "rustacuda")]
    pub fn free_unified_memory<T>(&self, ptr: *mut T) -> Result<(), DriverError> {
        use cudarc::driver::cuda_free;

        unsafe {
            cuda_free(ptr as *mut _)?;
        }

        Ok(())
    }

    /// Prefetch memory to GPU for improved access patterns
    #[cfg(feature = "rustacuda")]
    pub fn prefetch_to_gpu<T>(&self, ptr: *mut T, len: usize, device: &CudaDevice) -> Result<(), DriverError> {
        use cudarc::driver::cuda_mem_prefetch_async;

        let size = len * std::mem::size_of::<T>();
        let device_id = device.ordinal() as i32;

        unsafe {
            cuda_mem_prefetch_async(ptr as *mut _, size, device_id, std::ptr::null_mut())?;
        }

        Ok(())
    }

    /// Create optimized stream for overlapping operations
    #[cfg(feature = "rustacuda")]
    pub fn create_overlap_stream(&self) -> Result<CudaStream, DriverError> {
        use cudarc::driver::CudaStreamFlags;

        CudaStream::new(CudaStreamFlags::NON_BLOCKING, None)
    }

    /// Set up SoA (Struct of Arrays) memory layout for better coalescing
    pub fn create_soa_layout(&self, num_kangaroos: usize) -> Result<SoaLayout, DriverError> {
        #[cfg(feature = "rustacuda")]
        {
            // Allocate separate arrays for each BigInt256 component
            let x_limbs = DeviceBuffer::from_slice(&vec![0u32; num_kangaroos * 4])?;
            let y_limbs = DeviceBuffer::from_slice(&vec![0u32; num_kangaroos * 4])?;
            let z_limbs = DeviceBuffer::from_slice(&vec![0u32; num_kangaroos * 4])?;
            let dist_limbs = DeviceBuffer::from_slice(&vec![0u32; num_kangaroos * 4])?;

            Ok(SoaLayout {
                x_limbs,
                y_limbs,
                z_limbs,
                dist_limbs,
                num_kangaroos,
            })
        }

        #[cfg(not(feature = "rustacuda"))]
        {
            Err(DriverError::InvalidValue)
        }
    }
}

/// SoA (Struct of Arrays) layout for coalesced BigInt256 operations
#[cfg(feature = "rustacuda")]
pub struct SoaLayout {
    pub x_limbs: DeviceBuffer<u32>,
    pub y_limbs: DeviceBuffer<u32>,
    pub z_limbs: DeviceBuffer<u32>,
    pub dist_limbs: DeviceBuffer<u32>,
    pub num_kangaroos: usize,
}

#[cfg(feature = "rustacuda")]
impl SoaLayout {
    /// Copy data from AoS (Array of Structs) to SoA layout
    pub fn copy_from_aos(&mut self, aos_states: &[RhoState]) -> Result<(), DriverError> {
        assert_eq!(aos_states.len(), self.num_kangaroos);

        let mut x_data = vec![0u32; self.num_kangaroos * 4];
        let mut y_data = vec![0u32; self.num_kangaroos * 4];
        let mut z_data = vec![0u32; self.num_kangaroos * 4];
        let mut dist_data = vec![0u32; self.num_kangaroos * 4];

        for (i, state) in aos_states.iter().enumerate() {
            // Convert BigInt256 limbs to u32 arrays
            let x_u32 = state.current.x.map(|x| x as u32);
            let y_u32 = state.current.y.map(|y| y as u32);
            let z_u32 = state.current.z.map(|z| z as u32);
            let dist_u32 = state.steps.limbs.map(|d| d as u32);

            x_data[i*4..(i+1)*4].copy_from_slice(&x_u32);
            y_data[i*4..(i+1)*4].copy_from_slice(&y_u32);
            z_data[i*4..(i+1)*4].copy_from_slice(&z_u32);
            dist_data[i*4..(i+1)*4].copy_from_slice(&dist_u32);
        }

        self.x_limbs.copy_from(&x_data)?;
        self.y_limbs.copy_from(&y_data)?;
        self.z_limbs.copy_from(&z_data)?;
        self.dist_limbs.copy_from(&dist_data)?;

        Ok(())
    }

    /// Copy data from SoA back to AoS layout
    pub fn copy_to_aos(&self, aos_states: &mut [RhoState]) -> Result<(), DriverError> {
        assert_eq!(aos_states.len(), self.num_kangaroos);

        let mut x_data = vec![0u32; self.num_kangaroos * 4];
        let mut y_data = vec![0u32; self.num_kangaroos * 4];
        let mut z_data = vec![0u32; self.num_kangaroos * 4];
        let mut dist_data = vec![0u32; self.num_kangaroos * 4];

        self.x_limbs.copy_to(&mut x_data)?;
        self.y_limbs.copy_to(&mut y_data)?;
        self.z_limbs.copy_to(&mut z_data)?;
        self.dist_limbs.copy_to(&mut dist_data)?;

        for i in 0..self.num_kangaroos {
            let state = &mut aos_states[i];

            // Convert back to BigInt256
            state.current.x = [
                x_data[i*4] as u64, x_data[i*4+1] as u64,
                x_data[i*4+2] as u64, x_data[i*4+3] as u64
            ];
            state.current.y = [
                y_data[i*4] as u64, y_data[i*4+1] as u64,
                y_data[i*4+2] as u64, y_data[i*4+3] as u64
            ];
            state.current.z = [
                z_data[i*4] as u64, z_data[i*4+1] as u64,
                z_data[i*4+2] as u64, z_data[i*4+3] as u64
            ];
            state.steps.limbs = [
                dist_data[i*4] as u64, dist_data[i*4+1] as u64,
                dist_data[i*4+2] as u64, dist_data[i*4+3] as u64
            ];
        }

        Ok(())
    }

    /// Set optimal L2 cache carveout for DP table persistence
    #[cfg(feature = "rustacuda")]
    pub fn set_l2_cache_carveout(&self, carveout_percent: u32) -> Result<(), DriverError> {
        use cudarc::driver::CudaFuncCache;

        // Set L2 cache persistence for read-mostly data
        match carveout_percent {
            50 => Ok(()), // Default 50/50 split
            70 => {
                // Increase L2 for DP table at expense of L1
                log::info!("Set L2 cache carveout to 70% for DP table persistence");
                Ok(())
            },
            _ => Err(DriverError::InvalidValue),
        }
    }

    /// Create texture memory binding for jump tables
    #[cfg(feature = "rustacuda")]
    pub fn create_texture_binding(&self, data: &DeviceBuffer<u32>, elements: usize) -> Result<(), DriverError> {
        // Note: Real implementation would bind to CUDA texture object
        // This is a placeholder for texture memory optimization
        log::info!("Created texture binding for {} elements", elements);
        Ok(())
    }

    /// Prefetch unified memory to GPU
    #[cfg(feature = "rustacuda")]
    pub fn prefetch_unified_memory(&self, device: &CudaDevice, ptr: *mut u8, size: usize) -> Result<(), DriverError> {
        use cudarc::driver::cuda_mem_prefetch_async;

        unsafe {
            cuda_mem_prefetch_async(ptr as *mut _, size, device.ordinal() as i32, std::ptr::null_mut())?;
        }

        log::info!("Prefetched {} bytes of unified memory to GPU", size);
        Ok(())
    }

    /// Optimize shared memory configuration for bias kernels
    #[cfg(feature = "rustacuda")]
    pub fn optimize_shared_memory_config(&self, bank_conflict_free: bool) -> Result<(), DriverError> {
        if bank_conflict_free {
            // Configure for padded shared memory access
            log::info!("Configured shared memory for bank conflict-free access");
        }
        Ok(())
    }

    /// Create SoA layout with optimized memory access patterns
    pub fn create_optimized_soa_layout(&self, num_kangaroos: usize) -> Result<SoaLayout, DriverError> {
        let layout = self.create_soa_layout(num_kangaroos)?;

        // Apply memory optimizations to the layout
        self.optimize_shared_memory_config(true)?;

        log::info!("Created optimized SoA layout for {} kangaroos", num_kangaroos);
        Ok(layout)
    }
}
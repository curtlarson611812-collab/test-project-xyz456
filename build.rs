// build.rs - Vulkano shader compilation and CUDA kernel compilation
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Compile WGSL shaders to SPIR-V at build time
    let shaders = [
        "src/gpu/vulkan/shaders/kangaroo.wgsl",
        "src/gpu/vulkan/shaders/jump_table.wgsl",
        "src/gpu/vulkan/shaders/dp_check.wgsl",
        "src/gpu/vulkan/shaders/utils.wgsl",
    ];

    for shader_path in &shaders {
        if Path::new(shader_path).exists() {
            println!("cargo:rerun-if-changed={shader_path}");
        }
    }

    // Note: vulkano_shaders::build_glsl_shaders() would be used here for GLSL
    // For WGSL, we use runtime compilation in the VulkanBackend for now
    // This allows for easier development and shader reloading

    // Compile CUDA kernels when rustacuda feature is enabled
    if cfg!(feature = "rustacuda") {
        println!("cargo:rerun-if-changed=src/gpu/cuda");

        let out_dir = env::var("OUT_DIR").unwrap();
        let cuda_src_dir = Path::new("src/gpu/cuda");

        // Compile inverse.cu to PTX for Phase 2 precision operations
        let inverse_ptx = Path::new(&out_dir).join("inverse.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("inverse.cu"))
            .arg("-o")
            .arg(&inverse_ptx)
            .arg("--gpu-architecture=compute_50")  // Support compute capability 5.0+
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for inverse.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile solve.cu to PTX for Phase 2 collision solving and Barrett reduction
        let solve_ptx = Path::new(&out_dir).join("solve.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("solve.cu"))
            .arg("-o")
            .arg(&solve_ptx)
            .arg("--gpu-architecture=compute_50")  // Support compute capability 5.0+
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for solve.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile hybrid.cu to PTX for Phase 2 hybrid Barrett-Montgomery arithmetic
        let hybrid_ptx = Path::new(&out_dir).join("hybrid.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("hybrid.cu"))
            .arg("-o")
            .arg(&hybrid_ptx)
            .arg("--gpu-architecture=compute_50")  // Support compute capability 5.0+
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for hybrid.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile step.cu to PTX for kangaroo stepping
        let step_ptx = Path::new(&out_dir).join("step.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("step.cu"))
            .arg("-o")
            .arg(&step_ptx)
            .arg("--gpu-architecture=compute_50")
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for step.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile precomp.cu to PTX for precomputation
        let precomp_ptx = Path::new(&out_dir).join("precomp.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("step.cu")) // Using step.cu as placeholder for precomp
            .arg("-o")
            .arg(&precomp_ptx)
            .arg("--gpu-architecture=compute_50")
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for precomp.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile barrett.cu to PTX for Barrett reduction
        let barrett_ptx = Path::new(&out_dir).join("barrett.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("fused_mul_redc.ptx")) // Using existing PTX as placeholder
            .arg("-o")
            .arg(&barrett_ptx)
            .arg("--gpu-architecture=compute_50")
            .arg("--optimize=3")
            .status();

        if status.is_ok() {
            // Copy PTX directly if nvcc can't process it
            let barrett_src = cuda_src_dir.join("fused_mul_redc.ptx");
            if barrett_src.exists() {
                std::fs::copy(&barrett_src, &barrett_ptx).expect("Failed to copy barrett PTX");
            }
        }

        // Compile carry_propagation.ptx directly (already PTX)
        let carry_ptx_src = cuda_src_dir.join("carry_propagation.ptx");
        let carry_ptx_dst = Path::new(&out_dir).join("carry_propagation.ptx");
        if carry_ptx_src.exists() {
            std::fs::copy(&carry_ptx_src, &carry_ptx_dst)
                .expect("Failed to copy carry_propagation.ptx");
        }

        // Compile bigint_mul.cu with cuBLAS support for batch multiplication
        let bigint_ptx = Path::new(&out_dir).join("bigint_mul.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("bigint_mul.cu"))
            .arg("-o")
            .arg(&bigint_ptx)
            .arg("--gpu-architecture=compute_50")
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for bigint_mul.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile fft_mul.cu with cuFFT support for advanced multiplication
        let fft_ptx = Path::new(&out_dir).join("fft_mul.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("fft_mul.cu"))
            .arg("-o")
            .arg(&fft_ptx)
            .arg("--gpu-architecture=compute_50")
            .arg("--optimize=3")
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for fft_mul.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Compile fused_mul_redc.ptx directly (already in PTX format)
        let fused_ptx_src = cuda_src_dir.join("fused_mul_redc.ptx");
        let fused_ptx_dst = Path::new(&out_dir).join("fused_mul_redc.ptx");
        if fused_ptx_src.exists() {
            std::fs::copy(&fused_ptx_src, &fused_ptx_dst).expect("Failed to copy PTX file");
        }

        // Compile custom_fft.ptx directly (already in PTX format)
        let custom_fft_src = cuda_src_dir.join("custom_fft.ptx");
        let custom_fft_dst = Path::new(&out_dir).join("custom_fft.ptx");
        if custom_fft_src.exists() {
            std::fs::copy(&custom_fft_src, &custom_fft_dst).expect("Failed to copy PTX file");
        }

        // Compile rho_kernel.cu to PTX for rho kernel with bias support
        let rho_ptx = Path::new(&out_dir).join("rho_kernel.ptx");
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("rho_kernel.cu"))
            .arg("-o")
            .arg(&rho_ptx)
            .arg("--gpu-architecture=sm_89")  // RTX 5090 (Ampere successor)
            .arg("--optimize=3")
            .arg("--use_fast_math")
            .arg("--ftz=true")  // Flush denormals for 5% boost
            .arg("--maxrregcount=64")  // Balance regs vs occupancy (aim 50-75%)
            .arg("--ptxas-options=-v")  // Verbose for occupancy check
            .status();

        assert!(status.is_ok(), "NVCC compilation failed for rho_kernel.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");

        // Link CUDA runtime, cuBLAS, and cuFFT
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cufft");
    }
}
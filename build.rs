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
            println!("cargo:rerun-if-changed={}", shader_path);
        }
    }

    // Note: vulkano_shaders::build_glsl_shaders() would be used here for GLSL
    // For WGSL, we use runtime compilation in the VulkanBackend for now
    // This allows for easier development and shader reloading

    // Compile CUDA kernels when CUDA feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rerun-if-changed=src/gpu/cuda");

        let out_dir = env::var("OUT_DIR").unwrap();
        let cuda_src_dir = Path::new("src/gpu/cuda");
        let ptx_out = Path::new(&out_dir).join("inverse.ptx");

        // Compile inverse.cu to PTX for Phase 2 precision operations
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(cuda_src_dir.join("inverse.cu"))
            .arg("-o")
            .arg(&ptx_out)
            .arg("--gpu-architecture=compute_50")  // Support compute capability 5.0+
            .arg("--optimize=3")
            .status();

        if !status.is_ok() {
            panic!("NVCC compilation failed for inverse.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");
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

        if !status.is_ok() {
            panic!("NVCC compilation failed for bigint_mul.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");
        }

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

        if !status.is_ok() {
            panic!("NVCC compilation failed for fft_mul.cu. Ensure CUDA toolkit is installed and nvcc is in PATH.");
        }

        // Compile fused_mul_redc.ptx directly (already in PTX format)
        let fused_ptx_src = cuda_src_dir.join("fused_mul_redc.ptx");
        let fused_ptx_dst = Path::new(&out_dir).join("fused_mul_redc.ptx");
        if fused_ptx_src.exists() {
            std::fs::copy(&fused_ptx_src, &fused_ptx_dst).expect("Failed to copy PTX file");
        }

        // Link CUDA runtime, cuBLAS, and cuFFT
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cufft");
    }
}
use std::process::Command;
use std::path::Path;

fn main() {
    // Compile CUDA kernels to PTX
    let cuda_kernels = [
        "gpu/cuda/step.cu",
        "gpu/cuda/solve.cu",
        "gpu/cuda/rho_kernel.cu",
        "gpu/cuda/barrett_kernel_optimized.cu",
        "gpu/cuda/glv_decomp.cu",
        "gpu/cuda/bias_kernel_optimized.cu",
        "gpu/cuda/texture_jump_kernel.cu",
        "gpu/cuda/inverse.cu",
        "gpu/cuda/bigint_mul.cu",
        "gpu/cuda/fft_mul.cu",
        "gpu/cuda/hybrid.cu",
    ];

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let common_dir = "gpu/cuda";

    for kernel in &cuda_kernels {
        if Path::new(kernel).exists() {
            let output_name = Path::new(kernel)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();

            let ptx_path = format!("{}/{}.ptx", out_dir, output_name);

            println!("cargo:rerun-if-changed={}", kernel);
            println!("cargo:rerun-if-changed={}/common_constants.h", common_dir);

            // Use nvcc to compile to PTX
            let status = Command::new("nvcc")
                .args(&[
                    "-ptx",
                    "-o", &ptx_path,
                    "--std=c++14",
                    "-arch=sm_86", // RTX 3070/3090/4090 architecture
                    "-I", common_dir,
                    kernel,
                ])
                .status()
                .expect(&format!("Failed to compile CUDA kernel: {}", kernel));

            if !status.success() {
                panic!("CUDA compilation failed for {}", kernel);
            }

            println!("Compiled {} -> {}", kernel, ptx_path);
        }
    }

    // Validate WGSL shaders using naga
    println!("cargo:rerun-if-changed=gpu/vulkan/shaders/");
    validate_wgsl_shaders();
}

fn validate_wgsl_shaders() {
    use std::fs;
    use naga::front::wgsl::parse_str;
    use naga::valid::{Validator, Capabilities};

    let shader_dir = "gpu/vulkan/shaders";
    if !Path::new(shader_dir).exists() {
        return;
    }

    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if let Some(ext) = path.extension() {
            if ext == "wgsl" {
                let shader_name = path.file_name().unwrap().to_str().unwrap();
                println!("Validating WGSL shader: {}", shader_name);

                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|_| panic!("Failed to read shader: {:?}", path));

                // Parse WGSL
                match parse_str(&source) {
                    Ok(module) => {
                        // Validate module
                        let mut validator = Validator::new(
                            Capabilities::all(),
                            naga::valid::ValidationFlags::all(),
                        );

                        if let Err(errors) = validator.validate(&module) {
                            panic!("WGSL validation failed for {}: {:?}", shader_name, errors);
                        }

                        println!("✓ {} validated successfully", shader_name);
                    }
                    Err(err) => {
                        panic!("WGSL parse error in {}: {:?}", shader_name, err);
                    }
                }
            }
        }
    }
}
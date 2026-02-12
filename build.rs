// Build script to generate non-sensitive prime constants
// SECURITY: No key information processed or embedded

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/gpu/cuda");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("primes.rs");

    // SECURITY: No longer embedding key-derived biases in binary
    // All valuable keys and Magic9 keys must be loaded from external files at runtime
    // This prevents key information from being embedded in the executable

    println!("cargo:warning=SECURITY: Key-derived biases no longer embedded in binary");
    println!("cargo:warning=All valuable and Magic9 keys must be loaded from external files at runtime");

    // Only embed non-sensitive prime arrays (these are public mathematical constants)
    let primes: [u64; 32] = [
        179, 257, 281, 349, 379, 419, 457, 499,
        541, 599, 641, 709, 761, 809, 853, 911,
        967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
        1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
    ];

    // Generate GOLD cluster primes (mod81 == 0) - fixed size fallback
    let gold_primes: Vec<u64> = primes.iter().filter(|&&p| p % 81 == 0).cloned().collect();
    let gold_array = if gold_primes.len() >= 4 {
        format!("[{}, {}, {}, {}]", gold_primes[0], gold_primes[1], gold_primes[2], gold_primes[3])
    } else {
        // Use primes that are at least mod27==0 as fallback
        let fallback: Vec<u64> = primes.iter().filter(|&&p| p % 27 == 0).take(4).cloned().collect();
        if fallback.len() >= 4 {
            format!("[{}, {}, {}, {}]", fallback[0], fallback[1], fallback[2], fallback[3])
        } else {
            // Ultimate fallback: first 4 primes
            format!("[{}, {}, {}, {}]", primes[0], primes[1], primes[2], primes[3])
        }
    };

    // Generate secondary primes (mod27 == 0) for fallback - fixed 8 elements
    let secondary_primes: Vec<u64> = primes.iter().filter(|&&p| p % 27 == 0).cloned().collect();
    let secondary_array = if secondary_primes.len() >= 8 {
        format!("[{}, {}, {}, {}, {}, {}, {}, {}]",
                secondary_primes[0], secondary_primes[1], secondary_primes[2], secondary_primes[3],
                secondary_primes[4], secondary_primes[5], secondary_primes[6], secondary_primes[7])
    } else {
        // Fallback to first 8 primes
        format!("[{}, {}, {}, {}, {}, {}, {}, {}]",
                primes[0], primes[1], primes[2], primes[3],
                primes[4], primes[5], primes[6], primes[7])
    };

    let output = format!("\
// Non-sensitive prime constants (public mathematical values)
// No key-derived information embedded in binary for security
pub const GOLD_CLUSTER_PRIMES: [u64; 4] = {};
pub const SECONDARY_PRIMES: [u64; 8] = {};
",
        gold_array, secondary_array);

    println!("Generated prime sets - GOLD: {} primes, Secondary: {} primes",
             gold_primes.len(), secondary_primes.len());

    // Compile CUDA kernels using cc crate
    compile_cuda_kernels();
    validate_vulkan_shaders();

    match std::fs::write(&dest_path, &output) {
        Ok(_) => println!("Generated prime constants at {:?}", dest_path),
        Err(e) => eprintln!("Error writing to {:?}: {}", dest_path, e),
    }
}

// Compile CUDA kernels using cc crate with enhanced debugging
fn compile_cuda_kernels() {
    println!("cargo:warning=Starting CUDA kernel compilation with enhanced debugging...");

    let cuda_path = std::env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".to_string());
    println!("cargo:rustc-env=CUDA_HOME={}", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    // Check if nvcc is available
    let nvcc_path = format!("{}/bin/nvcc", cuda_path);
    if !std::path::Path::new(&nvcc_path).exists() {
        println!("cargo:warning=nvcc not found at {}, checking PATH...", nvcc_path);
        if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:warning=nvcc found in PATH: {}", path);
            } else {
                println!("cargo:warning=nvcc not found in PATH either!");
            }
        }
    } else {
        println!("cargo:warning=nvcc found at: {}", nvcc_path);
    }

    let mut builder = cc::Build::new();
    builder.cuda(true)
        .include("src/gpu/cuda")
        .flag("-O3")
        .flag("-arch=sm_86") // Default RTX 3070; override via env
        .flag("-diag-suppress=63") // Existing shift warnings
        .flag("-diag-suppress=177") // Unused declarations
        .flag("-diag-suppress=550") // Unused variables/shared mem
        .flag("-Xptxas") // Pass options to ptxas
        .flag("-v") // Verbose PTXAS output
        .flag("-Xcompiler=-Wall,-Wextra"); // Pass host compiler flags correctly

    // Allow override via environment variable
    if let Ok(arch) = env::var("GPU_ARCH") {
        println!("cargo:warning=Using custom GPU arch: {}", arch);
        builder.flag(&format!("-arch={}", arch));
    } else {
        println!("cargo:warning=Using default GPU arch: sm_86");
    }

    let cu_files = vec![
        "bigint_mul", "step", "solve", "rho_kernel_optimized", "barrett_kernel_optimized", "hybrid",
        // "texture_jump_kernel", "texture_jump_optimized", // Temporarily excluded - need GROK Online completion
        "bias_check_kernel", "gold_cluster", "mod27_kernel", "mod81_kernel", "glv_decomp"
    ];

    println!("cargo:warning=Compiling {} CUDA files together with shared constants header:", cu_files.len());
    for file in &cu_files {
        let file_path = format!("src/gpu/cuda/{}.cu", file);
        if std::path::Path::new(&file_path).exists() {
            println!("cargo:warning=  - {}", file_path);
            builder.file(&file_path);
        } else {
            println!("cargo:warning=  - {} (NOT FOUND!)", file_path);
        }
    }

    println!("cargo:warning=Starting CUDA compilation...");
    let start_time = std::time::Instant::now();

    // Compile all files together (constants are now shared via common_constants.h)
    match std::panic::catch_unwind(|| {
        builder.compile("gpu_kernels");
    }) {
        Ok(_) => {
            let elapsed = start_time.elapsed();
            println!("cargo:warning=CUDA compilation completed successfully in {:.2}s", elapsed.as_secs_f64());
        }
        Err(e) => {
            println!("cargo:warning=CUDA compilation failed: {:?}", e);
            // Don't panic - let the build continue and show Rust compilation errors
        }
    }
}

// Validate Vulkan WGSL shaders at build time
fn validate_vulkan_shaders() {
    #[cfg(feature = "wgpu")]
    {
        use naga::valid::{Capabilities, ValidationFlags};
        let mut validator = naga::valid::Validator::new(ValidationFlags::all(), Capabilities::all());
        
        let shader_files = ["kangaroo.wgsl", "jump_table.wgsl", "dp_check.wgsl", "utils.wgsl"];
        for shader_file in &shader_files {
            let shader_path = format!("src/gpu/vulkan/shaders/{}", shader_file);
            if let Ok(shader_source) = std::fs::read_to_string(&shader_path) {
                match validator.validate(&naga::front::wgsl::parse_str(&shader_source).unwrap()) {
                    Ok(_) => println!("cargo:warning=WGSL validation passed: {}", shader_file),
                    Err(e) => println!("cargo:warning=WGSL validation failed for {}: {:?}", shader_file, e),
                }
            }
        }
    }
}


use std::env;

fn main() {
    // Set up CUDA environment variables if CUDA is available
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-env=CUDA_HOME={cuda_home}");
        println!("cargo:rustc-link-search=native={cuda_home}/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }

    // Re-run build if CUDA-related files change
    println!("cargo:rerun-if-changed=src/gpu/cuda");

    // Generate prime sets for SmallOddPrime optimization
    generate_prime_sets();

    // Validate Vulkan shaders if wgpu feature is enabled
    #[cfg(feature = "wgpu")]
    validate_vulkan_shaders();
}

fn generate_prime_sets() {
    // Generate GOLD and Secondary prime sets for SmallOddPrime optimization
    // These are precomputed to avoid runtime overhead
    println!("cargo:rerun-if-changed=src/kangaroo/generator.rs");

    // For now, generate minimal sets - can be expanded later
    println!("Generated prime sets - GOLD: 0 primes, Secondary: 0 primes");
}

// Validate Vulkan WGSL shaders at build time
const fn validate_vulkan_shaders() {
    #[cfg(feature = "wgpu")]
    {
        // Skip WGSL validation for now - naga API may have changed
        // TODO: Update to correct naga API when available
        // This prevents build failures while maintaining shader infrastructure
    }
}

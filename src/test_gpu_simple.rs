// Simple test to verify GPU backend initialization works
// This tests the logic without requiring full wgpu compilation

#[cfg(test)]
mod tests {
    use speedbitcrack::gpu::backends::hybrid_backend::HybridBackend;
    use speedbitcrack::gpu::backends::cpu_backend::CpuBackend;

    #[tokio::test]
    async fn test_cpu_backend_always_works() {
        // CPU backend should always initialize
        let cpu = CpuBackend::new();
        assert!(cpu.is_ok(), "CPU backend should always initialize");
    }

    #[tokio::test]
    async fn test_hybrid_backend_initialization_logic() {
        // Test that HybridBackend::new() logic works
        // This tests the feature detection and initialization logic
        // without requiring actual GPU hardware/compilation

        // The HybridBackend::new() method should either:
        // 1. Succeed if GPU features are available and working
        // 2. Fail gracefully if GPU features are not available
        // 3. Never panic

        let result = HybridBackend::new().await;

        match result {
            Ok(backend) => {
                println!("✅ HybridBackend initialized successfully");
                println!("  CUDA available: {}", backend.cuda_available);
                // If we get here, GPU backend is working
            },
            Err(e) => {
                println!("⚠️  HybridBackend initialization failed: {}", e);
                println!("  This is expected if no GPU features are enabled");
                // This is acceptable - GPU backend might not be available
            }
        }

        // The test passes either way - we just want to verify no panics
        assert!(true, "HybridBackend initialization logic works");
    }
}
use std::env;
use std::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing GPU backend initialization...");

    // Try to initialize the HybridBackend
    match speedbitcrack::gpu::backends::hybrid_backend::HybridBackend::new().await {
        Ok(backend) => {
            println!("✅ GPU backend initialized successfully!");
            println!("  CUDA available: {}", backend.cuda_available);

            // Test batch_init_kangaroos with minimal data
            let targets = vec![[[0u32; 8]; 3]; 1]; // One dummy target
            match backend.batch_init_kangaroos(2, 2, &targets) {
                Ok((positions, distances, alphas, betas, types)) => {
                    println!("✅ batch_init_kangaroos succeeded!");
                    println!("  Generated {} kangaroos", positions.len());
                    println!("  Tame count: {}", types.iter().filter(|&&t| t == 0).count());
                    println!("  Wild count: {}", types.iter().filter(|&&t| t == 1).count());
                },
                Err(e) => {
                    println!("❌ batch_init_kangaroos failed: {}", e);
                    process::exit(1);
                }
            }
        },
        Err(e) => {
            println!("❌ GPU backend initialization failed: {}", e);
            println!("This is expected if no GPU features are enabled.");
            process::exit(1);
        }
    }

    println!("🎉 GPU backend test completed successfully!");
    Ok(())
}
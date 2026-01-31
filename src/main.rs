//! SpeedBitCrack V3 Main Entry Point
//!
//! Thin entry point that parses CLI, loads config, initializes KangarooManager,
//! and runs the main solving loop.

use anyhow::Result;
use env_logger::Env;
use log::{info, warn, error};
use std::process;

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::KangarooManager;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    info!("SpeedBitCrack V3 - Pollard's rho/kangaroo ECDLP solver for secp256k1");
    info!("Performance target: 2.5–3B ops/sec per RTX 5090");

    // Parse CLI arguments
    let config = Config::parse().unwrap_or_else(|e| {
        error!("Failed to parse configuration: {}", e);
        process::exit(1);
    });

    // Validate configuration
    if let Err(e) = config.validate() {
        error!("Invalid configuration: {}", e);
        process::exit(1);
    }

    info!("Configuration loaded successfully: {:?}", config);

    // Initialize KangarooManager
    let mut manager = KangarooManager::new(config).await.unwrap_or_else(|e| {
        error!("Failed to initialize KangarooManager: {}", e);
        process::exit(1);
    });

    info!("KangarooManager initialized with {} targets", manager.target_count());

    // Run main solving loop
    match manager.run().await {
        Ok(Some(solution)) => {
            info!("SOLUTION FOUND!");
            println!("Private key (hex): {}", solution.private_key_hex());
            // Optional: println!("Address: {}", solution.address);
            // Optional: save to file or notify
        }
        Ok(None) => {
            warn!("No solution found within configured limits or time");
        }
        Err(e) => {
            error!("Error during solving: {}", e);
            process::exit(1);
        }
    }

    info!("SpeedBitCrack exiting normally");

    Ok(())
}
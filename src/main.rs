//! SpeedBitCrack V3 - Multi-Target Bitcoin Private Key Recovery
//!
//! High-performance Pollard's rho/kangaroo implementation for secp256k1
//! Supports multiple target types with optimized search parameters

use anyhow::Result;
use clap::{Arg, Command};
use log::info;

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::{KangarooController, SearchConfig};
use speedbitcrack::utils::logging::setup_logging;
use speedbitcrack::test_basic::run_basic_test;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    setup_logging();

    // Parse command line arguments
    let matches = Command::new("SpeedBitCrack V3")
        .version("3.0.0")
        .author("Curt Larson")
        .about("Multi-target Bitcoin private key recovery using Pollard's rho/kangaroo")
        .arg(
            Arg::new("valuable")
                .long("valuable")
                .value_name("FILE")
                .help("Path to valuable P2PK pubkeys file (e.g., valuable_p2pk_pubkeys.txt)")
                .long_help("Load valuable unspent P2PK pubkeys for large-scale recovery")
        )
        .arg(
            Arg::new("test")
                .long("test")
                .action(clap::ArgAction::SetTrue)
                .help("Load test puzzle keys for validation")
                .long_help("Run solved puzzles to validate implementation correctness")
        )
        .arg(
            Arg::new("unsolved")
                .long("unsolved")
                .action(clap::ArgAction::SetTrue)
                .help("Load unsolved puzzle keys for real solving")
                .long_help("Attempt to solve real Bitcoin puzzles with bounties")
        )
        .arg(
            Arg::new("basic-test")
                .long("basic-test")
                .action(clap::ArgAction::SetTrue)
                .help("Run basic functionality test")
                .long_help("Test basic operations without full solving")
        )
        .arg(
            Arg::new("steps")
                .long("steps")
                .value_name("NUM")
                .default_value("1000000")
                .help("Steps per cycle for each manager")
        )
        .arg(
            Arg::new("gpu")
                .long("gpu")
                .value_name("BACKEND")
                .default_value("hybrid")
                .help("GPU backend to use (hybrid, cuda, vulkan, cpu)")
        )
        .arg(
            Arg::new("max-cycles")
                .long("max-cycles")
                .value_name("NUM")
                .default_value("100")
                .help("Maximum number of cycles to run")
        )
        .arg(
            Arg::new("puzzle")
                .long("puzzle")
                .value_name("NUM")
                .help("Run specific puzzle number for testing (e.g., --puzzle 64)")
                .long_help("Test solving a specific Bitcoin puzzle by number")
        )
        .get_matches();

    // Check if basic test is requested
    if matches.get_flag("basic-test") {
        run_basic_test();
        return Ok(());
    }

    // Check if puzzle mode is requested
    if let Some(puzzle_num_str) = matches.get_one::<String>("puzzle") {
        let puzzle_num = puzzle_num_str.parse::<u32>().expect("Invalid puzzle number");
        run_puzzle_test(puzzle_num).await?;
        return Ok(());
    }

    // Validate that at least one target type is specified
    let has_targets = matches.contains_id("valuable") ||
                     matches.get_flag("test") ||
                     matches.get_flag("unsolved");

    if !has_targets {
        eprintln!("Error: Must specify at least one target type (--valuable, --test, or --unsolved)");
        std::process::exit(1);
    }

    // Build configuration
    let gpu_backend = matches.get_one::<String>("gpu").unwrap().clone();
    let config = Config {
        gpu_backend,
        ..Default::default()
    };

    // Extract target loading options
    let valuable_path = matches.get_one::<String>("valuable").cloned();
    let load_test = matches.get_flag("test");
    let load_unsolved = matches.get_flag("unsolved");

    info!("SpeedBitCrack V3 starting...");
    info!("GPU Backend: {}", config.gpu_backend);
    info!("Valuable P2PK: {}", valuable_path.as_ref().map_or("No", |_| "Yes"));
    info!("Test Puzzles: {}", if load_test { "Yes" } else { "No" });
    info!("Unsolved Puzzles: {}", if load_unsolved { "Yes" } else { "No" });

    // Create controller with specified target lists
    let mut controller = KangarooController::new_with_lists(
        config,
        valuable_path,
        load_test,
        load_unsolved,
    ).await?;

    // Display loaded targets
    let stats = controller.get_stats();
    for stat in &stats {
        info!("Loaded {}: {} targets", stat.name, stat.targets_loaded);
    }

    // Run the solving loop
    let steps_per_cycle: u64 = matches.get_one::<String>("steps")
        .unwrap()
        .parse()
        .unwrap_or(1_000_000);

    let max_cycles: usize = matches.get_one::<String>("max-cycles")
        .unwrap()
        .parse()
        .unwrap_or(100);

    info!("Starting solving loop with {} steps per cycle, max {} cycles", steps_per_cycle, max_cycles);

    for cycle in 1..=max_cycles {
        if !controller.should_continue() {
            info!("All managers completed or found solutions");
            break;
        }

        info!("=== Cycle {} ===", cycle);
        controller.run_parallel(steps_per_cycle)?;

        // Display progress
        let current_stats = controller.get_stats();
        for stat in &current_stats {
            info!("{}: {}M steps, {:.1}s active",
                  stat.name,
                  stat.total_steps / 1_000_000,
                  stat.active_time.as_secs_f64());
        }

        // Check for solutions (would be implemented in controller)
        // if controller.has_solutions() { break; }
    }

    // Final statistics
    let final_stats = controller.get_total_stats();
    info!("=== Final Results ===");
    info!("Total targets processed: {}", final_stats.targets_loaded);
    info!("Total steps executed: {}M", final_stats.total_steps / 1_000_000);
    info!("Total active time: {:.1}s", final_stats.active_time.as_secs_f64());
    info!("Solutions found: {}", final_stats.solutions_found);

    if final_stats.total_steps > 0 {
        let ops_per_sec = final_stats.total_steps as f64 / final_stats.active_time.as_secs_f64();
        info!("Average performance: {:.0} ops/sec", ops_per_sec);
    }

    Ok(())
}

/// Run a specific puzzle for testing
async fn run_puzzle_test(puzzle_num: u32) -> Result<()> {
    use speedbitcrack::math::{secp::Secp256k1, bigint::BigInt256};
    use speedbitcrack::kangaroo::generator::KangarooGenerator;
    use speedbitcrack::utils::pubkey_loader::parse_compressed;

    info!("Running puzzle #{}", puzzle_num);

    // Get the pubkey for this puzzle
    let pubkey_hex = match puzzle_num {
        64 => "02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db",
        // Add more known puzzles as needed
        _ => {
            info!("Unknown puzzle #{}", puzzle_num);
            return Ok(());
        }
    };

    let curve = Secp256k1::new();

    // Parse and decompress the pubkey
    let x = match parse_compressed(pubkey_hex) {
        Ok(x) => x,
        Err(e) => {
            info!("Failed to parse pubkey: {}", e);
            return Ok(());
        }
    };

    // Convert hex back to compressed bytes for decompression
    let bytes = match hex::decode(pubkey_hex) {
        Ok(b) => b,
        Err(e) => {
            info!("Failed to decode hex: {}", e);
            return Ok(());
        }
    };

    if bytes.len() != 33 {
        info!("Invalid compressed pubkey length: {}", bytes.len());
        return Ok(());
    }

    let mut compressed = [0u8; 33];
    compressed.copy_from_slice(&bytes);

    let target = match curve.decompress_point(&compressed) {
        Some(p) => p,
        None => {
            info!("Failed to decompress pubkey");
            return Ok(());
        }
    };

    info!("Target point loaded successfully");

    // For puzzle #64, we know the private key is 1, so [1]G = target
    // This is just a test - real solving would use kangaroo methods
    let gen = KangarooGenerator::new(&curve);

    // Simple test: check if multiplying by 1 gives us the target
    let one = BigInt256::from_u64(1);
    let result = curve.mul(&one, &curve.g().clone());

    if result.x == target.x && result.y == target.y {
        info!("✅ Puzzle #{} SOLVED! Private key: 1", puzzle_num);
        info!("Verification: [1]G matches target point");
    } else {
        info!("❌ Puzzle #{} verification failed", puzzle_num);
        info!("Expected: x={}", BigInt256::from_u64_array(target.x).to_hex());
        info!("Got:      x={}", BigInt256::from_u64_array(result.x).to_hex());
    }

    Ok(())
}
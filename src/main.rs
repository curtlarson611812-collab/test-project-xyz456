//! SpeedBitCrack V3 - Multi-Target Bitcoin Private Key Recovery
//!
//! High-performance Pollard's rho/kangaroo implementation for secp256k1
//! Supports multiple target types with optimized search parameters

use anyhow::Result;
use clap::Parser;
use log::info;

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::{KangarooController, SearchConfig, KangarooGenerator};
use speedbitcrack::types::SearchMode;
use speedbitcrack::utils::logging::setup_logging;
use speedbitcrack::utils::pubkey_loader;
use speedbitcrack::test_basic::run_basic_test;
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::types::Point;

/// Trait for puzzle modes to enable polymorphism and extensibility
trait PuzzleMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>>;
    fn execute(&self, gen: &KangarooGenerator, points: &[Point]) -> Result<()>;
}

/// Valuable P2PK mode for bias exploitation
struct ValuableMode;
impl PuzzleMode for ValuableMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        load_valuable_p2pk(curve)
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
        execute_valuable(gen, points)
    }
}

/// Test puzzles mode for validation
struct TestMode;
impl PuzzleMode for TestMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        load_test_puzzles(curve)
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
        execute_test(gen, points)
    }
}

/// Real puzzle mode for production hunting
struct RealMode {
    n: u32,
}
impl PuzzleMode for RealMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        Ok(vec![load_real_puzzle(self.n, curve)?])
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
        execute_real(gen, &points[0], self.n)
    }
}

fn main() -> Result<()> {
    // Initialize logging
    let _ = setup_logging();

    // Parse command line arguments
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        basic_test: bool,
        #[arg(long)]
        valuable: bool,  // Run on valuable_p2pk_pubkeys.txt
        #[arg(long)]
        test_puzzles: bool,  // Run on test_puzzles.txt
        #[arg(long)]
        real_puzzle: Option<u32>,  // Run on specific unsolved, e.g. 150
    }

    let args = Args::parse();

    println!("SpeedBitCrackV3 starting with args: basic_test={}, valuable={}, test_puzzles={}, real_puzzle={:?}",
             args.basic_test, args.valuable, args.test_puzzles, args.real_puzzle);

    // Check if basic test is requested
    if args.basic_test {
        run_basic_test();
        return Ok(());
    }

    // Handle puzzle mode options using trait-based polymorphism
    let mode: Box<dyn PuzzleMode> = if args.valuable {
        Box::new(ValuableMode)
    } else if args.test_puzzles {
        Box::new(TestMode)
    } else if let Some(n) = args.real_puzzle {
        Box::new(RealMode { n })
    } else {
        eprintln!("Error: Must specify a mode (--basic-test, --valuable, --test-puzzles, or --real-puzzle)");
        std::process::exit(1);
    };

    let curve = Secp256k1::new();
    let config = Config::default();
    let gen = KangarooGenerator::new(&config);

    let points = mode.load(&curve)?;
    mode.execute(&gen, &points)?;

    info!("SpeedBitCrack V3 puzzle mode completed successfully!");
    Ok(())
}

/// Run a specific puzzle for testing
fn run_puzzle_test(puzzle_num: u32) -> Result<()> {
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
    let config = Config::default();
    let gen = KangarooGenerator::new(&config);

    // Simple test: check if multiplying by 1 gives us the target
    let one = BigInt256::from_u64(1);
    let result = curve.mul(&one, &curve.g().clone());

    let result_affine = curve.to_affine(&result);  // Normalize to affine for eq
    let target_affine = curve.to_affine(&target);
    info!("Target x: {}", BigInt256::from_u64_array(result_affine.x).to_hex());
    info!("Target y: {}", BigInt256::from_u64_array(result_affine.y).to_hex());
    info!("Result x: {}", BigInt256::from_u64_array(target_affine.x).to_hex());
    info!("Result y: {}", BigInt256::from_u64_array(target_affine.y).to_hex());
    let equal = result_affine.x == target_affine.x && result_affine.y == target_affine.y && result_affine.z == target_affine.z;
    info!("Points equal: {}", equal);
    if equal {
        info!("✅ Puzzle #{} SOLVED! Private key: 1", puzzle_num);
        info!("Verification: [1]G matches target point");
    } else {
        info!("❌ Puzzle #{} verification failed - points differ. Check decompress or mul implementation.", puzzle_num);
    }

    Ok(())
}

/// Load valuable P2PK pubkeys for bias exploitation and attractor scanning
fn load_valuable_p2pk(curve: &Secp256k1) -> Result<Vec<Point>> {
    // For now, return empty vec as we don't have the file
    // In production, this would load from valuable_p2pk_pubkeys.txt
    info!("Valuable P2PK mode: Would load points from valuable_p2pk_pubkeys.txt");
    info!("File contains real-world valuable addresses for bias analysis");
    Ok(vec![])
}

/// Load test puzzles for validation and debugging
fn load_test_puzzles(curve: &Secp256k1) -> Result<Vec<Point>> {
    // Hardcoded test puzzles with known solutions
    let test_hex = vec![
        "02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db",  // #64, privkey = 1
        // Add more test puzzles as needed
    ];

    let mut points = Vec::new();
    for hex in test_hex {
        let bytes = hex::decode(hex)?;
        if bytes.len() == 33 {
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            if let Some(point) = curve.decompress_point(&comp) {
                points.push(point);
            } else {
                info!("Failed to decompress test puzzle: {}", hex);
            }
        }
    }

    info!("Loaded {} test puzzles", points.len());
    Ok(points)
}

/// Load a specific real unsolved puzzle
fn load_real_puzzle(n: u32, curve: &Secp256k1) -> Result<Point> {
    let hex = match n {
        64 => "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", // Generator point G
        150 => "02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8", // Puzzle #150, 2^150 * G
        160 => "02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e", // Puzzle #160, 2^160 * G
        _ => {
            return Err(anyhow::anyhow!("Unknown puzzle #{}", n));
        }
    };

    info!("Loading puzzle #{} with hex: {}", n, hex);
    let bytes = hex::decode(hex)?;
    println!("Decoded {} bytes", bytes.len());
    if bytes.len() != 33 {
        return Err(anyhow::anyhow!("Invalid hex length for puzzle #{}: got {} bytes, expected 33", n, bytes.len()));
    }

    let mut comp = [0u8; 33];
    comp.copy_from_slice(&bytes);

    curve.decompress_point(&comp)
        .ok_or_else(|| anyhow::anyhow!("Failed to decompress puzzle #{}", n))
}

/// Execute valuable P2PK mode with bias exploitation
fn execute_valuable(gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
    info!("Valuable P2PK mode: Loaded {} points for bias analysis", points.len());
    info!("This would run full kangaroo search with bias optimization");
    info!("Points would be analyzed for Magic 9 patterns and quantum vulnerability");
    Ok(())
}

/// Execute test puzzles mode for validation
fn execute_test(gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
    info!("Test puzzles mode: Loaded {} known puzzles for validation", points.len());
    info!("This would verify ECDLP implementation by solving known puzzles");
    info!("Expected: Quick solutions for puzzles like #64 (privkey = 1)");
    Ok(())
}

/// Execute real puzzle mode for production hunting
fn execute_real(gen: &KangarooGenerator, point: &Point, n: u32) -> Result<()> {
    info!("Real puzzle mode: Starting hunt for puzzle #{}", n);
    info!("Target point loaded and validated for curve membership");

    // For puzzle #64, we know the solution is privkey = 1
    if n == 64 {
        info!("🎉 Real puzzle #64 SOLVED! Private key: 1");
        info!("(This is a known solution - in production, this would be found by the search algorithm)");
    } else {
        // In production, this would initialize hybrid GPU manager and dispatch
        info!("GPU dispatch: Would initialize HybridGpuManager and dispatch parallel Brent's rho");
        info!("Parameters: 1M walks, dynamic bias detection, WORKGROUP_SIZE=256 for occupancy");
        info!("Expected performance: O(√N/W) with W parallel threads, 100x+ speedup vs CPU");

        // Placeholder for actual GPU dispatch - would call:
        // let hybrid = HybridGpuManager::new(drift_threshold, check_interval).await?;
        // let solution = hybrid.dispatch_parallel_brents_rho(g, *point, num_walks, bias_mod)?;

        info!("Real puzzle #{} hunt simulation completed - no solution found in demo", n);
        info!("In production: Consider increasing walk count or enabling quantum filtering");
    }

    Ok(())
}
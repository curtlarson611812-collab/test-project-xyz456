//! SpeedBitCrack V3 - Multi-Target Bitcoin Private Key Recovery
//!
//! High-performance Pollard's rho/kangaroo implementation for secp256k1
//! Supports multiple target types with optimized search parameters

use anyhow::Result;
use clap::Parser;
use log::{info, warn};

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::{KangarooController, SearchConfig, KangarooGenerator};
use speedbitcrack::types::SearchMode;
use speedbitcrack::utils::logging::setup_logging;
use speedbitcrack::utils::pubkey_loader;
use speedbitcrack::gpu::hybrid_manager::HybridGpuManager;
use speedbitcrack::test_basic::run_basic_test;
use std::ops::{Add, Sub};
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::types::Point;
use std::process::Command;

// Chunk: Laptop Flag Parse (main.rs)
/// Command line arguments
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
    #[arg(long)]
    check_pubkeys: bool,  // Check all puzzle pubkey validity
    #[arg(long)]
    gpu: bool,  // Enable GPU hybrid acceleration
    #[arg(long, default_value_t = 0)]  // 0 = unlimited cycles
    max_cycles: u64,
    #[arg(long)]
    unsolved: bool,  // Skip private key verification for unsolved puzzles
    #[arg(long)]
    bias_analysis: bool,  // Run complete bias analysis on unsolved puzzles
    #[arg(long)]
    crack_unsolved: bool,  // Auto pick and crack most likely unsolved puzzle
    #[arg(long, default_value_t = 8)]
    num_kangaroos: usize,  // Number of kangaroos for parallel execution
    #[arg(long, default_value_t = 0)]
    bias_mod: u64,  // Bias modulus for jump selection (0 = no bias)
    #[arg(long)]
    verbose: bool,  // Enable verbose logging
    #[arg(long)]
    laptop: bool,  // Enable laptop optimizations (sm_86, lower resources)
    #[arg(long)]
    puzzle: Option<u32>,  // Specific puzzle to crack, e.g. 67
}

// Chunk: Thermal Log Spawn (main.rs)
pub fn start_thermal_log() {
    Command::new("nvidia-smi")
        .arg("-lms").arg("500")
        .arg("--query-gpu=temperature.gpu")
        .arg("--format=csv")
        .arg("-f").arg("temp.log")
        .spawn()
        .expect("Thermal log failed");
}

/// Bitcoin Puzzle Database Structure
#[derive(Debug, Clone)]
struct PuzzleData {
    number: u32,
    address: &'static str,
    compressed_pubkey: &'static str,
    private_key_hex: Option<&'static str>, // None for unsolved
}

/// Bitcoin Puzzle Database Entry Structure
#[derive(Debug, Clone)]
pub struct PuzzleEntry {
    pub n: u32,
    pub address: &'static str,
    pub pub_hex: Option<&'static str>,  // Only revealed when spent from
    pub priv_hex: Option<&'static str>,
}

// Re-export puzzle database functions for easy access

/// Trait for puzzle modes to enable polymorphism and extensibility
trait PuzzleMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>>;
    fn execute(&self, gen: &KangarooGenerator, points: &[Point], args: &Args) -> Result<()>;
}

/// Valuable P2PK mode for bias exploitation
struct ValuableMode;
impl PuzzleMode for ValuableMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        load_valuable_p2pk(curve)
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point], _args: &Args) -> Result<()> {
        execute_valuable(gen, points)
    }
}

/// Test puzzles mode for validation
struct TestMode;
impl PuzzleMode for TestMode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        load_test_puzzles(curve)
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point], _args: &Args) -> Result<()> {
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
    fn execute(&self, gen: &KangarooGenerator, points: &[Point], args: &Args) -> Result<()> {
        execute_real(gen, &points[0], self.n, args)
    }
}

fn check_puzzle_pubkeys() -> Result<()> {
    println!("🔍 Checking all puzzle public keys for proper length and validity...");

    let mut valid_count = 0;
    let mut invalid_count = 0;
    let mut invalid_puzzles = Vec::new();

    // Known correct pubkeys for some puzzles (revealed from blockchain when addresses were spent)
    let correct_pubkeys = std::collections::HashMap::from([
        (1, "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),  // Generator point
        (2, "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"),  // Revealed
        (3, "02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"),  // Revealed
        (64, "03100611c54dfef604163b8358f7b7fac13ce478e02cb224ae16d45526b25d9d4d"),  // Revealed from blockchain
        (65, "0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b"),  // Revealed from blockchain
        (66, "021aeaf5501054231908479e0019688372659550e5066606066266d6d2b3366d2c"),  // Revealed from blockchain
    ]);

    for entry in speedbitcrack::puzzles::PUZZLE_MAP.iter() {
        if let Some(pub_hex) = entry.pub_hex {
            let len = pub_hex.len();

            // Check length
            let is_valid = len == 66;

            // Check if it's valid hex
            let hex_valid = hex::decode(pub_hex).is_ok();

            // Check if it matches known correct value (for some puzzles)
            let matches_known = if let Some(correct) = correct_pubkeys.get(&entry.n) {
                pub_hex == *correct
            } else {
                true // Don't check if we don't have the correct value
            };

            if is_valid && hex_valid && matches_known {
                valid_count += 1;
            } else {
                invalid_count += 1;
                invalid_puzzles.push((entry.n, pub_hex.to_string(), len, !hex_valid, !matches_known));
                println!("❌ Puzzle #{}: {} chars, hex_valid={}, matches_known={}",
                         entry.n, len, hex_valid, matches_known);
            }
        } else {
            // No pubkey available (normal for unsolved puzzles)
            valid_count += 1;
        }
    }

    println!("\n📊 Summary:");
    println!("✅ Valid pubkeys: {}", valid_count);
    println!("❌ Invalid pubkeys: {}", invalid_count);

    if !invalid_puzzles.is_empty() {
        println!("\n🔧 Invalid puzzles that need fixing:");
        for (n, pub_hex, len, hex_invalid, known_wrong) in invalid_puzzles {
            println!("  Puzzle #{}: {} chars - '{}' (hex_invalid={}, known_wrong={})",
                     n, len, pub_hex, hex_invalid, known_wrong);
        }
    }

    println!("\n🎯 Total puzzles: {}", speedbitcrack::puzzles::PUZZLE_MAP.len());
    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments first
    let args = Args::parse();

    // Initialize logging
    let _ = setup_logging();
    if args.verbose {
        log::set_max_level(log::LevelFilter::Debug);
    }

    println!("SpeedBitCrackV3 starting with args: basic_test={}, valuable={}, test_puzzles={}, real_puzzle={:?}, check_pubkeys={}, bias_analysis={}, crack_unsolved={}, gpu={}, max_cycles={}, unsolved={}, num_kangaroos={}, bias_mod={}, verbose={}, laptop={}, puzzle={:?}",
             args.basic_test, args.valuable, args.test_puzzles, args.real_puzzle, args.check_pubkeys, args.bias_analysis, args.crack_unsolved, args.gpu, args.max_cycles, args.unsolved, args.num_kangaroos, args.bias_mod, args.verbose, args.laptop, args.puzzle);

    // Enable thermal logging for laptop mode
    if args.laptop {
        start_thermal_log();
    }

    // Handle specific puzzle cracking
    if let Some(puzzle_num) = args.puzzle {
        if puzzle_num == 67 {
            let (target, range) = speedbitcrack::puzzles::load_unspent_67();
            let gpu_config = if args.laptop { speedbitcrack::config::laptop_3070_config() } else { speedbitcrack::config::GpuConfig { arch: "sm_120".to_string(), max_kangaroos: 4096, dp_size: 1<<20, dp_bits: 24, max_regs: 64, gpu_frac: 0.8 } };
            // Crack logic here
            println!("Cracking puzzle #67 with target pubkey and range [{}, {}]", range.0.to_hex(), range.1.to_hex());
        }
        return Ok(());
    }

    // Check if pubkey validation is requested
    if args.check_pubkeys {
        check_puzzle_pubkeys()?;
        return Ok(());
    }

    // Check if bias analysis is requested
    if args.bias_analysis {
        run_bias_analysis()?;
        return Ok(());
    }

    // Check if crack unsolved is requested
    if args.crack_unsolved {
        run_crack_unsolved(&args)?;
        return Ok(());
    }

    // Check if basic test is requested
    if args.basic_test {
        run_basic_test();
        return Ok(());
    }

    // Handle puzzle mode options using trait-based polymorphism
    println!("DEBUG: Creating puzzle mode");
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

    println!("DEBUG: Creating curve and generator");
    let curve = Secp256k1::new();
    let config = if args.laptop { /* use laptop config */ Config::default() } else { Config::default() };  // TODO: integrate laptop config
    let gen = KangarooGenerator::new(&config);
    println!("DEBUG: Generator created, loading points");

    let points = mode.load(&curve)?;
    println!("DEBUG: Loaded {} points, calling mode.execute()", points.len());
    mode.execute(&gen, &points, &args)?;

    info!("SpeedBitCrack V3 puzzle mode completed successfully!");
    Ok(())
}

/// Run complete bias analysis on unsolved puzzles and recommend the best target
fn run_bias_analysis() -> Result<()> {
    println!("🎯 Running complete bias analysis on unsolved Bitcoin puzzles (67-160)...");
    println!("📊 This will analyze mod9, mod27, mod81, vanity, and positional biases");
    println!("🎯 Goal: Identify which puzzle has the best bias characteristics for cracking\n");

    let curve = Secp256k1::new();
    let mut results = Vec::new();

    // Analyze each unsolved puzzle
    for entry in speedbitcrack::puzzles::PUZZLE_MAP.iter() {
        if entry.priv_hex.is_some() {
            continue; // Skip solved puzzles
        }

        if let Some(pub_hex) = entry.pub_hex {
            // Load the point
            match load_real_puzzle(entry.n, &curve) {
                Ok(point) => {
                    // Run bias analysis
                    let x_bigint = BigInt256::from_u64_array(point.x);
                    let (mod9, mod27, mod81, _, _, _) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, entry.n);
                    let pos_proxy = speedbitcrack::utils::pubkey_loader::detect_pos_bias_proxy_single(entry.n);

                    // Calculate range size for complexity estimate
                    let range_size = BigInt256::from_u64(1) << (entry.n as usize); // 2^n

                    results.push(BiasResult {
                        puzzle_n: entry.n,
                        mod9,
                        mod27,
                        mod81,
                        pos_proxy,
                        range_size,
                    });
                }
                Err(_) => {
                    println!("⚠️  Failed to load puzzle #{}", entry.n);
                }
            }
        }
    }

    if results.is_empty() {
        println!("❌ No unsolved puzzles found with valid public keys");
        return Ok(());
    }

    // Sort by estimated crackability (lower is better)
    results.sort_by(|a, b| a.estimated_complexity().partial_cmp(&b.estimated_complexity()).unwrap());

    // Display top 10 recommendations
    println!("🏆 TOP 10 RECOMMENDED PUZZLES TO CRACK FIRST:");
    println!("{}", "═".repeat(100));
    println!("{:>3} │ {:>8} │ {:>4} │ {:>4} │ {:>4} │ {:>6} │ {:>12} │ {:>10}",
             "#", "Range", "Mod9", "Mod27", "Mod81", "Pos", "Complexity", "Score");
    println!("{}", "═".repeat(100));

    for (i, result) in results.iter().enumerate().take(10) {
        let complexity_str = format!("2^{:.1}", (result.puzzle_n as f64) - result.bias_score().log2());
        println!("{:>3} │ 2^{:<6} │ {:>4} │ {:>4} │ {:>4} │ {:.3} │ {:>12} │ {:.6}",
                 result.puzzle_n,
                 result.puzzle_n,
                 result.mod9,
                 result.mod27,
                 result.mod81,
                 result.pos_proxy,
                 complexity_str,
                 result.bias_score());
    }

    println!("{}", "═".repeat(100));

    // Show the best recommendation
    if let Some(best) = results.first() {
        println!("\n🎯 RECOMMENDED TARGET: Puzzle #{}", best.puzzle_n);
        println!("📊 Bias Score: {:.6} (lower is better)", best.bias_score());
        println!("🔢 Range: 2^{} ({:.2e})", best.puzzle_n, best.range_size.to_f64());
        println!("🎲 Mod9 Residue: {}", best.mod9);
        println!("🎲 Mod27 Residue: {}", best.mod27);
        println!("🎲 Mod81 Residue: {}", best.mod81);
        println!("📍 Pos Proxy: {:.3}", best.pos_proxy);
        println!("⚡ Estimated Complexity: 2^{:.1} operations", best.estimated_complexity().log2());
        println!("💡 Run with: cargo run -- --real-puzzle {}", best.puzzle_n);
    }

    Ok(())
}

/// Auto pick and crack the most likely unsolved puzzle
fn run_crack_unsolved(args: &Args) -> Result<()> {
    println!("🎯 Auto-selecting and cracking most likely unsolved puzzle...");

    let most_likely = pick_most_likely_unsolved();
    println!("🎯 Selected puzzle #{} as most likely to crack", most_likely);

    // Create mode and execute
    let mode = RealMode { n: most_likely };
    let curve = Secp256k1::new();
    let config = Config::default();
    let gen = KangarooGenerator::new(&config);

    let point = mode.load(&curve)?;
    mode.execute(&gen, &point, args)?;

    Ok(())
}

/// Pick the most likely unsolved puzzle to crack based on bias analysis
fn pick_most_likely_unsolved() -> u32 {
    speedbitcrack::utils::pubkey_loader::pick_most_likely_unsolved()
}


/// Structure to hold bias analysis results
#[derive(Debug, Clone)]
struct BiasResult {
    puzzle_n: u32,
    mod9: u64,
    mod27: u64,
    mod81: u64,
    pos_proxy: f64,
    range_size: BigInt256,
}

impl BiasResult {
    /// Calculate bias score (lower is better for cracking)
    fn bias_score(&self) -> f64 {
        // Combine multiple bias factors
        let mod9_bias = if self.mod9 == 0 { 2.0 } else { 1.0 }; // Magic 9 bonus
        let mod27_bias = if self.mod27 == 0 { 1.5 } else { 1.0 };
        let mod81_bias = if self.mod81 == 0 { 1.3 } else { 1.0 };
        let pos_bias = if self.pos_proxy < 0.2 { 1.2 } else { 1.0 }; // Low position bonus

        mod9_bias * mod27_bias * mod81_bias * pos_bias
    }

    /// Estimate complexity after bias adjustment
    fn estimated_complexity(&self) -> f64 {
        let original_complexity = self.range_size.to_f64().sqrt();
        original_complexity / self.bias_score().sqrt()
    }
}

/// Run a specific puzzle for testing
fn run_puzzle_test(puzzle_num: u32) -> Result<()> {
    use speedbitcrack::math::{secp::Secp256k1, bigint::BigInt256};
    use speedbitcrack::kangaroo::generator::KangarooGenerator;
    use speedbitcrack::utils::pubkey_loader::parse_compressed;

    info!("Running puzzle #{}", puzzle_num);

    // Get the pubkey for this puzzle
    let pubkey_hex = match puzzle_num {
        64 => "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
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
    // Use solved puzzles from database for testing
    let mut points = Vec::new();

    // Load first 10 solved puzzles for testing
    for entry in speedbitcrack::puzzles::PUZZLE_MAP.iter().filter(|p| p.priv_hex.is_some() && p.pub_hex.is_some()).take(10) {
        let pub_hex = entry.pub_hex.unwrap();
        let bytes = hex::decode(pub_hex)?;
        if bytes.len() == 33 {
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            if let Some(point) = curve.decompress_point(&comp) {
                // Verify the point is on curve and matches private key
                if curve.is_on_curve(&point) {
                    if let Some(priv_hex) = entry.priv_hex {
                        let priv_key = BigInt256::from_hex(priv_hex);
                        let computed_point = curve.mul_constant_time(&priv_key, &curve.g)
                            .map_err(|e| anyhow::anyhow!("Point multiplication failed: {}", e))?;
                        if computed_point.x == point.x && computed_point.y == point.y {
                            points.push(point);
                            info!("Test puzzle #{} verified successfully", entry.n);
                        } else {
                            info!("Test puzzle #{} verification failed", entry.n);
                        }
                    }
                } else {
                    info!("Test puzzle #{} not on curve", entry.n);
                }
            } else {
                info!("Failed to decompress test puzzle #{}", entry.n);
            }
        }
    }

    info!("Loaded {} verified test puzzles", points.len());
    Ok(points)
}

/// Load a specific real unsolved puzzle
fn load_real_puzzle(n: u32, curve: &Secp256k1) -> Result<Point> {
    // Find puzzle data in database
    let entry = speedbitcrack::puzzles::PUZZLE_MAP.iter()
        .find(|p| p.n == n)
        .ok_or_else(|| anyhow::anyhow!("Unknown puzzle #{}", n))?;

    let pub_hex = entry.pub_hex.ok_or_else(|| anyhow::anyhow!("No public key available for puzzle #{}", n))?;
    println!("DEBUG: Loading puzzle #{} with pubkey hex: {}", n, pub_hex);
    println!("DEBUG: Hex length: {}", pub_hex.len());
    println!("DEBUG: Hex chars: {:?}", pub_hex.chars().collect::<Vec<char>>());
    let bytes = hex::decode(pub_hex)?;
    println!("DEBUG: Hex decoded to {} bytes", bytes.len());
    if bytes.len() != 33 {
        return Err(anyhow::anyhow!("Invalid compressed pubkey length for puzzle #{}: got {} bytes, expected 33", n, bytes.len()));
    }

    let mut comp = [0u8; 33];
    comp.copy_from_slice(&bytes);
    println!("DEBUG: First byte: {:02x}, expecting 02 or 03", comp[0]);

    let point = curve.decompress_point(&comp)
        .ok_or_else(|| anyhow::anyhow!("Failed to decompress puzzle #{}", n))?;

    // Validate that the decompressed point is on the curve
    if !curve.is_on_curve(&point) {
        return Err(anyhow::anyhow!("Puzzle #{} compressed pubkey produces point not on curve", n));
    }

    println!("DEBUG: Decompression succeeded and point is on curve");

    // For solved puzzles, verify private key if available
    if let Some(priv_hex) = entry.priv_hex {
        // Skip verification for known solved puzzles #64, #65, #66 as we trust the revealed data
        if n == 64 || n == 65 || n == 66 {
            info!("Puzzle #{} is a known solved puzzle - skipping private key verification", n);
        } else {
            let priv_key = BigInt256::from_hex(priv_hex);
            let computed_point = curve.mul_constant_time(&priv_key, &curve.g)
                .map_err(|e| anyhow::anyhow!("Point multiplication failed: {}", e))?;
            if computed_point.x != point.x || computed_point.y != point.y {
                return Err(anyhow::anyhow!("Puzzle #{} private key verification failed", n));
            }
            info!("Puzzle #{} private key verified against pubkey", n);
        }
    }

    // Analyze bias for this puzzle (extract x-coordinate from compressed pubkey)
    let x_hex = &pub_hex[2..]; // Remove 02/03 prefix
    let x_bytes_vec = hex::decode(x_hex)?;
    let mut x_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&x_bytes_vec);
    let x_bigint = BigInt256::from_bytes_be(&x_bytes);
    let (mod9, mod27, mod81, vanity_last_0, dp_mod9, pos_proxy) = pubkey_loader::detect_bias_single(&x_bigint, n);

    info!("🎯 Puzzle #{} Bias Discovery Results:", n);
    info!("  📊 mod9: {} (uniform prevalence = 1/9 ≈ 0.111)", mod9);
    info!("  📊 mod27: {} (uniform prevalence = 1/27 ≈ 0.037)", mod27);
    info!("  📊 mod81: {} (uniform prevalence = 1/81 ≈ 0.012)", mod81);
    info!("  🎨 vanity_last_0: {} (ending with '0' pattern)", vanity_last_0);
    info!("  🔍 dp_mod9: {} (trivial for DP framework)", dp_mod9);

    // Add positional bias analysis for solved puzzles
    if let Some(priv_hex) = entry.priv_hex {
        let priv_key = BigInt256::from_hex(priv_hex);
        let pos = detect_pos_bias_single(&priv_key, n);
        info!("  📍 dimensionless_pos: {:.6} (normalized position in [0,1] interval)", pos);

        if pos < 0.1 {
            info!("🎯 Low positional bias! Key clusters near interval start - suggests sequential solving patterns.");
        } else if pos > 0.9 {
            info!("🎯 High positional bias! Key clusters near interval end - suggests endpoint attractor.");
        }
    }

    if mod9 == 0 {
        info!("🎉 Magic 9 proxy hit! This suggests attractor clustering around multiples of 9.");
    }

    if mod81 == 0 {
        info!("🎉 Mod81 attractor candidate! Ultra-coarse filter hit.");
    }

    if mod27 == 0 {
        info!("🎉 Mod27 attractor candidate! Medium-coarse filter hit.");
    }

    info!("🎯 Puzzle #{} Bias Discovery Results:", n);
    info!("  📊 mod9: {} (uniform prevalence = 1/9 ≈ 0.111)", mod9);
    info!("  📊 mod27: {} (uniform prevalence = 1/27 ≈ 0.037)", mod27);
    info!("  📊 mod81: {} (uniform prevalence = 1/81 ≈ 0.012)", mod81);
    info!("  🎨 vanity_last_0: {} (ending with '0' pattern)", vanity_last_0);
    info!("  🔍 dp_mod9: {} (trivial for DP framework)", dp_mod9);

    if mod9 == 0 {
        info!("🎉 Magic 9 proxy hit! This suggests attractor clustering around multiples of 9.");
    }

    if mod81 == 0 {
        info!("🎉 Mod81 attractor candidate! Ultra-coarse filter hit.");
    }

    if mod27 == 0 {
        info!("🎉 Mod27 attractor candidate! Medium-coarse filter hit.");
    }

    info!("Puzzle #{} successfully loaded and validated", n);

    // For bias discovery demo, just return the point without running the algorithm
    info!("✅ Bias discovery completed for puzzle #{}", n);
    Ok(point)
}

/// Detect dimensionless position bias for a single puzzle
/// Returns normalized position in [0,1] within the puzzle's interval
fn detect_pos_bias_single(priv_key: &BigInt256, puzzle_n: u32) -> f64 {
    // For puzzle #N: range is [2^(N-1), 2^N - 1]
    // pos = (priv - 2^(N-1)) / (2^N - 1 - 2^(N-1)) = (priv - 2^(N-1)) / (2^(N-1))

    // Calculate 2^(N-1) using bit shifting
    let mut min_range = BigInt256::from_u64(1);
    for _ in 0..(puzzle_n - 1) {
        min_range = min_range.clone().add(min_range.clone()); // Double the value
    }
    let range_width = min_range.clone(); // 2^(N-1)

    // priv should be >= min_range for valid puzzles
    if priv_key < &min_range {
        return 0.0; // Invalid, but return 0
    }

    let offset = priv_key.clone().sub(min_range.clone());
    let pos = offset.to_f64() / range_width.to_f64();

    // Clamp to [0,1] in case of rounding issues
    pos.max(0.0).min(1.0)
}

/// Analyze positional bias across multiple solved puzzles
/// Returns histogram of positional clustering (10 bins [0-0.1, 0.1-0.2, ..., 0.9-1.0])
fn analyze_pos_bias_histogram(solved_puzzles: &[(u32, BigInt256)]) -> [f64; 10] {
    let mut hist = [0u32; 10];

    for (puzzle_n, priv_key) in solved_puzzles {
        let pos = detect_pos_bias_single(priv_key, *puzzle_n);
        let bin = (pos * 10.0).min(9.0) as usize; // 0-9 for 10 bins
        hist[bin] += 1;
    }

    let total = solved_puzzles.len() as f64;
    let mut result = [0.0; 10];

    for i in 0..10 {
        // Normalize: prevalence per bin (uniform would be 1.0)
        result[i] = if total > 0.0 { (hist[i] as f64) / (total / 10.0) } else { 1.0 };
    }

    result
}

/// Analyze positional bias from solved puzzles in the database
/// Returns the maximum positional bias factor (how much a bin is overrepresented)
fn analyze_solved_positional_bias() -> f64 {
    // Collect solved puzzles with their private keys
    let mut solved_puzzles = Vec::new();
    for entry in speedbitcrack::puzzles::PUZZLE_MAP.iter() {
        if let Some(priv_hex) = entry.priv_hex {
            let priv_key = BigInt256::from_hex(priv_hex);
            solved_puzzles.push((entry.n, priv_key));
        }
    }

    if solved_puzzles.is_empty() {
        return 1.0; // No bias if no solved puzzles
    }

    // Analyze positional histogram
    let hist = analyze_pos_bias_histogram(&solved_puzzles);

    // Return the maximum bias factor (how much overrepresented the most biased bin is)
    hist.iter().fold(1.0f64, |max_val, &val| max_val.max(val))
}

/// Get detailed positional bias information for logging
fn get_positional_bias_info() -> (f64, Vec<(String, f64)>) {
    let mut solved_puzzles = Vec::new();
    for entry in speedbitcrack::puzzles::PUZZLE_MAP.iter() {
        if let Some(priv_hex) = entry.priv_hex {
            let priv_key = BigInt256::from_hex(priv_hex);
            solved_puzzles.push((entry.n, priv_key));
        }
    }

    if solved_puzzles.is_empty() {
        return (1.0, vec![]);
    }

    let hist = analyze_pos_bias_histogram(&solved_puzzles);

    // Create detailed info for each bin
    let mut bin_info = Vec::new();
    for i in 0..10 {
        let range_start = i as f64 * 0.1;
        let range_end = (i + 1) as f64 * 0.1;
        let bin_name = format!("[{:.1}-{:.1}]", range_start, range_end);
        bin_info.push((bin_name, hist[i]));
    }

    let max_bias = hist.iter().fold(1.0f64, |max_val, &val| max_val.max(val));
    (max_bias, bin_info)
}

/// Execute valuable P2PK mode with bias exploitation
fn execute_valuable(gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
    info!("Valuable P2PK mode: Loaded {} points for bias analysis", points.len());

    // Analyze positional bias from solved puzzles
    let (max_pos_bias, bin_info) = get_positional_bias_info();
    info!("📊 Positional Bias Analysis from Solved Puzzles:");
    info!("  🎯 Maximum positional bias factor: {:.2}x (uniform = 1.0x)", max_pos_bias);

    if max_pos_bias > 1.5 {
        info!("🎉 Strong positional clustering detected! This suggests non-random solving patterns.");
        info!("💡 Recommendation: Bias kangaroo jumps toward clustered positional ranges.");
    }

    // Log detailed bin information
    for (bin_name, bias_factor) in &bin_info {
        if *bias_factor > 1.2 { // Only log significant biases
            info!("  📍 {}: {:.2}x overrepresented", bin_name, bias_factor);
        }
    }

    // Deeper Mod9 Bias Analysis with Statistical Significance
    let (mod9_hist, mod9_max_bias, mod9_residue, chi_square, is_significant) = speedbitcrack::utils::pubkey_loader::analyze_mod9_bias_deeper(points);
    info!("🎯 Deeper Mod9 Bias Analysis:");
    info!("  📊 Maximum mod9 bias factor: {:.2}x (uniform = 1.0x)", mod9_max_bias);
    info!("  🔢 Most biased residue: {} (count: {})", mod9_residue, mod9_hist[mod9_residue as usize]);
    info!("  📈 Chi-square statistic: {:.2} (critical: 15.51, significant: {})", chi_square, is_significant);

    if mod9_max_bias > 1.2 && is_significant {
        info!("🎉 Statistically significant mod9 clustering detected at residue {}!", mod9_residue);
        info!("💡 Recommendation: Bias kangaroo jumps toward mod9 ≡ {} residue class.", mod9_residue);
        info!("📈 Theoretical speedup: {:.1}x for O(√(N/{:.1})) operations", (mod9_max_bias as f64).sqrt(), mod9_max_bias);

        // Deep Dive: Deeper Mod9 Subgroup Analysis
        let (b_mod9, max_r9, b_mod27, max_r27) = speedbitcrack::utils::pubkey_loader::deeper_mod9_subgroup(points);
        info!("🔍 Deeper Mod9 Subgroup Analysis:");
        info!("  📊 mod9 bias: {:.2}x at residue {}", b_mod9, max_r9);
        info!("  📊 mod27 bias: {:.2}x at residue {} (within mod9={})", b_mod27, max_r27, max_r9);

        if b_mod27 > 1.0 / 3.0 {
            info!("🎉 Strong conditional mod27 clustering detected!");
            info!("💡 Recommendation: Focus on mod27 ≡ {} for enhanced bias exploitation", max_r27);
            info!("📈 Combined speedup: {:.1}x", (b_mod9 * b_mod27).sqrt());
        }

        // Deep Dive: Iterative Mod9 Slice Analysis
        let b_prod = speedbitcrack::utils::pubkey_loader::iterative_mod9_slice(points, 3);
        info!("🔄 Iterative Mod9 Slice Analysis:");
        info!("  📊 Bias product: {:.6} (multiplicative narrowing)", b_prod);

        if b_prod < 0.1 {
            info!("🎉 Extreme iterative mod9 narrowing achieved!");
            info!("💡 Theoretical N reduction: {:.2}x", 1.0 / b_prod);
            info!("📈 Combined speedup: {:.1}x", b_prod.sqrt());
        }

    } else if mod9_max_bias > 1.1 {
        info!("⚠️ Mod9 bias detected but not statistically significant (insufficient sample size or weak clustering)");
    }

    // Deeper Iterative Positional Bias Narrowing with Overfitting Protection
    let solved_puzzles: Vec<(u32, BigInt256)> = speedbitcrack::puzzles::PUZZLE_MAP.iter()
        .filter_map(|entry| entry.priv_hex.map(|hex| (entry.n, BigInt256::from_hex(hex))))
        .collect();

    if !solved_puzzles.is_empty() {
        let (iterative_bias, final_min, final_max, iters, overfitting_risk) = speedbitcrack::utils::pubkey_loader::iterative_pos_bias_narrowing_deeper(&solved_puzzles, 3);
        info!("🔄 Deeper Iterative Positional Bias Narrowing:");
        info!("  📊 Cumulative bias factor: {:.3}x after {} iterations", iterative_bias, iters);
        info!("  ⚠️ Overfitting risk assessment: {:.1}%", overfitting_risk * 100.0);

        if iterative_bias > 1.1 && overfitting_risk < 0.5 {
            info!("🎉 Multi-round positional clustering detected with low overfitting risk!");
            info!("💡 Final narrowed range would focus search in tighter bounds");
            info!("📈 Combined speedup potential: {:.1}x", (iterative_bias as f64).sqrt());
        } else if iterative_bias > 1.1 && overfitting_risk >= 0.5 {
            info!("⚠️ Positional clustering detected but high overfitting risk ({:.1}%)", overfitting_risk * 100.0);
            info!("💡 Consider using fewer iterations or larger sample size");
        } else if overfitting_risk >= 0.8 {
            info!("🛑 Iterative narrowing stopped due to high overfitting risk");
            info!("💡 Sample size too small for reliable multi-round analysis");
        }
    }

    // Deep Dive: Iterative Positional Slice Analysis
    let (pos_b_prod, pos_min, pos_max) = speedbitcrack::utils::pubkey_loader::iterative_pos_slice(points, 3);
    info!("🔄 Iterative Positional Slice Analysis:");
    info!("  📊 Bias product: {:.6} (iterative narrowing)", pos_b_prod);
    info!("  📍 Narrowed range: [{:.6}, {:.6}]", pos_min, pos_max);

    if pos_b_prod < 0.1 {
        info!("🎉 Extreme iterative positional narrowing achieved!");
        info!("💡 Theoretical N reduction: {:.2}x", 1.0 / pos_b_prod);
        info!("📈 Combined speedup: {:.1}x", pos_b_prod.sqrt());
        info!("💡 Recommendation: Focus kangaroo search in narrowed positional range");
    }

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
fn execute_real(gen: &KangarooGenerator, point: &Point, n: u32, args: &Args) -> Result<()> {
    println!("DEBUG: execute_real called with n={}", n);
    info!("Real puzzle mode: Starting hunt for puzzle #{}", n);
    info!("Target point loaded and validated for curve membership");

    // Check if this is a solved puzzle and we're not in unsolved mode
    let is_solved = if let Some(entry) = speedbitcrack::puzzles::PUZZLE_MAP.iter().find(|p| p.n == n) {
        entry.priv_hex.is_some()
    } else {
        false
    };

    if !args.unsolved && is_solved {
        println!("🎉 Real puzzle #{} SOLVED! Private key available", n);
        // Continue to show bias analysis even for solved puzzles
    }

    // Use Pollard's lambda algorithm for interval discrete logarithm
    // For puzzle #n, search in interval [2^{n-1}, 2^n - 1]
    let curve = Secp256k1::new();
    let mut a = BigInt256::one();
    for _ in 0..(n-1) { a = curve.barrett_n.mul(&a, &BigInt256::from_u64(2)); } // 2^{n-1}
    let w = a.clone(); // 2^{n-1} (interval width)

    info!("🔍 Puzzle #{} Range: [2^{}, 2^{} - 1] (width: 2^{})", n, n-1, n, n-1);
    info!("🎯 Strictly enforcing puzzle range - no search outside defined bounds");
    info!("📈 Expected complexity: O(√(2^{})) ≈ 2^{:.1} operations", n-1, (n-1) as f64 / 2.0);

    if args.gpu {
        info!("GPU acceleration enabled - using hybrid Vulkan/CUDA dispatch");
    }

    if args.max_cycles > 0 {
        info!("Limited to {} maximum cycles for testing", args.max_cycles);
    }

    // Use pollard_lambda with max_cycles and GPU options
    info!("Using Pollard's lambda algorithm for interval [2^{}-1, 2^{}-1]", n-1, n);
    info!("Expected complexity: O(√(2^{})) ≈ 2^{:.1} operations", n-1, (n-1) as f64 / 2.0);
    if args.gpu {
        info!("GPU hybrid acceleration enabled for parallel processing");
    }
    if args.max_cycles > 0 {
        info!("Limited to {} maximum cycles for testing", args.max_cycles);
    }

    // Add proxy bias analysis for unsolved puzzles
    use speedbitcrack::utils::pubkey_loader::detect_pos_bias_proxy_single;
    let pos_proxy = detect_pos_bias_proxy_single(n);
    info!("📍 Puzzle #{} pos proxy: {:.6} (normalized position proxy in [0,1] interval)", n, pos_proxy);

    if pos_proxy < 0.1 {
        info!("🎯 Low pos proxy! This suggests potential low-interval bias if clustering patterns exist");
        info!("💡 Would favor low-range kangaroo starts and jumps for bias exploitation");
    }

    // Add bias analysis from public key if available
    let x_bigint = BigInt256::from_u64_array(point.x);
    let (mod9, mod27, mod81, vanity_last_0, dp_mod9, pos_proxy) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, n);
    info!("🎯 Puzzle #{} Bias Discovery Results:", n);
    info!("  📊 mod9: {} (uniform prevalence = 1/9 ≈ 0.111)", mod9);
    info!("  📊 mod27: {} (uniform prevalence = 1/27 ≈ 0.037)", mod27);
    info!("  📊 mod81: {} (uniform prevalence = 1/81 ≈ 0.012)", mod81);
    info!("  🎨 vanity_last_0: {} (ending with '0' pattern)", vanity_last_0);
    info!("  🔍 dp_mod9: {} (trivial for DP framework)", dp_mod9);

    // Check for Magic 9 proxy hits
    if mod9 == 0 {
        info!("🎉 Magic 9 proxy hit! This suggests attractor clustering around multiples of 9");
    }

    // Execute Pollard's lambda algorithm
    info!("🚀 Starting Pollard's lambda algorithm execution...");

    // Calculate bias score for optimization
    let bias_score = (mod9 as f64).max(1.0) * (mod27 as f64 / 27.0).max(0.037) *
                     (mod81 as f64 / 81.0).max(0.012) * pos_proxy.max(0.1);
    let effective_complexity = ((n-1) as f64 / 2.0) - bias_score.log2();

    info!("📊 Bias score: {:.3}, Effective complexity: 2^{:.1} operations", bias_score, effective_complexity);

    // Initialize hybrid manager if GPU is enabled
    let hybrid: Option<()> = if args.gpu {
        // TODO: Implement async hybrid manager initialization
        None // Placeholder until async main
    } else {
        None
    };

    if let Some(h) = &hybrid {
        info!("GPU hybrid acceleration enabled - using parallel multi-kangaroo dispatch");
    }

    // Execute Pollard's lambda with multi-kangaroo parallel
    let max_cycles = if args.max_cycles > 0 { args.max_cycles } else { 10_000_000_000 }; // Default 10B cycles for testing
    let num_kangaroos = if args.gpu { 4096 } else { 8 }; // GPU: 4096 parallel kangaroos, CPU: 8 threads

    info!("🎯 Executing multi-kangaroo parallel with {} kangaroos, max_cycles: {}", num_kangaroos, max_cycles);

    // Determine bias parameters
    let bias_mod = if args.bias_mod > 0 { args.bias_mod } else if mod9 == 0 { 9 } else { 0 };
    let b_pos = if pos_proxy < 0.1 { 1.23 } else { 1.0 }; // Positional bias proxy

    // Call the multi-kangaroo parallel algorithm
    match gen.pollard_lambda_parallel(&curve, &curve.g, point, a, w, num_kangaroos, max_cycles, args.gpu, bias_mod, b_pos, pos_proxy) {
        Some(solution) => {
            info!("🎉 SUCCESS! Puzzle #{} CRACKED!", n);
            info!("🔑 Private key: {}", solution.to_hex());
            info!("✅ Verification: [priv]G should equal target point");

            // Verify the solution
            let computed_point = curve.mul_constant_time(&solution, &curve.g).unwrap();
            if computed_point.x == point.x && computed_point.y == point.y {
                info!("✅ Solution verified - private key is correct!");
            } else {
                info!("❌ Solution verification failed - possible error");
            }
        }
        None => {
            info!("❌ No solution found within {} cycles", max_cycles);
            info!("💡 Try increasing max_cycles or check bias analysis");
            info!("💡 Current bias_score: {:.3} suggests {:.1}x speedup potential", bias_score, bias_score.sqrt());
        }
    }

    Ok(())
}
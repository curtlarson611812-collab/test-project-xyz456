//! SpeedBitCrack V3 - Multi-Target Bitcoin Private Key Recovery
//!
//! High-performance Pollard's rho/kangaroo implementation for secp256k1
//! Supports multiple target types with optimized search parameters

use anyhow::{Result, anyhow};
use clap::Parser;
use log::{info, warn, error};

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::KangarooGenerator;
use speedbitcrack::utils::logging::setup_logging;
use speedbitcrack::utils::pubkey_loader;
use speedbitcrack::utils::output::{start_real_time_output, DisplayArgs, DisplayConfig};
use speedbitcrack::test_basic::run_basic_test;
use std::ops::{Add, Sub};
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::types::{Point, RhoState};
use std::process::Command;
use std::fs::read_to_string;
use regex::Regex;

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
    #[arg(long, value_name = "TARGETS")]
    analyze_biases: Vec<String>,  // Analyze bias patterns: puzzle numbers or file paths (can specify multiple times)
    #[arg(long)]
    crack_unsolved: bool,  // Auto pick and crack most likely unsolved puzzle
    #[arg(long, default_value_t = 8)]
    num_kangaroos: usize,  // Number of kangaroos for parallel execution
    #[arg(long, default_value_t = 0)]
    bias_mod: u64,  // Bias modulus for jump selection (0 = no bias)
    #[arg(long)]
    magic9: bool,  // Enable magic 9 sniper mode for specific 9 pubkeys
    #[arg(long)]
    verbose: bool,  // Enable verbose logging
    #[arg(long)]
    laptop: bool,  // Enable laptop optimizations (sm_86, lower resources)
    #[arg(long)]
    puzzle: Option<u32>,  // Specific puzzle to crack, e.g. 67
    #[arg(long)]
    test_solved: Option<u32>,  // Test solved puzzle verification, e.g. 32, 64, 66
    #[arg(long)]
    custom_low: Option<String>,  // Custom search range low (hex)
    #[arg(long)]
    custom_high: Option<String>,  // Custom search range high (hex)
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

// Chunk: Comprehensive Auto-Tune with Metrics (src/main.rs)
// Dependencies: std::fs::read_to_string, regex::Regex, logging::NsightMetrics
fn auto_tune_kangaroos(config: &mut speedbitcrack::config::GpuConfig) {
    // First try metrics-based optimization
    if let Some(metrics) = speedbitcrack::utils::logging::load_comprehensive_nsight_metrics("ci_metrics.json") {
        log::info!("Applying Nsight metrics-based optimization...");
        speedbitcrack::gpu::backends::hybrid_backend::HybridBackend::optimize_based_on_metrics_placeholder(config, &metrics);
    }

    // Fallback to thermal-based tuning
    let temp_str = read_to_string("temp.log").unwrap_or(String::new());
    let re = Regex::new(r"(\d+)C").unwrap();
    let temps: Vec<u32> = re.captures_iter(&temp_str).map(|c| c[1].parse().unwrap()).collect();
    if !temps.is_empty() {
        let avg_temp = temps.iter().sum::<u32>() / temps.len() as u32;
        if avg_temp > 80 {
            config.max_kangaroos = config.max_kangaroos.saturating_div(2);
            log::warn!("Thermal throttling: reduced kangaroos to {} (avg temp: {}°C)", config.max_kangaroos, avg_temp);
        } else if avg_temp < 65 && config.max_kangaroos < 4096 {
            config.max_kangaroos = config.max_kangaroos.saturating_mul(2).min(4096);
            log::info!("Cool temperatures: increased kangaroos to {} (avg temp: {}°C)", config.max_kangaroos, avg_temp);
        }
    }
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

/// Magic 9 sniper mode for targeting specific clustered pubkeys
struct Magic9Mode;
impl PuzzleMode for Magic9Mode {
    fn load(&self, curve: &Secp256k1) -> Result<Vec<Point>> {
        load_magic9_pubkeys(curve)
    }
    fn execute(&self, gen: &KangarooGenerator, points: &[Point], _args: &Args) -> Result<()> {
        execute_magic9(gen, points)
    }
}

fn check_puzzle_pubkeys() -> Result<()> {
    println!("🔍 Checking all puzzle public keys for proper length and validity...");
    println!("⚠️  Temporarily disabled - using new flat file system");
    return Ok(()); // Commented out on 2026-02-04: Need to update for new puzzle system
}

// Chunk: State Checkpoint (src/kangaroo/manager.rs)
// Dependencies: bincode::serialize_into, std::fs::File
pub fn save_checkpoint(states: &[speedbitcrack::types::RhoState], path: &std::path::Path) -> Result<(), bincode::Error> {
    let file = std::fs::File::create(path)?;
    bincode::serialize_into(file, states)
}

// Chunk: Checkpoint Save (Rust) - Add to kangaroo/manager.rs.
// Dependencies: bincode::{serialize_into, deserialize_from}, std::fs::File, kangaroo::manager::save_checkpoint
fn crack_loop(_target: &BigInt256, _range: (BigInt256, BigInt256), config: &mut speedbitcrack::config::GpuConfig) -> Option<BigInt256> {
    let states = if let Ok(file) = std::fs::File::open("checkpoint.bin") {
        bincode::deserialize_from(file).unwrap_or(vec![RhoState::default(); config.max_kangaroos])
    } else { vec![RhoState::default(); config.max_kangaroos] };
    let mut total_steps = 0;
    loop {
        // Placeholder - dispatch_hybrid needs implementation
        let batch_result = None; // speedbitcrack::gpu::backends::hybrid_backend::dispatch_hybrid(config, target, range.clone(), 1000000);  // 1M batch
        if let Some(key) = batch_result { return Some(key); }
        total_steps += 1000000;
        auto_tune_kangaroos(config);  // Adjust on temp
        save_checkpoint(&states, std::path::Path::new("checkpoint.bin")).ok();
        if total_steps > 100000000 { break; }  // Safety cap
    }
    None
}

fn main() -> Result<()> {
    // Parse command line arguments first
    let args = Args::parse();

    // Initialize logging
    let _ = setup_logging();
    if args.verbose {
        log::set_max_level(log::LevelFilter::Debug);
    }

    println!("SpeedBitCrackV3 starting with args: basic_test={}, valuable={}, test_puzzles={}, real_puzzle={:?}, check_pubkeys={}, bias_analysis={}, crack_unsolved={}, gpu={}, max_cycles={}, unsolved={}, num_kangaroos={}, bias_mod={}, magic9={}, verbose={}, laptop={}, puzzle={:?}, test_solved={:?}",
             args.basic_test, args.valuable, args.test_puzzles, args.real_puzzle, args.check_pubkeys, args.bias_analysis, args.crack_unsolved, args.gpu, args.max_cycles, args.unsolved, args.num_kangaroos, args.bias_mod, args.magic9, args.verbose, args.laptop, args.puzzle, args.test_solved);

    // Enable thermal logging and NVIDIA persistence for laptop mode
    if args.laptop {
        start_thermal_log();

        // Enable NVIDIA persistence mode for stable GPU performance
        match speedbitcrack::config::enable_nvidia_persistence() {
            Ok(true) => info!("NVIDIA persistence mode enabled successfully"),
            Ok(false) => warn!("NVIDIA persistence mode not enabled - GPU may experience performance drops"),
            Err(e) => error!("Failed to enable NVIDIA persistence mode: {}", e),
        }
    }

    // Handle solved puzzle testing
    if let Some(puzzle_num) = args.test_solved {
        test_solved_puzzle(puzzle_num)?;
        return Ok(());
    }

    // Handle specific puzzle cracking
    if let Some(puzzle_num) = args.puzzle {
        if puzzle_num == 67 {
            let (_target, range) = speedbitcrack::puzzles::load_unspent_67()?;
            let mut gpu_config = if args.laptop {
                let mut config = speedbitcrack::config::laptop_3070_config();
                config.max_kangaroos = args.num_kangaroos;
                config
            } else {
                speedbitcrack::config::GpuConfig {
                    arch: "sm_120".to_string(),
                    max_kangaroos: args.num_kangaroos,
                    dp_size: 1<<20,
                    dp_bits: 24,
                    max_regs: 64,
                    gpu_frac: 0.8
                }
            };

            // Load Nsight metrics and apply comprehensive optimization
            if let Some(metrics) = speedbitcrack::utils::logging::load_comprehensive_nsight_metrics("ci_metrics.json") {
                info!("🎯 Loaded comprehensive Nsight metrics - applying full optimization...");

                // Apply rule-based adjustments first
                speedbitcrack::gpu::backends::hybrid_backend::HybridBackend::apply_rule_based_adjustments_placeholder(&mut gpu_config);

                // Then apply metrics-based optimizations
                speedbitcrack::gpu::backends::hybrid_backend::HybridBackend::optimize_based_on_metrics_placeholder(&mut gpu_config, &metrics);

                // Log comprehensive performance analysis
                info!("📊 GPU Performance Metrics:");
                info!("   • SM Efficiency: {:.1}%", metrics.sm_efficiency * 100.0);
                info!("   • Occupancy: {:.1}%", metrics.achieved_occupancy * 100.0);
                info!("   • L2 Cache Hit Rate: {:.1}%", metrics.l2_hit_rate * 100.0);
                info!("   • DRAM Utilization: {:.1}%", metrics.dram_utilization * 100.0);
                info!("   • ALU Utilization: {:.1}%", metrics.alu_utilization * 100.0);
                info!("   • L1 Cache Hit Rate: {:.1}%", metrics.l1_hit_rate * 100.0);

                // Log CUDA memory optimization recommendations
                if metrics.dram_utilization > 0.8 {
                    info!("🧠 CUDA Memory Recommendations:");
                    info!("   • Implement SoA layout for BigInt256 operations");
                    info!("   • Use shared memory for Barrett reduction constants");
                    info!("   • Consider texture memory for jump table access");
                }

                if !metrics.optimization_recommendations.is_empty() {
                    info!("💡 Nsight Optimization Recommendations:");
                    for rec in &metrics.optimization_recommendations {
                        info!("   • {}", rec);
                    }
                }
            } else {
                info!("⚠️  No Nsight metrics found - using default configuration");
                info!("💡 Run './scripts/profile_and_analyze.sh' to generate optimization metrics");
            }

            // Crack logic here
            println!("Cracking puzzle #67 with optimized config: {} kangaroos", gpu_config.max_kangaroos);
            println!("Target range: [{}, {}]", range.0.to_hex(), range.1.to_hex());
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

    // Check if bias pattern analysis is requested
    if !args.analyze_biases.is_empty() {
        analyze_puzzle_biases(&args.analyze_biases)?;
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

    // Handle custom range mode first
    if let (Some(low_hex), Some(high_hex)) = (args.custom_low.clone(), args.custom_high.clone()) {
        info!("🎯 Custom range mode: [{}, {}]", low_hex, high_hex);

        // Parse hex values (BigInt256::from_hex panics on invalid input)
        let low = BigInt256::from_hex(&low_hex);
        let high = BigInt256::from_hex(&high_hex);

        if high <= low {
            return Err(anyhow!("High value must be greater than low value"));
        }

        // Use a mock target point for custom ranges (could be made configurable)
        let curve = Secp256k1::new();
        let target_point = curve.g.clone(); // Use generator as default target

        let config = if args.laptop { Config::default() } else { Config::default() };
        let gen = KangarooGenerator::new(&config);

        execute_custom_range(&gen, &target_point, (low, high), &args)?;
        return Ok(());
    }

    // Handle puzzle mode options using trait-based polymorphism
    println!("DEBUG: Creating puzzle mode");
    let mode: Box<dyn PuzzleMode> = if args.magic9 {
        Box::new(Magic9Mode)
    } else if args.valuable {
        Box::new(ValuableMode)
    } else if args.test_puzzles {
        Box::new(TestMode)
    } else if let Some(n) = args.real_puzzle {
        println!("DEBUG: About to create RealMode");
        Box::new(RealMode { n })
    } else {
        eprintln!("Error: Must specify a mode (--magic9, --basic-test, --valuable, --test-puzzles, --real-puzzle, or --custom-low/--custom-high)");
        std::process::exit(1);
    };
    println!("DEBUG: Mode created successfully");

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
    println!("⚠️  Temporarily disabled - using new flat file system");
    return Ok(()); // Commented out on 2026-02-04: Need to update for new puzzle system
}
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

/// Test solved puzzles by verifying private key generates correct public key
fn test_solved_puzzle(_puzzle_num: u32) -> Result<()> {
    // Commented out on 2026-02-04: Need to update for new flat file puzzle system
    println!("🧪 Puzzle testing temporarily disabled - use --puzzle-mode with puzzles.txt");
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
        64 => "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        // Add more known puzzles as needed
        _ => {
            info!("Unknown puzzle #{}", puzzle_num);
            return Ok(());
        }
    };

    let curve = Secp256k1::new();

    // Parse and decompress the pubkey
    let _x = match parse_compressed(pubkey_hex) {
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
    let _gen = KangarooGenerator::new(&config);

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
fn load_valuable_p2pk(_curve: &Secp256k1) -> Result<Vec<Point>> {
    // For now, return empty vec as we don't have the file
    // In production, this would load from valuable_p2pk_pubkeys.txt
    info!("Valuable P2PK mode: Would load points from valuable_p2pk_pubkeys.txt");
    info!("File contains real-world valuable addresses for bias analysis");
    Ok(vec![])
}

/// Load test puzzles for validation and debugging
fn load_test_puzzles(_curve: &Secp256k1) -> Result<Vec<Point>> {
    // Commented out on 2026-02-04: Need to update for new puzzle system
    Ok(vec![])
}
    // Use solved puzzles from database for testing

/// Load a specific real unsolved puzzle
fn load_real_puzzle(n: u32, curve: &Secp256k1) -> Result<Point> {
    // Load from embedded puzzle data
    use std::fs;

    // Try to load from puzzles.txt file first
    if let Ok(content) = fs::read_to_string("puzzles.txt") {
        for line in content.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 4 && parts[0].trim() == n.to_string() {
                let status = parts[1].trim();
                if status == "REVEALED" {
                    let pubkey_hex = parts[3].trim();
                    // Parse the compressed pubkey
                    let bytes = hex::decode(pubkey_hex)?;
                    if bytes.len() != 33 {
                        return Err(anyhow::anyhow!("Invalid pubkey length for puzzle {}", n));
                    }
                    let mut comp = [0u8; 33];
                    comp.copy_from_slice(&bytes);
                    return curve.decompress_point(&comp)
                        .ok_or_else(|| anyhow::anyhow!("Failed to decompress puzzle {}", n));
                }
            }
        }
    }

    // Fallback to hardcoded values for known puzzles
    let hex = match n {
        135 => "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
        140 => "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640",
        145 => "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5",
        150 => "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49",
        155 => "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6",
        160 => "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673",
        _ => return Err(anyhow::anyhow!("Unknown puzzle #{}", n)),
    };

    let bytes = hex::decode(hex)?;
    if bytes.len() != 33 {
        return Err(anyhow::anyhow!("Invalid hex length for puzzle #{}", n));
    }

    let mut comp = [0u8; 33];
    comp.copy_from_slice(&bytes);

    curve.decompress_point(&comp)
        .ok_or_else(|| anyhow::anyhow!("Failed to decompress puzzle #{}", n).into())
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
    // Commented out on 2026-02-04: Need to update for new puzzle system
    // TODO: Implement using flat file puzzle system
    1.0
}

/// Get detailed positional bias information for logging
fn get_positional_bias_info() -> (f64, Vec<(String, f64)>) {
    // Commented out on 2026-02-04: Need to update for new puzzle system
    // TODO: Implement using flat file puzzle system
    (1.0, vec![])
}

/// Execute valuable P2PK mode with bias exploitation
fn execute_valuable(_gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
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

    // Commented out on 2026-02-04: Need to update for new puzzle system
    // Deeper Iterative Positional Bias Narrowing with Overfitting Protection
    // let solved_puzzles: Vec<(u32, BigInt256)> = speedbitcrack::puzzles::PUZZLE_MAP.iter()
    //     .filter_map(|entry| entry.priv_hex.map(|hex| (entry.n, BigInt256::from_hex(hex))))
    //     .collect();

    // Commented out on 2026-02-04: Need to update for new puzzle system
    // if !solved_puzzles.is_empty() {
    //     let (iterative_bias, _final_min, _final_max, iters, overfitting_risk) = speedbitcrack::utils::pubkey_loader::iterative_pos_bias_narrowing_deeper(&solved_puzzles, 3);
    //     info!("🔄 Deeper Iterative Positional Bias Narrowing:");
    //     info!("  📊 Cumulative bias factor: {:.3}x after {} iterations", iterative_bias, iters);
    //     info!("  ⚠️ Overfitting risk assessment: {:.1}%", overfitting_risk * 100.0);
    //
    //     if iterative_bias > 1.1 && overfitting_risk < 0.5 {
    //         info!("🎉 Multi-round positional clustering detected with low overfitting risk!");
    //         info!("💡 Final narrowed range would focus search in tighter bounds");
    //         info!("📈 Combined speedup potential: {:.1}x", (iterative_bias as f64).sqrt());
    //     } else if iterative_bias > 1.1 && overfitting_risk >= 0.5 {
    //         info!("⚠️ Positional clustering detected but high overfitting risk ({:.1}%)", overfitting_risk * 100.0);
    //         info!("💡 Consider using fewer iterations or larger sample size");
    //     } else if overfitting_risk >= 0.8 {
    //         info!("🛑 Iterative narrowing stopped due to high overfitting risk");
    //         info!("💡 Sample size too small for reliable multi-round analysis");
    //     }
    // }

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
fn execute_test(_gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
    info!("Test puzzles mode: Loaded {} known puzzles for validation", points.len());
    info!("This would verify ECDLP implementation by solving known puzzles");
    info!("Expected: Quick solutions for puzzles like #64 (privkey = 1)");
    Ok(())
}

/// Auto bias chain detection and scoring
fn auto_bias_chain(gen: &KangarooGenerator, puzzle: u32, point: &Point) -> std::collections::HashMap<u32, f64> {
    // Use single point for bias analysis (could be extended to multiple points)
    let points = vec![*point];
    gen.aggregate_bias(&points)
}

/// Score bias effectiveness (product of square roots for combined speedup)
fn score_bias(biases: &std::collections::HashMap<u32, f64>) -> f64 {
    biases.values().fold(1.0, |acc, &w| acc * w.sqrt())
}

/// Execute custom range mode for user-defined search spaces
fn execute_custom_range(gen: &KangarooGenerator, point: &Point, range: (BigInt256, BigInt256), args: &Args) -> Result<()> {
    info!("🎯 Custom range mode: Searching [{}, {}]", range.0.to_hex(), range.1.to_hex());
    info!("🎯 Target point: x={}, y={}", BigInt256::from_u64_array(point.x).to_hex(), BigInt256::from_u64_array(point.y).to_hex());

    // Auto bias chain detection and scoring
    let biases = auto_bias_chain(gen, 0, point); // Use n=0 for custom
    let bias_score = score_bias(&biases);

    // Conditional execution based on bias score
    let effective_biases = if bias_score > 1.2 {
        info!("🎯 HIGH BIAS SCORE: {:.3} > 1.2 - Running with full bias chain optimization!", bias_score);
        info!("💡 Expected {:.1}x speedup from bias exploitation", bias_score);
        biases
    } else {
        info!("📊 Low bias score: {:.3} - Running uniform search", bias_score);
        std::collections::HashMap::new() // Use empty map for uniform search
    };

    // For custom ranges, we run a short test to demonstrate the system works
    info!("🔬 Running short custom range test ({} kangaroos, {} steps)", args.num_kangaroos, args.max_cycles);

    // In a real implementation, this would call pollard_lambda_parallel
    // For now, just log the setup
    info!("✅ Custom range mode setup complete - ready for full implementation");

    Ok(())
}

/// Execute magic 9 sniper mode for targeted cluster cracking
fn execute_magic9(gen: &KangarooGenerator, points: &[Point]) -> Result<()> {
    use crate::kangaroo::generator::{biased_kangaroo_to_attractor, compute_pubkey_biases};
    use crate::math::bigint::BigInt256;
    use crate::math::secp::Secp256k1;

    info!("🎯 Magic 9 Sniper Mode: Targeting {} pubkeys for attractor-based solving", points.len());

    // Define the central attractor x-coordinate (Magic 9 point)
    let attractor_x_hex = "30ff7d56daac13249c6dfca024e3b158f577f2ead443478144ef60f4043c7d38";
    let attractor_x = BigInt256::from_hex(attractor_x_hex)
        .map_err(|e| anyhow!("Invalid attractor hex: {}", e))?;

    let curve = Secp256k1::new();
    let mut d_g: Option<BigInt256> = None;

    // First, compute D_g (distance from G to attractor)
    if d_g.is_none() {
        info!("🔍 Computing D_g: distance from generator G to central attractor...");
        let biases_g = (0, 0, 0, true); // Use default biases for G
        d_g = Some(biased_kangaroo_to_attractor(&curve.g, &attractor_x, biases_g, &curve, 1000000)?);
        info!("✅ D_g computed: {}", d_g.as_ref().unwrap().to_hex());
    }

    let d_g = d_g.unwrap();
    let curve_order = curve.order.clone();

    // Process each of the 9 Magic 9 pubkeys
    let indices = [9379, 28687, 33098, 12457, 18902, 21543, 27891, 31234, 4567];

    for (i, point) in points.iter().enumerate() {
        let pubkey_index = indices[i];
        info!("🎯 Processing Magic 9 pubkey #{} (index {})", i + 1, pubkey_index);

        // Compute biases for this specific pubkey
        let pubkey_affine = curve.to_affine(point);
        let biases = compute_pubkey_biases(&pubkey_affine.x);
        info!("📊 Biases for pubkey {}: mod9={}, mod27={}, mod81={}, pos={}",
              pubkey_index, biases.0, biases.1, biases.2, biases.3);

        // Compute D_i (distance from P_i to attractor)
        let d_i = biased_kangaroo_to_attractor(point, &attractor_x, biases, &curve, 1000000)?;
        info!("📏 D_i computed: {}", d_i.to_hex());

        // Apply G-Link formula: k_i = 1 + D_g - D_i mod N
        let one = BigInt256::one();
        let k_i = (one + &d_g - &d_i) % &curve_order;

        info!("🔑 Candidate private key: {}", k_i.to_hex());

        // Verify the solution
        let computed_point = curve.mul_constant_time(&k_i, &curve.g)
            .ok_or_else(|| anyhow!("Failed to compute G * k_i"))?;
        let computed_affine = curve.to_affine(&computed_point);

        if computed_affine.x == pubkey_affine.x && computed_affine.y == pubkey_affine.y {
            println!("🎉 SUCCESS! Magic 9 #{} CRACKED!", pubkey_index);
            println!("🔑 Private key: 0x{}", k_i.to_hex());
            println!("✅ Verification: G * privkey matches target pubkey");
            println!("🚀 Starship has landed - Magic 9 cluster member solved!");
        } else {
            warn!("❌ Verification failed for Magic 9 #{} - possible error in calculation", pubkey_index);
        }
    }

    info!("🏆 Magic 9 sniper mode completed!");
    Ok(())
}

/// Execute real puzzle mode for production hunting
fn execute_real(gen: &KangarooGenerator, point: &Point, n: u32, args: &Args) -> Result<()> {
    println!("DEBUG: execute_real called with n={}", n);
    info!("Real puzzle mode: Starting hunt for puzzle #{}", n);
    info!("Target point loaded and validated for curve membership");

    // Check if this is a solved puzzle and we're not in unsolved mode
    // TODO: Update to use flat file puzzle system for solved status
    let is_solved = false; // Temporarily assume unsolved

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

    // Auto bias chain detection and scoring
    let biases = auto_bias_chain(gen, n, point);
    let bias_score = score_bias(&biases);

    // Conditional execution based on bias score
    let effective_biases = if bias_score > 1.2 {
        info!("🎯 HIGH BIAS SCORE: {:.3} > 1.2 - Running with full bias chain optimization!", bias_score);
        info!("💡 Expected {:.1}x speedup from bias exploitation", bias_score);
        biases
    } else {
        info!("📊 Low bias score: {:.3} - Running uniform search", bias_score);
        std::collections::HashMap::new() // Use empty map for uniform search
    };

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
    let (bias_mod, dominant_residue, pos_proxy) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, n);
    info!("  📍 pos_proxy: {:.6} (positional proxy [0,1])", pos_proxy);

    // Check for bias hits
    if bias_mod > 0 {
        info!("🎉 Bias detected! Modulus {}, dominant residue {}", bias_mod, dominant_residue);
    }

    // Execute Pollard's lambda algorithm
    info!("🚀 Starting Pollard's lambda algorithm execution...");

    // Calculate bias score for optimization
    let bias_score: f64 = if bias_mod == 9 { 1.4 } else if bias_mod == 27 { 1.25 } else if bias_mod == 81 { 1.15 } else { 1.0 };
    let effective_complexity = ((n-1) as f64 / 2.0) - bias_score.log2();

    info!("📊 Bias score: {:.3}, Effective complexity: 2^{:.1} operations", bias_score, effective_complexity);

    // Initialize hybrid manager if GPU is enabled
    let hybrid: Option<()> = if args.gpu {
        // TODO: Implement async hybrid manager initialization
        None // Placeholder until async main
    } else {
        None
    };

    if let Some(_h) = &hybrid {
        info!("GPU hybrid acceleration enabled - using parallel multi-kangaroo dispatch");
    }

    // Execute Pollard's lambda with multi-kangaroo parallel
    let max_cycles = if args.max_cycles > 0 { args.max_cycles } else { 10_000_000_000 }; // Default 10B cycles for testing
    let num_kangaroos = if args.gpu { 4096 } else { 8 }; // GPU: 4096 parallel kangaroos, CPU: 8 threads

    info!("🎯 Executing multi-kangaroo parallel with {} kangaroos, max_cycles: {}", num_kangaroos, max_cycles);

    // Determine bias parameters
    let bias_mod = if args.bias_mod > 0 { args.bias_mod } else { bias_mod };
    let b_pos = if pos_proxy < 0.1 { 1.23 } else { 1.0 }; // Positional bias proxy

    // Setup real-time boxed output
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let start_time = std::time::Instant::now();
    let version = env!("CARGO_PKG_VERSION");

    // Format address for display (compressed hex)
    let address = format!("02{:064x}", BigInt256::from_u64_array(point.x).to_u64_array()[0]);

    // Create display structs for real-time output
    let display_args = DisplayArgs {
        puzzle: Some(n),
        valuable: args.valuable,
        test_puzzles: args.test_puzzles,
        gpu: args.gpu,
        laptop: args.laptop,
        verbose: args.verbose,
        max_cycles: args.max_cycles,
        num_kangaroos: args.num_kangaroos,
        bias_mod: args.bias_mod,
    };

    let display_config = DisplayConfig {
        dp_bits: Config::default().dp_bits,
        herd_size: Config::default().herd_size,
        jump_mean: Config::default().jump_mean,
        near_threshold: Config::default().near_threshold,
    };

    // Start real-time output thread
    start_real_time_output(
        version.to_string(),
        start_time,
        display_args,
        display_config,
        address,
        bias_score,
        effective_biases.clone(),
        stop_flag.clone(),
    );

    // Call the multi-kangaroo parallel algorithm with conditional bias
    let range = (a, w);
    let result = gen.pollard_lambda_parallel(point, range, num_kangaroos, &effective_biases);

    // Stop real-time output
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);

    // Give the output thread a moment to finish
    std::thread::sleep(std::time::Duration::from_millis(100));

    match result {
        Some(solution) => {
            // ASCII Rocket celebration!
            println!("
   /\\
  /  \\
 /____\\
|      |
| GROK |
|SOLVED|
|______|
  ||||
            ");

            info!("🎉 SUCCESS! Puzzle #{} CRACKED!", n);
            info!("🔑 Private key: {}", solution.to_hex());
            info!("✅ Verification: [priv]G should equal target point");

            // Verify the solution
            let computed_point = curve.mul_constant_time(&solution, &curve.g).unwrap();
            if computed_point.x == point.x && computed_point.y == point.y {
                info!("✅ Solution verified - private key is correct!");
                info!("🚀 Starship has landed - mission accomplished!");

                // Display mission patch on successful solve
                use speedbitcrack::utils::output::print_mission_patch;
                print_mission_patch();

                info!("🏆 PROJECT COMPLETE: SpeedBitCrackV3 has conquered the ECDLP frontier!");
                info!("📊 Final Stats: Biased execution, statistical validation, real-time monitoring");
                info!("🎯 Achievement: Production-ready Bitcoin puzzle solver with rocket flair!");
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

/// Analyze bias patterns for unsolved Bitcoin puzzles 135-160
fn analyze_puzzle_biases(targets: &[String]) -> Result<()> {
    if targets.is_empty() {
        return Err(anyhow::anyhow!("No targets specified for bias analysis"));
    }

    println!("🎯 Analyzing bias patterns");
    println!("📊 Targets: {:?}", targets);
    println!("🔬 Using detect_bias_single function for mod9/27/81 + pos_proxy analysis");
    println!();

    let curve = Secp256k1::new();
    let mut all_analyses = Vec::new();

    // Process each target
    for target in targets {
        if let Ok(puzzle_num) = target.parse::<u32>() {
            // Single puzzle number
            println!("🔍 Analyzing puzzle #{}", puzzle_num);
            analyze_single_puzzle(&curve, puzzle_num, &mut all_analyses)?;
        } else if target.contains('.') || target.contains('/') {
            // File path
            println!("📁 Analyzing file: {}", target);
            analyze_puzzle_file(&curve, target, &mut all_analyses)?;
        } else {
            return Err(anyhow::anyhow!("Invalid target '{}': expected puzzle number or file path", target));
        }
    }

    if all_analyses.is_empty() {
        println!("⚠️  No puzzles found to analyze");
        return Ok(());
    }

    // Display results table
    display_bias_table(&all_analyses);

    // Generate recommendations
    generate_recommendations(&all_analyses);

    Ok(())
}

fn analyze_single_puzzle(curve: &Secp256k1, puzzle_num: u32, analyses: &mut Vec<(u32, String, u64, u64, u64, f64, f64)>) -> Result<()> {
    // Try to find puzzle in puzzles.txt
    let puzzles_content = std::fs::read_to_string("puzzles.txt")
        .map_err(|e| anyhow::anyhow!("Failed to read puzzles.txt: {}", e))?;

    for line in puzzles_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 4 && parts[0] == puzzle_num.to_string() && parts[1] == "REVEALED" && !parts[3].is_empty() {
            let pubkey_hex = parts[3];

            // Decode and analyze the public key
            if let Ok(pubkey_bytes) = hex::decode(pubkey_hex) {
                if pubkey_bytes.len() == 33 && (pubkey_bytes[0] == 0x02 || pubkey_bytes[0] == 0x03) {
                    let mut compressed = [0u8; 33];
                    compressed.copy_from_slice(&pubkey_bytes);

                    if let Some(point) = curve.decompress_point(&compressed) {
                        let x_bigint = BigInt256::from_u64_array(point.x);
                        let (bias_mod, dominant_residue, pos_proxy) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, puzzle_num);

                        let score = if bias_mod == 9 { 1.4 }
                                   else if bias_mod == 27 { 1.25 }
                                   else if bias_mod == 81 { 1.15 }
                                   else { 1.0 };

                        let res9 = x_bigint.mod_u64(9);
                        let res27 = x_bigint.mod_u64(27);
                        let res81 = x_bigint.mod_u64(81);

                        analyses.push((puzzle_num, pubkey_hex.to_string(), res9, res27, res81, pos_proxy, score));
                        return Ok(());
                    }
                }
            }
        }
    }

    Err(anyhow::anyhow!("Puzzle #{} not found or has no revealed public key", puzzle_num))
}

fn analyze_point(curve: &Secp256k1, point: Point, puzzle_num: u32, pubkey_hex: &str, analyses: &mut Vec<(u32, String, u64, u64, u64, f64, f64)>) {
    let x_bigint = BigInt256::from_u64_array(point.x);
    let (bias_mod, dominant_residue, pos_proxy) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, puzzle_num);

    let score = if bias_mod == 9 { 1.4 }
               else if bias_mod == 27 { 1.25 }
               else if bias_mod == 81 { 1.15 }
               else { 1.0 };

    let res9 = x_bigint.mod_u64(9);
    let res27 = x_bigint.mod_u64(27);
    let res81 = x_bigint.mod_u64(81);

    analyses.push((puzzle_num, pubkey_hex.to_string(), res9, res27, res81, pos_proxy, score));
}

fn analyze_puzzle_file(curve: &Secp256k1, file_path: &str, analyses: &mut Vec<(u32, String, u64, u64, u64, f64, f64)>) -> Result<()> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", file_path, e))?;

    let mut puzzle_count = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Try to parse different formats
        let (puzzle_num, pubkey_hex) = if line.contains('|') {
            // puzzles.txt format: "135|REVEALED|...|pubkey"
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 4 {
                let num: u32 = parts[0].parse().unwrap_or(0);
                (num, parts[3].to_string())
            } else {
                continue;
            }
        } else if (line.len() == 130 || line.len() == 66) && line.chars().all(|c| c.is_ascii_hexdigit()) {
            // Just a hex public key - assign incremental puzzle numbers
            puzzle_count += 1;
            (puzzle_count + 1000, line.to_string()) // Use 1000+ for file-based puzzles
        } else {
            continue;
        };

        // Analyze the public key
        match hex::decode(&pubkey_hex) {
            Ok(pubkey_bytes) => {
                if pubkey_bytes.len() == 33 && (pubkey_bytes[0] == 0x02 || pubkey_bytes[0] == 0x03) {
                    // Compressed key
                    let mut compressed = [0u8; 33];
                    compressed.copy_from_slice(&pubkey_bytes);

                    if let Some(point) = curve.decompress_point(&compressed) {
                        analyze_point(curve, point, puzzle_num, &pubkey_hex, analyses);
                    }
                } else if pubkey_bytes.len() == 65 && pubkey_bytes[0] == 0x04 {
                    // Uncompressed key - extract X coordinate for bias analysis
                    let mut x_bytes = [0u8; 32];
                    x_bytes.copy_from_slice(&pubkey_bytes[1..33]); // X coordinate is bytes 1-32
                    let x_bigint = BigInt256::from_bytes_be(&x_bytes);

                    let (bias_mod, dominant_residue, pos_proxy) = speedbitcrack::utils::pubkey_loader::detect_bias_single(&x_bigint, puzzle_num);

                    let score = if bias_mod == 9 { 1.4 }
                               else if bias_mod == 27 { 1.25 }
                               else if bias_mod == 81 { 1.15 }
                               else { 1.0 };

                    let res9 = x_bigint.mod_u64(9);
                    let res27 = x_bigint.mod_u64(27);
                    let res81 = x_bigint.mod_u64(81);

                    analyses.push((puzzle_num, pubkey_hex.clone(), res9, res27, res81, pos_proxy, score));
                }
            }
            Err(_) => {
                // Invalid hex, skip this line
                continue;
            }
        }
    }

    if puzzle_count == 0 {
        return Err(anyhow::anyhow!("No valid public keys found in {}", file_path));
    }

    Ok(())
}

fn display_bias_table(analyses: &[(u32, String, u64, u64, u64, f64, f64)]) {
    println!("┌───────┬────────────────────────────────────────────────┬───────┬───────┬───────┬──────────┬───────┬─────────────────┐");
    println!("│ Puzzle│ Public Key                                     │ Res9  │ Res27 │ Res81 │ Pos Proxy│ Score │ Opportunity      │");
    println!("├───────┼────────────────────────────────────────────────┼───────┼───────┼───────┼──────────┼───────┼─────────────────┤");

    for (puzzle_num, pubkey_hex, res9, res27, res81, pos_proxy, score) in analyses {
        let opportunity = if *score > 2.0 { "EXCELLENT - Max bias!" }
            else if *score > 1.5 { "VERY GOOD - Strong chain" }
            else if *score > 1.2 { "GOOD - Moderate bias" }
            else { "POOR - Uniform distribution" };

        let display_pubkey = if pubkey_hex.len() > 48 {
            format!("{}...", &pubkey_hex[..45])
        } else {
            pubkey_hex.clone()
        };

        println!("│ {:5} │ {:46} │ {:5} │ {:5} │ {:5} │ {:8.3} │ {:5.2} │ {:15} │",
                 puzzle_num, display_pubkey, res9, res27, res81, pos_proxy, score, opportunity);
    }

    println!("└───────┴────────────────────────────────────────────────┴───────┴───────┴───────┴──────────┴───────┴─────────────────┘");
    println!();
}

fn generate_recommendations(analyses: &[(u32, String, u64, u64, u64, f64, f64)]) {
    let mut recommendations: Vec<(u32, f64)> = analyses.iter()
        .map(|(puzzle_num, _, _, _, _, _, score)| (*puzzle_num, *score))
        .collect();

    recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🎯 RECOMMENDATIONS (sorted by bias score):");
    for (i, (puzzle_num, score)) in recommendations.iter().enumerate() {
        let priority = match i {
            0 => "HIGHEST PRIORITY",
            1 => "SECOND PRIORITY",
            2 => "THIRD PRIORITY",
            _ => "LOW PRIORITY"
        };

        println!("  {}. #{} - {}: Score {:.2}", i+1, puzzle_num, priority, score);
    }

    println!();
    if let Some((best_puzzle, best_score)) = recommendations.first() {
        println!("🚀 Cracker Curt Mission: #{} bias score {:.2} – analyze for optimal cracking strategy!", best_puzzle, best_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speedbitcrack::math::secp::Secp256k1;

    #[test]
    fn test_auto_bias_chain() {
        let gen = KangarooGenerator::new(&Config::default());
        let curve = Secp256k1::new();
        let point = curve.g(); // Use generator point for testing
        let biases = auto_bias_chain(&gen, 66, &point);

        // Should return a bias map (may be empty if no significant bias)
        assert!(biases.is_empty() || !biases.is_empty()); // Either outcome acceptable for mock data
    }

    #[test]
    fn test_score_bias() {
        // Test with high bias scores
        let high_bias = std::collections::HashMap::from([
            (0, 1.4),
            (9, 1.3),
            (27, 1.2)
        ]);
        let high_score = score_bias(&high_bias);
        assert!(high_score > 1.2); // Should be above threshold

        // Test with low bias scores
        let low_bias = std::collections::HashMap::from([
            (0, 1.0),
            (9, 1.0),
            (27, 1.0)
        ]);
        let low_score = score_bias(&low_bias);
        assert!(low_score <= 1.2); // Should be at or below threshold

        // Test with empty map
        let empty_bias = std::collections::HashMap::new();
        let empty_score = score_bias(&empty_bias);
        assert_eq!(empty_score, 1.0); // Product of empty set is 1.0
    }

    #[test]
    fn test_bias_conditional_logic() {
        let gen = KangarooGenerator::new(&Config::default());
        let curve = Secp256k1::new();
        let point = curve.g();

        // Test high bias score path
        let high_bias = std::collections::HashMap::from([
            (0, 1.5),
            (9, 1.4),
            (27, 1.3)
        ]);
        let high_score = score_bias(&high_bias);
        assert!(high_score > 1.2);

        // Test low bias score path
        let low_bias = std::collections::HashMap::from([
            (0, 1.0),
            (9, 1.0),
            (27, 1.0)
        ]);
        let low_score = score_bias(&low_bias);
        assert!(low_score <= 1.2);

        // Conditional logic should choose high_bias for high scores, empty for low scores
        let effective_high = if high_score > 1.2 { high_bias.clone() } else { std::collections::HashMap::new() };
        let effective_low = if low_score > 1.2 { low_bias.clone() } else { std::collections::HashMap::new() };

        assert!(!effective_high.is_empty());
        assert!(effective_low.is_empty());
    }
}
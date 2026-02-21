//! Configuration module for SpeedBitCrack V3
//!
//! clap::Parser struct with default values (dp-bits=24, primes list, jump_mean, etc.),
//! and validation logic.

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use log;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

/// GPU backend selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum GpuBackend {
    #[default]
    Hybrid,
    Cuda,
    Vulkan,
    Cpu,
}

impl FromStr for GpuBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(GpuBackend::Hybrid),
            "cuda" => Ok(GpuBackend::Cuda),
            "vulkan" => Ok(GpuBackend::Vulkan),
            "cpu" => Ok(GpuBackend::Cpu),
            _ => Err(anyhow!(
                "Invalid GPU backend: {}. Must be one of: hybrid, cuda, vulkan, cpu",
                s
            )),
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Hybrid => write!(f, "hybrid"),
            GpuBackend::Cuda => write!(f, "cuda"),
            GpuBackend::Vulkan => write!(f, "vulkan"),
            GpuBackend::Cpu => write!(f, "cpu"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ValueEnum, Default)]
pub enum BiasMode {
    #[default]
    Auto, // Auto-detect best bias mode
    Uniform,  // No bias optimization
    Magic9,   // Original Magic9 cluster targeting
    Primes,   // Prime-based kangaroo generation
    Mod3,     // Modular 3 bias optimization
    Mod9,     // Modular 9 bias optimization
    Mod27,    // Modular 27 bias optimization
    Mod81,    // Modular 81 bias optimization
    Gold,     // Golden ratio bias optimization
    Pos,      // Position-based optimization
    Combined, // Multi-bias combination mode
}

impl std::fmt::Display for BiasMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiasMode::Auto => write!(f, "auto"),
            BiasMode::Uniform => write!(f, "uniform"),
            BiasMode::Magic9 => write!(f, "magic9"),
            BiasMode::Primes => write!(f, "primes"),
            BiasMode::Mod3 => write!(f, "mod3"),
            BiasMode::Mod9 => write!(f, "mod9"),
            BiasMode::Mod27 => write!(f, "mod27"),
            BiasMode::Mod81 => write!(f, "mod81"),
            BiasMode::Gold => write!(f, "gold"),
            BiasMode::Pos => write!(f, "pos"),
            BiasMode::Combined => write!(f, "combined"),
        }
    }
}

/// SpeedBitCrack V3 - Multi-target Pollard's rho/kangaroo ECDLP solver for secp256k1
///
/// High-performance implementation targeting early unspent P2PK outputs (blocks 1–500k, >1 BTC)
/// and Bitcoin puzzle addresses with hybrid GPU acceleration (Vulkan/wgpu + CUDA).
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    // **General / Core**
    /// Primary operating mode
    #[arg(short = 'm', long, default_value = "full-range")]
    pub mode: SearchMode,

    /// Target public keys file (one per line)
    #[arg(short = 't', long, default_value = "test_tiny.txt")]
    pub targets: PathBuf,

    /// Path to valuable P2PK public keys file (legacy compatibility)
    #[arg(long, default_value = "valuable_p2pk_pubkeys.txt")]
    pub p2pk_file: PathBuf,

    /// Path to Bitcoin puzzles file
    #[arg(long, default_value = "puzzles.txt")]
    pub puzzles_file: PathBuf,

    /// Enable puzzle mode (append puzzles to P2PK targets)
    #[arg(long)]
    pub puzzle_mode: bool,

    /// Test mode (shrink target list to 1/10 for testing)
    #[arg(long)]
    pub test_mode: bool,

    /// Run on valuable_p2pk_pubkeys.txt (legacy compatibility)
    #[arg(long)]
    pub valuable: bool,

    /// Distinguished point bits (tradeoff: higher = fewer DPs, lower = more DPs)
    /// Recommendation: 20-24 for good collision rate without excessive DPs
    #[arg(short = 'd', long, default_value = "24")]
    pub dp_bits: usize,

    /// Kangaroo herd size (GPU memory limited)
    #[arg(short = 'H', long, default_value = "1000")]
    pub herd_size: usize,

    /// GPU kernel batch size (power of 2)
    #[arg(short = 'B', long, default_value = "131072")]
    pub gpu_batch: u32,

    /// CPU threads for hybrid/DP management
    #[arg(short = 'T', long, default_value = "256")]
    pub threads: u32,

    /// Tame kangaroo count
    #[arg(long, default_value = "100000")]
    pub tame_count: usize,

    /// Wild kangaroo count
    #[arg(long, default_value = "100000")]
    pub wild_count: usize,

    /// Maximum steps per batch
    #[arg(long, default_value = "1000")]
    pub steps_per_batch: u32,

    /// Maximum total steps before stopping
    #[arg(long, default_value = "1000000")]
    pub max_steps: u64,

    /// GOLD hierarchical modulation level
    #[arg(long, default_value = "9")]
    pub gold_mod_level: Option<u32>,

    /// Stop hunt after finding first solution
    #[arg(long)]
    pub stop_on_first_solve: bool,

    /// Max cycles before forced stop
    #[arg(long, default_value = "0")]
    pub max_cycles: u64,

    // **Testing & Validation**
    /// Run basic functionality smoke test
    #[arg(long)]
    pub basic_test: bool,

    /// Run all puzzle validation suite
    #[arg(long)]
    pub test_puzzles: bool,

    /// Validate a specific puzzle number (e.g. 66)
    #[arg(short = 'p', long)]
    pub real_puzzle: Option<u32>,
    #[arg(long, help = "Solve Puzzle #135 (first unsolved challenge)")]
    pub solve_puzzle_135: bool,

    /// Validate all target pubkeys are on-curve
    #[arg(long)]
    pub check_pubkeys: bool,

    /// Full integration test suite
    #[arg(long)]
    pub integration_test: bool,

    /// Test a known solved key (for regression)
    #[arg(long)]
    pub test_solved: Option<String>,

    // **Bias & Magic Hunting**
    /// Bias distribution mode
    #[arg(long, default_value = "uniform")]
    pub bias_mode: BiasMode,

    /// Enable advanced bias hunting
    #[arg(long)]
    pub enable_bias_hunting: bool,

    /// Use gold-cluster + bias combo
    #[arg(long)]
    pub gold_bias_combo: bool,

    /// Enable Magic 9 attractor mode
    #[arg(short = 'M', long)]
    pub magic9: bool,

    /// Prime-based entropy biasing
    #[arg(long)]
    pub prime_entropy: bool,

    /// Use expanded prime set for bias
    #[arg(long)]
    pub expanded_primes: bool,

    // **Performance & Tuning**
    /// Near-collision probability (0.0–1.0)
    #[arg(short = 'n', long, default_value = "0.0")]
    pub enable_near_collisions: f64,

    /// Enable fast k_i/d_i mathematical solving for near collisions (before BSGS)
    #[arg(long)]
    pub fast_ki_di_solving: Option<bool>,

    /// Near collision threshold (0.0-1.0, percentage of DP bits)
    #[arg(long)]
    pub near_collision_threshold: Option<u32>,

    /// Number of walk-back steps on stagnation
    #[arg(long, default_value = "0")]
    pub walk_back_steps: Option<u32>,

    /// Enable cluster-based DP pruning
    #[arg(long)]
    pub enable_smart_pruning: bool,

    /// Auto-restart herd on stagnation
    #[arg(long)]
    pub enable_stagnant_restart: bool,

    /// Dynamically adjust jump table
    #[arg(long)]
    pub enable_adaptive_jumps: bool,

    /// Enable target eviction on hopeless targets
    #[arg(long)]
    pub enable_target_eviction: bool,

    /// Enable Magic 9 attractor mode (alternative to --magic9)
    #[arg(long)]
    pub enable_magic9_attractor: bool,

    /// Use Bloom filter for DP deduplication
    #[arg(long)]
    pub use_bloom: bool,

    /// Hybrid BSGS for final solve
    #[arg(long)]
    pub use_hybrid_bsgs: bool,

    /// Switch to BSGS at this many DPs
    #[arg(long, default_value = "4294967296")]
    pub bsgs_threshold: u64,

    // **Analysis & Debug**
    /// Run full bias analysis
    #[arg(long)]
    pub bias_analysis: bool,

    /// Analyze bias from a targets file
    #[arg(long)]
    pub analyze_biases: Option<PathBuf>,

    /// Special analysis for valuable keys
    #[arg(long)]
    pub analyze_valuable_bias: bool,

    /// Analyze bias patterns for a specific puzzle
    #[arg(long)]
    pub analyze_bias: Option<u32>,

    /// Select bias components: all,mod3,mod9,mod27,mod81,gold,pop,basic (comma-separated, default: all)
    #[arg(long)]
    pub bias_components: Option<String>,

    /// Sample size for bias analysis
    #[arg(long, default_value = "10000")]
    pub sample_bias_analysis: u32,

    /// Minimum bias strength to report
    #[arg(long, default_value = "0.01")]
    pub bias_threshold: f64,

    /// Enable POP bias keyspace partitioning (histogram-based recursive reduction)
    #[arg(long)]
    pub enable_pop_partitioning: bool,

    /// Verbose output
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Laptop-optimized settings (lower memory)
    #[arg(long)]
    pub laptop: bool,

    // **Utility**
    /// Only process unsolved puzzles
    #[arg(long)]
    pub unsolved: bool,

    /// Force crack mode on unsolved
    #[arg(long)]
    pub crack_unsolved: bool,

    /// Convert valuable keys to uncompressed format
    #[arg(long)]
    pub convert_valuable_uncompressed: bool,

    /// Custom range low bound
    #[arg(long, default_value = "1")]
    pub custom_low: String,

    /// Custom range high bound
    #[arg(long, default_value = "ffffffffffffffff")]
    pub custom_high: String,

    // Legacy compatibility fields (keeping for existing code)
    /// Jump mean (expected jump size)
    #[arg(long, default_value = "1000")]
    pub jump_mean: u64,

    /// Near collision threshold (distance units, default 1000)
    #[arg(long, default_value = "1000")]
    pub near_threshold: u64,

    /// Enable multi-herd merging booster (sacred rule)
    #[arg(long)]
    pub enable_multi_herd_merge: bool,

    /// Enable DP bit feedback booster (sacred rule)
    #[arg(long)]
    pub enable_dp_feedback: bool,

    /// Near G threshold for low-order point detection (x[0] < threshold, default 2^20)
    #[arg(long, default_value = "1048576")] // 2^20 = 1,048,576
    pub near_g_thresh: u64,

    /// Maximum operations before giving up
    #[arg(long, default_value = "1000000000000")] // 10^12
    pub max_ops: u64,

    /// Primes for wild kangaroo spacing (Magic 9 cluster discovery)
    #[arg(long, num_args = 1.., default_values = ["179", "257", "347", "461", "577", "691", "797", "919"])]
    pub wild_primes: Vec<u64>,

    /// Enable prime spacing entropy (breaks Magic 9 reproducibility)
    #[arg(long)]
    pub prime_spacing_with_entropy: bool,

    /// Enable expanded prime spacing
    #[arg(long)]
    pub expanded_prime_spacing: bool,

    /// Enable expanded jump table (16-32 ops instead of 8)
    #[arg(long)]
    pub expanded_jump_table: bool,

    /// Attractor start offset for tame kangaroos
    #[arg(long)]
    pub attractor_start: Option<i64>,

    /// Enable pre-seeded POS baseline (always active per rules, but configurable weights)
    #[arg(long, num_args = 3, default_values = ["0.5", "0.25", "0.25"])]
    pub preseed_pos_weights: Vec<f64>,

    /// Path to bias log file for empirical position data
    #[arg(long)]
    pub bias_log: Option<PathBuf>,

    /// Enable noise in random proxy samples for variance
    #[arg(long)]
    pub enable_noise: bool,

    /// Enable VOW-enhanced Rho on P2PK targets
    #[arg(long)]
    pub enable_vow_rho_p2pk: bool,

    /// VOW threads for parallel solving (default 8 for high-bias puzzles)
    #[arg(long, default_value = "8")]
    pub vow_threads: usize,

    /// Poisson lambda for jump distribution (default 1.3 for high-bias puzzles)
    #[arg(long, default_value = "1.3")]
    pub poisson_lambda: f64,

    /// Path to high-priority pubkey list (from bias_analyze tool)
    #[arg(long)]
    pub priority_list: Option<PathBuf>,

    /// Validate specific puzzle (runs Tier 1 test)
    #[arg(long, value_name = "PUZZLE_NUM")]
    pub validate_puzzle: Option<u32>,

    /// Solve Magic 9 cluster (special mode)
    #[arg(long)]
    pub solve_magic_9: bool,

    /// Enable birthday paradox near collision solving
    #[arg(long)]
    pub birthday_paradox_mode: bool,

    /// Force continue on parity/checkpoint failures
    #[arg(long)]
    pub force_continue: bool,

    /// Output directory for logs and checkpoints
    #[arg(long, default_value = "output")]
    pub output_dir: PathBuf,

    /// Checkpoint interval (operations)
    #[arg(long, default_value = "4294967296")] // 2^32
    pub checkpoint_interval: u64,

    /// Log level
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// GLV dimension for endomorphism optimization
    #[clap(long, default_value_t = 2)]
    pub glv_dim: u32,

    /// Enable LLL lattice reduction
    #[clap(long, default_value_t = false)]
    pub enable_lll_reduction: bool,

    /// Enable parallel rho method
    #[clap(long, default_value_t = false)]
    pub enable_rho_parallel: bool,

    /// Enable VOW parallel method
    #[clap(long, default_value_t = false)]
    pub enable_vow_parallel: bool,
    /// Enable LLL proof simulation
    #[clap(long, default_value_t = false)]
    pub enable_lll_proof_sim: bool,

    /// Enable Babai proof simulation
    #[clap(long, default_value_t = false)]
    pub enable_babai_proof_sim: bool,

    /// Enable Babai multi simulation
    #[clap(long, default_value_t = false)]
    pub enable_babai_multi_sim: bool,

    /// Enable Fermat ECDLP
    #[clap(long, default_value_t = false)]
    pub enable_fermat_ecdlp: bool,

    /// GPU backend to use
    #[arg(long, default_value = "hybrid")]
    pub gpu_backend: GpuBackend,

    /// Enable strict linting (for CI/CD)
    #[arg(long, default_value = "false")]
    pub enable_strict_lints: bool,

    /// Enable Vulkan shader precompilation
    #[arg(long, default_value = "false")]
    pub enable_shader_precompile: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // General / Core
            mode: SearchMode::FullRange,
            targets: "pubkeys.txt".into(),
            p2pk_file: "valuable_p2pk_pubkeys.txt".into(),
            puzzles_file: "puzzles.txt".into(),
            puzzle_mode: false,
            test_mode: false,
            valuable: false,
            dp_bits: 26,
            herd_size: 500000000,
            gpu_batch: 131072,
            threads: 256,
            max_cycles: 0,
            tame_count: 1000000,
            wild_count: 1000000,
            steps_per_batch: 1000000,
            max_steps: 1000000000000,
            gold_mod_level: None,
            stop_on_first_solve: false,

            // Testing & Validation
            basic_test: false,
            test_puzzles: false,
            real_puzzle: None,
            solve_puzzle_135: false,
            check_pubkeys: false,
            integration_test: false,
            test_solved: None,

            // Bias & Magic Hunting
            bias_mode: BiasMode::Uniform,
            enable_bias_hunting: false,
            gold_bias_combo: false,
            magic9: false,
            prime_entropy: false,
            expanded_primes: false,

            // Performance & Tuning
            enable_near_collisions: 0.0,
            fast_ki_di_solving: Some(true), // Enable by default
            near_collision_threshold: Some(80), // 80% of DP bits
            walk_back_steps: Some(20000),
            enable_smart_pruning: false,
            enable_stagnant_restart: false,
            enable_adaptive_jumps: false,
            enable_target_eviction: false,
            enable_magic9_attractor: false,
            use_bloom: false,
            use_hybrid_bsgs: false,
            bsgs_threshold: 4294967296,

            // Analysis & Debug
            bias_analysis: false,
            analyze_biases: None,
            analyze_valuable_bias: false,
            analyze_bias: None,
            bias_components: None,
            sample_bias_analysis: 10000,
            bias_threshold: 0.01,
            enable_pop_partitioning: false,
            verbose: false,
            laptop: false,

            // Utility
            unsolved: false,
            crack_unsolved: false,
            convert_valuable_uncompressed: false,
            custom_low: "1".into(),
            custom_high: "ffffffffffffffff".into(),

            // Legacy compatibility
            jump_mean: 1000,
            near_threshold: 1000,
            enable_multi_herd_merge: false,
            enable_dp_feedback: false,
            near_g_thresh: 1048576,
            max_ops: 1000000000000,
            wild_primes: vec![179, 257, 347, 461, 577, 691, 797, 919],
            prime_spacing_with_entropy: false,
            expanded_prime_spacing: false,
            expanded_jump_table: false,
            attractor_start: None,

            // Additional legacy fields
            preseed_pos_weights: vec![0.5, 0.25, 0.25],
            bias_log: None,
            enable_noise: false,
            enable_vow_rho_p2pk: false,
            vow_threads: 8,
            poisson_lambda: 1.3,
            priority_list: None,
            validate_puzzle: None,
            solve_magic_9: false,
            birthday_paradox_mode: false,
            force_continue: false,
            output_dir: "output".into(),
            checkpoint_interval: 4294967296,
            log_level: "info".into(),
            glv_dim: 2,
            enable_lll_reduction: false,
            enable_rho_parallel: false,
            enable_vow_parallel: false,
            enable_lll_proof_sim: false,
            enable_babai_proof_sim: false,
            enable_babai_multi_sim: false,
            enable_fermat_ecdlp: false,
            gpu_backend: GpuBackend::Hybrid,
            enable_strict_lints: false,
            enable_shader_precompile: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchMode {
    /// Full range search (default for P2PK/Magic 9 clusters)
    FullRange,
    /// Interval search for puzzles (low-high range)
    Interval { low: u64, high: u64 },
}

// Chunk: 3070 Max-Q Config (config.rs)
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub arch: String,         // "sm_86" for Ampere
    pub max_kangaroos: usize, // 2048 for mem
    pub dp_size: usize,       // 1<<19 =512K
    pub dp_bits: u32,         // 24 for prob
    pub max_regs: i32,        // 48 for occ
    pub gpu_frac: f64,        // 0.7 for hybrid
}

pub fn laptop_3070_config() -> GpuConfig {
    GpuConfig {
        arch: "sm_86".to_string(),
        max_kangaroos: 2048,
        dp_size: 1 << 19,
        dp_bits: 24,
        max_regs: 48,
        gpu_frac: 0.7,
    }
}

impl Config {
    /// Parse command line arguments into Config
    pub fn parse() -> Result<Self> {
        let config = <Self as Parser>::parse();
        Ok(config)
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate DP bits
        if !(20..=32).contains(&self.dp_bits) {
            return Err(anyhow!(
                "DP bits must be between 20 and 32, got {}",
                self.dp_bits
            ));
        }

        // Validate herd size
        if self.herd_size == 0 {
            return Err(anyhow!("Herd size must be > 0"));
        }

        // Validate jump mean
        if self.jump_mean == 0 {
            return Err(anyhow!("Jump mean must be > 0"));
        }

        // GPU backend is now validated by FromStr enum, always valid
        // No additional validation needed

        // Validate near collision threshold
        if self.enable_near_collisions < 0.0 || self.enable_near_collisions > 1.0 {
            return Err(anyhow!(
                "Near collision threshold must be between 0.0 and 1.0, got {}",
                self.enable_near_collisions
            ));
        }

        // Validate walk backs steps
        if let Some(steps) = self.walk_back_steps {
            if steps > 0 && steps < 1000 {
                return Err(anyhow!(
                    "Walk backs steps must be 0 (disabled) or >= 1000, got {}",
                    steps
                ));
            }
        }

        // Validate puzzle number
        if let Some(puzzle_num) = self.validate_puzzle {
            if !(64..=160).contains(&puzzle_num) {
                return Err(anyhow!("Puzzle number must be between 64 and 160"));
            }
        }

        // Validate bias mode (no-op, as enum ensures)

        // Validate BSGS threshold
        if self.use_hybrid_bsgs && self.bsgs_threshold == 0 {
            return Err(anyhow!("BSGS threshold must be > 0 when enabled"));
        }

        // Warn if bloom with low dp_bits (density risk)
        if self.use_bloom && self.dp_bits < 20 {
            log::warn!(
                "Bloom enabled with low dp_bits ({}); may cause high false positives",
                self.dp_bits
            );
        }

        // Validate gold combo requires a non-uniform bias mode
        if self.gold_bias_combo && matches!(self.bias_mode, BiasMode::Uniform) {
            return Err(anyhow!(
                "GOLD bias combo requires a non-uniform bias mode (try --bias-mode magic9 or --bias-mode primes)"
            ));
        }

        Ok(())
    }
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::FullRange
    }
}

impl std::str::FromStr for SearchMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "full-range" => Ok(SearchMode::FullRange),
            _ if s.starts_with("interval=") => {
                let parts: Vec<&str> = s.split('=').collect();
                if parts.len() != 2 {
                    return Err(anyhow!("Invalid interval format: {}", s));
                }
                let range: Vec<&str> = parts[1].split('-').collect();
                if range.len() != 2 {
                    return Err(anyhow!("Invalid interval format: {}", s));
                }
                let low = range[0].parse()?;
                let high = range[1].parse()?;
                Ok(SearchMode::Interval { low, high })
            }
            _ => Err(anyhow!("Unknown search mode: {}", s)),
        }
    }
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchMode::FullRange => write!(f, "full-range"),
            SearchMode::Interval { low, high } => write!(f, "interval={}-{}", low, high),
        }
    }
}

/// Enable NVIDIA persistence mode for stable GPU performance
/// Only effective on Linux systems with NVIDIA GPUs
pub fn enable_nvidia_persistence() -> Result<bool> {
    use std::process::{Command, Output};

    // Only attempt on Linux systems
    if !cfg!(target_os = "linux") {
        return Ok(false);
    }

    // Check if nvidia-smi exists before attempting
    if std::process::Command::new("which")
        .arg("nvidia-smi")
        .output()
        .is_err()
    {
        return Ok(false);
    }

    // Enable persistence mode
    let set_output: Output = Command::new("nvidia-smi")
        .arg("-pm")
        .arg("1")
        .output()
        .map_err(|e| anyhow!("Failed to run nvidia-smi: {}", e))?;

    if !set_output.status.success() {
        return Err(anyhow!(
            "Failed to enable persistence: {}",
            String::from_utf8_lossy(&set_output.stderr)
        ));
    }

    // Query persistence status to verify
    let query_output: Output = Command::new("nvidia-smi")
        .arg("-q")
        .arg("-d")
        .arg("PERSISTENCE")
        .output()
        .map_err(|e| anyhow!("Failed to query persistence: {}", e))?;

    if !query_output.status.success() {
        return Err(anyhow!(
            "Failed to query persistence: {}",
            String::from_utf8_lossy(&query_output.stderr)
        ));
    }

    let status = String::from_utf8_lossy(&query_output.stdout);
    Ok(status.contains("Enabled"))
}

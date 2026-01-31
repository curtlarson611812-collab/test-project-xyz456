//! Configuration module for SpeedBitCrack V3
//!
//! clap::Parser struct with default values (dp-bits=24, primes list, jump_mean, etc.),
//! and validation logic.

use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// SpeedBitCrack V3 - Pollard's rho/kangaroo ECDLP solver for secp256k1
#[derive(Parser, Debug, Clone, Serialize, Deserialize, Default)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// Search mode: full-range (default for P2PK/Magic 9) or interval=low-high (for puzzles)
    #[arg(long, default_value = "full-range")]
    pub mode: SearchMode,

    /// Path to valuable P2PK public keys file
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

    /// DP bits for distinguished points (default 24)
    #[arg(long, default_value = "24")]
    pub dp_bits: usize,

    /// Herd size (number of kangaroos)
    #[arg(long, default_value = "100000")]
    pub herd_size: usize,

    /// Jump mean (expected jump size)
    #[arg(long, default_value = "1000")]
    pub jump_mean: u64,

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

    /// Enable near collision detection (threshold 0.75-0.85)
    #[arg(long, value_name = "THRESHOLD")]
    pub enable_near_collisions: Option<f64>,

    /// Enable walk backs/forwards (max steps)
    #[arg(long, value_name = "STEPS")]
    pub enable_walk_backs: Option<u64>,

    /// Enable smart DP pruning (combo:bloom-value-cluster)
    #[arg(long)]
    pub enable_smart_pruning: bool,

    /// Enable target eviction on hopeless targets
    #[arg(long)]
    pub enable_target_eviction: bool,

    /// Validate specific puzzle (runs Tier 1 test)
    #[arg(long, value_name = "PUZZLE_NUM")]
    pub validate_puzzle: Option<u32>,

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

    /// GPU backend to use (cuda, vulkan, hybrid, cpu)
    #[arg(long, default_value = "cuda")]
    pub gpu_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    /// Full range search (default for P2PK/Magic 9 clusters)
    FullRange,
    /// Interval search for puzzles (low-high range)
    Interval { low: u64, high: u64 },
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
            return Err(anyhow!("DP bits must be between 20 and 32, got {}", self.dp_bits));
        }

        // Validate herd size
        if self.herd_size == 0 {
            return Err(anyhow!("Herd size must be > 0"));
        }

        // Validate jump mean
        if self.jump_mean == 0 {
            return Err(anyhow!("Jump mean must be > 0"));
        }

        // Validate GPU backend
        if !["hybrid", "cuda", "vulkan", "cpu"].contains(&self.gpu_backend.as_str()) {
            return Err(anyhow!("Invalid GPU backend: {}. Must be one of: hybrid, cuda, vulkan, cpu", self.gpu_backend));
        }

        // Validate near collision threshold
        if let Some(threshold) = self.enable_near_collisions {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(anyhow!("Near collision threshold must be between 0.0 and 1.0"));
            }
        }

        // Validate walk backs steps
        if let Some(steps) = self.enable_walk_backs {
            if steps == 0 {
                return Err(anyhow!("Walk backs steps must be > 0"));
            }
        }

        // Validate puzzle number
        if let Some(puzzle_num) = self.validate_puzzle {
            if !(64..=160).contains(&puzzle_num) {
                return Err(anyhow!("Puzzle number must be between 64 and 160"));
            }
        }

        // Validate GPU backend
        if !["hybrid", "cuda", "vulkan", "cpu"].contains(&self.gpu_backend.as_str()) {
            return Err(anyhow!("Invalid GPU backend: {}. Must be one of: hybrid, cuda, vulkan, cpu", self.gpu_backend));
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
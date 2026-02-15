//! Target loading and parsing
//!
//! Load & parse valuable_p2pk_publickey.txt + puzzles.txt, validate pubkeys

use crate::types::Target;
use crate::config::Config;
use crate::math::Secp256k1;
use anyhow::{anyhow, Result};
use log::{info, warn};
use std::fs;
use std::path::Path;

/// Target loader for P2PK addresses and puzzles
pub struct TargetLoader {
    curve: Secp256k1,
}

impl TargetLoader {
    /// Create new target loader
    pub fn new() -> Self {
        let curve = Secp256k1::new();
        TargetLoader {
            curve,
        }
    }

    /// Load all targets based on configuration - ALWAYS load FULL valuable_p2pk_publickey.txt
    pub fn load_targets(&self, config: &Config) -> Result<Vec<Target>> {
        println!("DEBUG: ENTERING load_targets method");
        println!("DEBUG: load_targets called with puzzle_mode: {}, test_mode: {}", config.puzzle_mode, config.test_mode);
        println!("DEBUG: Current working directory: {:?}", std::env::current_dir().unwrap_or_default());
        println!("DEBUG: Target file path: {:?}", config.targets);
        let mut targets = Vec::new();

        // Load P2PK targets - ALWAYS load the FULL file (~34,353 verified P2PK pubkeys)
        // NO shrinking to 1/10/test keys unless --test-mode flag is explicitly set
        // Use the --targets flag value for the main target file, with fallback
        let target_file = if config.targets.exists() {
            config.targets.clone()
        } else {
            let fallback = std::path::PathBuf::from("valuable_p2pk_pubkeys.txt");
            if fallback.exists() {
                println!("DEBUG: Specified target file '{}' not found, using fallback '{}'", config.targets.display(), fallback.display());
                fallback
            } else {
                config.targets.clone()
            }
        };

        println!("DEBUG: Using target file '{}' exists: {}", target_file.display(), target_file.exists());
        if target_file.exists() {
            println!("DEBUG: Loading targets from {}", target_file.display());
            let p2pk_targets = self.load_p2pk_targets(&target_file)?;
            let p2pk_count = p2pk_targets.len();
            targets.extend(p2pk_targets);
            println!("DEBUG: Loaded {} targets", p2pk_count);
        } else if config.mode == crate::config::SearchMode::FullRange && !config.test_mode && !config.puzzle_mode {
            return Err(anyhow!("Target file not found: {} (required for full-range mode)", target_file.display()));
        } else {
            println!("DEBUG: Skipping targets (puzzle mode, test mode, or file not found)");
        }

        // Load puzzle targets if enabled - append to P2PK targets
        println!("DEBUG: Puzzle mode enabled: {}", config.puzzle_mode);
        if config.puzzle_mode {
            println!("DEBUG: Puzzle file '{}' exists: {}", config.puzzles_file.display(), config.puzzles_file.exists());
            if config.puzzles_file.exists() {
                let puzzle_targets = self.load_puzzle_targets(&config.puzzles_file)?;
                targets.extend(puzzle_targets);
            } else {
                return Err(anyhow!("Puzzle file not found: {} (required for puzzle mode)", config.puzzles_file.display()));
            }
        }

        // Apply test mode filtering - ONLY when explicitly requested
        if config.test_mode {
            warn!("TEST MODE: Limiting targets to 100 for testing (rule violation if used in production)");
            targets.truncate(100);
        }

        // Sort by priority (higher value first)
        targets.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        info!("Loaded {} targets total ({} P2PK, {} puzzles)",
              targets.len(),
              targets.iter().filter(|t| t.key_range.is_none()).count(),
              targets.iter().filter(|t| t.key_range.is_some()).count());

        Ok(targets)
    }

    /// Load P2PK targets from valuable_p2pk_publickey.txt
    fn load_p2pk_targets(&self, file_path: &Path) -> Result<Vec<Target>> {
        info!("Loading P2PK targets from {}", file_path.display());
        let content = fs::read_to_string(file_path)?;
        let mut targets = Vec::new();
        let mut invalid_count = 0;

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match self.parse_p2pk_line(line, line_num + 1) {
                Ok(target) => targets.push(target),
                Err(e) => {
                    warn!("Skipping invalid P2PK line {}: {}", line_num + 1, e);
                    invalid_count += 1;
                }
            }
        }

        info!("Loaded {} valid P2PK targets from {} (skipped {} invalid)",
              targets.len(), file_path.display(), invalid_count);
        Ok(targets)
    }

    /// Parse P2PK target line: pubkey_hex,btc_value,address
    fn parse_p2pk_line(&self, line: &str, line_num: usize) -> Result<Target> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            return Err(anyhow!("Invalid P2PK line format (expected pubkey,btc_value[,address]): {}", line));
        }

        let pubkey_hex = parts[0].trim();
        let btc_value = parts[1].trim().parse::<f64>()
            .map_err(|e| anyhow!("Invalid BTC value '{}': {}", parts[1], e))?;
        let address = parts.get(2).map(|s| s.trim().to_string());

        // Parse compressed public key (33 bytes)
        let pubkey_bytes = hex::decode(pubkey_hex)
            .map_err(|e| anyhow!("Invalid hex pubkey '{}': {}", pubkey_hex, e))?;

        if pubkey_bytes.len() != 33 {
            return Err(anyhow!("Invalid compressed pubkey length: {} (expected 33)", pubkey_bytes.len()));
        }

        // Convert to fixed-size array
        let pubkey_array: [u8; 33] = pubkey_bytes.as_slice().try_into()
            .map_err(|_| anyhow!("Invalid pubkey length: expected 33 bytes, got {}", pubkey_bytes.len()))?;

        // Decompress to validate and get affine point
        let point = self.curve.decompress_point(&pubkey_array)
            .ok_or_else(|| anyhow!("Failed to decompress pubkey (not on curve)"))?;

        // Additional validation - ensure point is on curve
        if !self.curve.is_on_curve(&point) {
            return Err(anyhow!("Decompressed point is not on secp256k1 curve"));
        }

        Ok(Target {
            point,
            key_range: None, // P2PK has no specific key range
            id: line_num as u64,
            priority: btc_value, // Higher BTC value = higher priority
            address,
            value_btc: Some(btc_value),
        })
    }

    /// Load puzzle targets from puzzles.txt using the puzzle module
    pub fn load_puzzle_targets(&self, file_path: &Path) -> Result<Vec<Target>> {
        use crate::puzzles;

        info!("Loading puzzle targets from {}", file_path.display());

        // Load puzzles from file
        let puzzles = puzzles::load_puzzles_from_file()?;
        let mut targets = Vec::new();

        for puzzle in &puzzles {
            // Skip puzzles without public keys (unsolved sequential puzzles)
            if puzzle.pub_key_hex.is_empty() {
                info!("Skipping puzzle {}: no public key available (unsolved sequential)", puzzle.n);
                continue;
            }

            // Validate compressed public key format
            let pubkey_hex = puzzle.pub_key_hex.trim();
            if pubkey_hex.len() != 66 || (!pubkey_hex.starts_with("02") && !pubkey_hex.starts_with("03")) {
                warn!("Invalid compressed pubkey format for puzzle {}: expected 66 hex chars starting with 02 or 03", puzzle.n);
                continue;
            }

            // Parse compressed public key (33 bytes)
            let pubkey_bytes = hex::decode(pubkey_hex)
                .map_err(|e| anyhow!("Invalid hex pubkey for puzzle {} '{}': {}", puzzle.n, pubkey_hex, e))?;

            if pubkey_bytes.len() != 33 {
                warn!("Invalid compressed pubkey length for puzzle {}: {} (expected 33)", puzzle.n, pubkey_bytes.len());
                continue;
            }

            // Convert to fixed-size array
            let pubkey_array: [u8; 33] = pubkey_bytes.as_slice().try_into()
                .map_err(|_| anyhow!("Invalid pubkey length for puzzle {}: expected 33 bytes, got {}", puzzle.n, pubkey_bytes.len()))?;

            // Decompress to validate and get affine point
            let point = self.curve.decompress_point(&pubkey_array)
                .ok_or_else(|| anyhow!("Failed to decompress pubkey for puzzle {} (not on curve)", puzzle.n))?;

            // Additional validation - ensure point is on curve
            if !self.curve.is_on_curve(&point) {
                warn!("Decompressed point for puzzle {} is not on secp256k1 curve", puzzle.n);
                continue;
            }

            // Validate key range
            if puzzle.range_min >= puzzle.range_max {
                warn!("Invalid key range for puzzle {}: min ({}) >= max ({})", puzzle.n, puzzle.range_min.to_hex(), puzzle.range_max.to_hex());
                continue;
            }

            // For now, we'll use None for key_range since Target expects (u64, u64) but we have BigInt256
            // TODO: Update Target struct to support BigInt256 ranges for large puzzles
            let key_range = None;

            // Calculate priority: BTC value + bonus for lower puzzle numbers
            let priority = puzzle.btc_reward + (1000.0 / puzzle.n as f64);

            targets.push(Target {
                point,
                key_range,
                id: puzzle.n as u64,
                priority,
                address: Some(puzzle.target_address.clone()),
                value_btc: Some(puzzle.btc_reward),
            });
        }

        info!("Loaded {} valid puzzle targets from {} (total puzzles: {})",
              targets.len(), file_path.display(), puzzles.len());

        Ok(targets)
    }

    /// Validate all target points are on curve
    pub fn validate_targets(&self, targets: &[Target]) -> Result<Vec<Target>> {
        let mut valid_targets = Vec::new();
        let mut invalid_count = 0;

        for target in targets {
            if self.curve.is_on_curve(&target.point) {
                valid_targets.push(target.clone());
            } else {
                warn!("Target {} point is not on secp256k1 curve - skipping", target.id);
                invalid_count += 1;
            }
        }

        if invalid_count > 0 {
            warn!("Skipped {} invalid targets (not on curve)", invalid_count);
        }

        Ok(valid_targets)
    }

    /// Get comprehensive target statistics
    pub fn get_target_stats(&self, targets: &[Target]) -> TargetStats {
        let total_value: f64 = targets.iter()
            .filter_map(|t| t.value_btc)
            .sum();

        let p2pk_count = targets.iter()
            .filter(|t| t.key_range.is_none())
            .count();

        let puzzle_count = targets.iter()
            .filter(|t| t.key_range.is_some())
            .count();

        let high_value_count = targets.iter()
            .filter(|t| t.value_btc.unwrap_or(0.0) >= 1.0)
            .count();

        let max_value = targets.iter()
            .filter_map(|t| t.value_btc)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        TargetStats {
            total_targets: targets.len(),
            p2pk_targets: p2pk_count,
            puzzle_targets: puzzle_count,
            high_value_targets: high_value_count,
            total_btc_value: total_value,
            max_btc_value: max_value,
            avg_priority: targets.iter().map(|t| t.priority).sum::<f64>() / targets.len() as f64,
        }
    }
}

/// Comprehensive target loading statistics
#[derive(Debug, Clone)]
pub struct TargetStats {
    pub total_targets: usize,
    pub p2pk_targets: usize,
    pub puzzle_targets: usize,
    pub high_value_targets: usize, // >= 1 BTC
    pub total_btc_value: f64,
    pub max_btc_value: f64,
    pub avg_priority: f64,
}

impl std::fmt::Display for TargetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Targets: {} total ({} P2PK, {} puzzles, {} high-value), {:.2} BTC total (max {:.2}), avg priority {:.2}",
               self.total_targets, self.p2pk_targets, self.puzzle_targets, self.high_value_targets,
               self.total_btc_value, self.max_btc_value, self.avg_priority)
    }
}
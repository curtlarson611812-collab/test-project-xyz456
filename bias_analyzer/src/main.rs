use clap::Parser;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use k256::{AffinePoint};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::group::GroupEncoding;
use k256::elliptic_curve::point::AffineCoordinates;
use rayon::prelude::*;

/// Custom BigInt256 implementation for bias analysis
#[derive(Clone, Debug)]
struct BigInt256 {
    limbs: [u64; 4], // limbs[0] is least significant
}

impl BigInt256 {
    fn from_u64(value: u64) -> Self {
        BigInt256 {
            limbs: [value, 0, 0, 0],
        }
    }

    fn to_u64(&self) -> u64 {
        self.limbs[0]
    }
}

/// Convert big-endian bytes to BigInt256
fn bytes_to_bigint256(bytes: &[u8]) -> BigInt256 {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i < 4 {
            let mut limb_bytes = [0u8; 8];
            limb_bytes[..chunk.len()].copy_from_slice(chunk);
            // Convert from big-endian to little-endian for limbs
            limbs[3-i] = u64::from_be_bytes(limb_bytes);
        }
    }
    BigInt256 { limbs }
}

/// Compute (a % modulus) for small moduli
fn bigint_mod_u64(a: &BigInt256, modulus: u64) -> u64 {
    // For small moduli, we can do Barrett reduction or simple division
    // Since modulus is small (< 100), we can use a simple approach

    // Start with the most significant limb
    let mut remainder = 0u128;

    for &limb in a.limbs.iter().rev() { // Process from most significant
        remainder = ((remainder << 64) | limb as u128) % modulus as u128;
    }

    remainder as u64
}

/// Standalone bias analysis tool for SpeedBitCrack V3
/// Analyzes valuable_p2pk_pubkeys.txt for statistical biases
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Input pubkey file (e.g., valuable_p2pk_pubkeys.txt)
    #[arg(short, long)]
    input: String,

    /// Output file for high-priority pubkey list
    #[arg(long, default_value = "high_priority_list.txt")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🔍 SpeedBitCrack V3 - Standalone Bias Analysis Tool");
    println!("==================================================");

    println!("📂 Loading pubkeys from {}...", args.input);
    let pubkeys = load_pubkeys_parallel(&args.input)?;
    println!("✅ Loaded {} pubkeys", pubkeys.len());

    if pubkeys.is_empty() {
        println!("❌ No valid pubkeys found");
        return Ok(());
    }

    println!("🎯 Computing REAL statistical biases for all {} pubkeys...", pubkeys.len());

    // Convert pubkeys to hex strings for modular analysis
    let pubkey_hexes: Vec<String> = pubkeys.par_iter()
        .map(|point| hex::encode(point.to_encoded_point(false).as_bytes()))
        .collect();

    // Compute modular bias statistics using REAL statistical analysis
    println!("📊 Computing mod3 bias statistics...");
    let mod3_counts = compute_modular_counts(&pubkeys, 3);
    println!("   Mod3 distribution: {:?}", mod3_counts);

    println!("📊 Computing mod9 bias statistics...");
    let mod9_counts = compute_modular_counts(&pubkeys, 9);
    println!("   Mod9 distribution: {:?}", mod9_counts);

    println!("📊 Computing mod27 bias statistics...");
    let mod27_counts = compute_modular_counts(&pubkeys, 27);

    println!("📊 Computing mod81 bias statistics...");
    let mod81_counts = compute_modular_counts(&pubkeys, 81);

    println!("🎯 Computing individual bias scores...");

    // Compute individual bias scores for each pubkey
    let bias_scores: Vec<(String, f64, usize)> = pubkeys.par_iter().enumerate().map(|(i, point)| {
        let hex = hex::encode(point.to_encoded_point(false).as_bytes());

        // Calculate modular residues using full 256-bit x coordinate
        let x_bytes = point.x();
        let x = bytes_to_bigint256(&x_bytes);
        let mod3_residue = bigint_mod_u64(&x, 3) as usize;
        let mod9_residue = bigint_mod_u64(&x, 9) as usize;
        let mod27_residue = bigint_mod_u64(&x, 27) as usize;
        let mod81_residue = bigint_mod_u64(&x, 81) as usize;

        // Compute bias score based on statistical deviation from uniform distribution
        // Higher scores indicate higher bias (more deviation from expected)
        let mut score = 0.0;

        // Mod3 contribution - chi-squared style scoring
        let mod3_expected = pubkey_hexes.len() as f64 / 3.0;
        let mod3_observed = mod3_counts[mod3_residue] as f64;
        let mod3_chi = (mod3_observed - mod3_expected).powi(2) / mod3_expected;
        score += mod3_chi * 0.25;

        // Mod9 contribution
        let mod9_expected = pubkey_hexes.len() as f64 / 9.0;
        let mod9_observed = mod9_counts[mod9_residue] as f64;
        let mod9_chi = (mod9_observed - mod9_expected).powi(2) / mod9_expected;
        score += mod9_chi * 0.25;

        // Mod27 contribution
        let mod27_expected = pubkey_hexes.len() as f64 / 27.0;
        let mod27_observed = mod27_counts[mod27_residue] as f64;
        let mod27_chi = (mod27_observed - mod27_expected).powi(2) / mod27_expected;
        score += mod27_chi * 0.25;

        // Mod81 contribution (GOLD cluster detection) - highest weight
        let mod81_expected = pubkey_hexes.len() as f64 / 81.0;
        let mod81_observed = mod81_counts[mod81_residue] as f64;
        let mod81_chi = (mod81_observed - mod81_expected).powi(2) / mod81_expected;
        score += mod81_chi * 0.25;

        // GOLD cluster bonus (addresses with mod81=0 get massive bonus)
        let is_gold = mod81_residue == 0;
        if is_gold {
            score *= 4.0; // Quadruple score for GOLD cluster addresses
        }

        // Normalize to 0-1 range (rough approximation)
        score = (score / 10.0).min(1.0);

        if i < 5 { // Debug first few
            println!("   Pubkey {}: mod3={}, mod9={}, mod27={}, mod81={}, score={:.3}, gold={}",
                    i, mod3_residue, mod9_residue, mod27_residue, mod81_residue, score, is_gold);
        }

        (hex, score, if is_gold { 1 } else { 0 })
    }).collect();

    // Sort by score descending
    let mut sorted_scores = bias_scores;
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Extract GOLD cluster addresses (mod81=0)
    let gold_cluster: Vec<String> = sorted_scores.iter()
        .filter(|(_, _, is_gold)| *is_gold == 1)
        .map(|(hex, score, _)| {
            println!("  🏆 GOLD Cluster: {:.2}x - {}", score, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Extract top 100 highest bias addresses
    let top_100: Vec<String> = sorted_scores.iter()
        .take(100)
        .map(|(hex, score, is_gold)| {
            let marker = if *is_gold == 1 { " [GOLD]" } else { "" };
            println!("  🎯 Top 100: {:.2}x{} - {}", score, marker, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Extract high bias targets (above threshold)
    let high_bias: Vec<String> = sorted_scores.iter()
        .filter(|&(_, score, _)| *score > 2.0)
        .map(|(hex, score, _)| {
            println!("  🎯 High Bias: {:.2}x - {}", score, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Statistics
    let avg_score = sorted_scores.iter().map(|(_, s, _)| s).sum::<f64>() / sorted_scores.len() as f64;
    let max_score = sorted_scores.first().map(|(_, s, _)| *s).unwrap_or(0.0);
    let gold_percentage = (gold_cluster.len() as f64 / sorted_scores.len() as f64) * 100.0;
    let high_percentage = (high_bias.len() as f64 / sorted_scores.len() as f64) * 100.0;

    println!("\n📊 Analysis Statistics:");
    println!("  📈 Average bias score: {:.3}x", avg_score);
    println!("  🎯 Maximum bias score: {:.3}x", max_score);
    println!("  🏆 GOLD cluster addresses: {} ({:.2}%)", gold_cluster.len(), gold_percentage);
    println!("  🎯 High bias targets: {} ({:.2}%)", high_bias.len(), high_percentage);
    println!("  🥇 Top 100 addresses: {}", top_100.len());

    // Write results
    write_priority_list(&format!("{}_gold.txt", args.output.trim_end_matches(".txt")), &gold_cluster)?;
    write_priority_list(&format!("{}_top100.txt", args.output.trim_end_matches(".txt")), &top_100)?;
    write_priority_list(&args.output, &high_bias)?;

    println!("\n✅ Bias analysis complete!");
    println!("  🏆 GOLD cluster: {} addresses", gold_cluster.len());
    println!("  🥇 Top 100: {} addresses", top_100.len());
    println!("  🎯 High bias: {} addresses", high_bias.len());

    Ok(())
}

/// Compute counts for each modular residue
fn compute_modular_counts(pubkeys: &[AffinePoint], modulus: u64) -> Vec<usize> {
    let mut counts = vec![0; modulus as usize];
    let mut processed = 0;

    for point in pubkeys {
        let x_bytes = point.x();
        let x = bytes_to_bigint256(&x_bytes);
        let residue = bigint_mod_u64(&x, modulus) as usize;
        if residue < counts.len() {
            counts[residue] += 1;
            processed += 1;
        }
    }

    println!("   Processed {} pubkeys for mod{}", processed, modulus);
    counts
}

/// Load pubkeys from file
fn load_pubkeys_parallel(path: &str) -> Result<Vec<AffinePoint>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines()
        .filter_map(|line| line.ok())
        .collect();

    let pubkeys: Vec<AffinePoint> = lines
        .par_iter()
        .filter_map(|line| {
            let hex_str = line.trim();
            if hex_str.is_empty() || hex_str.starts_with('#') {
                return None;
            }

            match hex::decode(hex_str) {
                Ok(bytes) => {
                    if bytes.len() == 33 || bytes.len() == 65 {
                        AffinePoint::from_bytes(bytes.as_slice().try_into().unwrap()).into_option()
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        })
        .collect();

    Ok(pubkeys)
}

/// Write priority list to file
fn write_priority_list(path: &str, pubkeys: &[String]) -> Result<()> {
    let mut file = File::create(path)?;
    let list_type = if path.contains("gold") {
        "GOLD Cluster Addresses"
    } else if path.contains("top100") {
        "Top 100 High-Bias Addresses"
    } else {
        "High-Bias Priority Addresses"
    };

    writeln!(file, "# {}", list_type)?;
    writeln!(file, "# Generated by SpeedBitCrack V3 standalone bias analyzer")?;
    writeln!(file, "# Statistical bias analysis for enhanced kangaroo efficiency")?;
    writeln!(file, "# Total addresses: {}", pubkeys.len())?;
    writeln!(file, "# Analysis method: Real modular bias statistics (mod3/9/27/81)")?;
    writeln!(file, "# GOLD cluster detection: mod81=0 pattern recognition")?;
    writeln!(file)?;

    for (i, pubkey) in pubkeys.iter().enumerate() {
        writeln!(file, "{:4}: {}", i + 1, pubkey)?;
    }

    Ok(())
}
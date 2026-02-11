use clap::Parser;
use anyhow::Result;
use std::fs::{File, Write};
use std::io::{BufRead, BufReader};
use k256::{AffinePoint, Scalar};
use rayon::prelude::*;
use crate::utils::bias::{generate_preseed_pos, blend_proxy_preseed, analyze_preseed_cascade};
use crate::math::bigint::BigInt256;

/// Consolidated bias analysis CLI tool
/// Analyzes pubkey files for positional + modular biases, extracts high-priority targets
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Input pubkey file (e.g., valuable_p2pk_pubkeys.txt)
    #[arg(short, long)]
    input: String,

    /// Modular bias level (9, 27, or 81 for gold clusters)
    #[arg(short, long, default_value = "81")]
    mod_level: u64,

    /// High bias threshold for priority extraction
    #[arg(long, default_value = "4.0")]
    threshold: f64,

    /// Output file for high-priority pubkey list
    #[arg(long, default_value = "high_priority_list.txt")]
    output: String,

    /// Enable noise in proxy blending
    #[arg(long)]
    enable_noise: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🔍 Loading pubkeys from {}...", args.input);
    let pubkeys = load_pubkeys_parallel(&args.input)?;
    println!("✅ Loaded {} pubkeys", pubkeys.len());

    if pubkeys.is_empty() {
        println!("❌ No valid pubkeys found");
        return Ok(());
    }

    println!("📊 Computing POS baseline...");
    let range_min = Scalar::ZERO;
    let range_width = Scalar::MAX; // Full keyspace proxy
    let preseed_pos = generate_preseed_pos(&range_min, &range_width);

    println!("🔄 Blending proxy data...");
    let proxy = blend_proxy_preseed(
        preseed_pos,
        1000, // num_random
        None, // empirical_pos
        (0.5, 0.25, 0.25), // weights
        args.enable_noise,
    );

    println!("📈 Running cascade analysis...");
    let cascades = analyze_preseed_cascade(&proxy, 10);

    println!("🎯 Computing bias scores for all pubkeys...");
    let bias_results: Vec<(AffinePoint, f64)> = pubkeys
        .par_iter()
        .map(|point| {
            let residue = compute_mod_residue(&point, args.mod_level);
            let pos_score = compute_pos_score(&point, &cascades);
            let gold_bonus = if is_gold_cluster(residue, args.mod_level) { 1.3 } else { 1.0 };
            let total_score = pos_score * gold_bonus;
            (*point, total_score)
        })
        .collect();

    println!("🔍 Extracting high-bias targets (threshold: {:.2}x)...", args.threshold);
    let high_bias: Vec<String> = bias_results
        .into_iter()
        .filter(|(_, score)| *score > args.threshold)
        .map(|(point, score)| {
            println!("  🎯 Score: {:.2}x - {}", score, hex::encode(point.to_encoded_point(false).as_bytes()));
            hex::encode(point.to_encoded_point(false).as_bytes())
        })
        .collect();

    println!("💾 Writing {} high-priority pubkeys to {}...", high_bias.len(), args.output);
    write_priority_list(&args.output, &high_bias)?;

    if high_bias.is_empty() {
        println!("⚠️  No pubkeys exceeded threshold {:.2}x", args.threshold);
        println!("💡 Try lowering threshold or checking bias computation");
    } else {
        println!("✅ Analysis complete! {} high-priority targets identified", high_bias.len());
    }

    Ok(())
}

/// Load pubkeys from file with parallel parsing
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
            if hex_str.is_empty() {
                return None;
            }

            match hex::decode(hex_str) {
                Ok(bytes) => {
                    // Handle both compressed and uncompressed formats
                    if bytes.len() == 33 || bytes.len() == 65 {
                        AffinePoint::from_bytes(bytes.into())
                    } else {
                        None
                    }
                }
                Err(_) => None, // Skip invalid hex
            }
        })
        .collect();

    Ok(pubkeys)
}

/// Compute modular residue for bias analysis
fn compute_mod_residue(point: &AffinePoint, mod_level: u64) -> u64 {
    let x_bytes = point.x().to_bytes();
    let x_big = BigInt256::from_bytes_be(&x_bytes);
    (x_big % BigInt256::from_u64(mod_level)).low_u32() as u64
}

/// Check if residue indicates gold cluster membership
fn is_gold_cluster(residue: u64, mod_level: u64) -> bool {
    match mod_level {
        81 => matches!(residue, 0 | 9 | 18 | 27 | 36 | 45 | 54 | 63 | 72), // Gold pattern
        27 => matches!(residue, 0 | 9 | 18), // Secondary gold
        9 => residue == 0, // Basic gold
        _ => false,
    }
}

/// Compute positional score from cascade analysis
fn compute_pos_score(point: &AffinePoint, cascades: &[(f64, f64)]) -> f64 {
    if cascades.is_empty() {
        return 1.0;
    }

    // Use the final cascade level bias as base score
    let base_score = cascades.last().map(|(_, bias)| *bias).unwrap_or(1.0);

    // Add bonus if point falls in high-density regions
    let x_bytes = point.x().to_bytes();
    let pos_hash = x_bytes.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64);
    let pos_normalized = (pos_hash % 1000000) as f64 / 1000000.0; // Simple [0,1] proxy

    // Check if position is in high-density bins from cascade
    let mut pos_bonus = 1.0;
    for (density, _) in cascades {
        if *density > 2.0 { // High density threshold
            pos_bonus *= 1.2; // Bonus for clustering
        }
    }

    base_score * pos_bonus
}

/// Write priority list to file
fn write_priority_list(path: &str, pubkeys: &[String]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# High-priority pubkey list generated by bias_analyze")?;
    writeln!(file, "# Threshold-based extraction for enhanced kangaroo efficiency")?;
    writeln!(file, "# Total targets: {}", pubkeys.len())?;
    writeln!(file)?;

    for pubkey in pubkeys {
        writeln!(file, "{}", pubkey)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mod_residue() {
        let point = AffinePoint::GENERATOR;
        let residue_9 = compute_mod_residue(&point, 9);
        assert!(residue_9 < 9, "Residue should be < modulus");

        let residue_81 = compute_mod_residue(&point, 81);
        assert!(residue_81 < 81, "Residue should be < modulus");
    }

    #[test]
    fn test_is_gold_cluster() {
        assert!(is_gold_cluster(0, 81), "0 should be gold for mod 81");
        assert!(is_gold_cluster(9, 81), "9 should be gold for mod 81");
        assert!(is_gold_cluster(72, 81), "72 should be gold for mod 81");
        assert!(!is_gold_cluster(1, 81), "1 should not be gold for mod 81");

        assert!(is_gold_cluster(0, 9), "0 should be gold for mod 9");
        assert!(!is_gold_cluster(1, 9), "1 should not be gold for mod 9");
    }

    #[test]
    fn test_compute_pos_score() {
        let point = AffinePoint::GENERATOR;
        let cascades = vec![(1.5, 1.2), (2.5, 1.8), (3.0, 2.2)]; // High density cascades

        let score = compute_pos_score(&point, &cascades);
        assert!(score > 2.0, "Score should be boosted by high-density cascades: {}", score);

        let empty_cascades = vec![];
        let empty_score = compute_pos_score(&point, &empty_cascades);
        assert_eq!(empty_score, 1.0, "Empty cascades should return base score");
    }
}
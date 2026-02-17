use clap::Parser;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use k256::{AffinePoint, Scalar};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::group::GroupEncoding;
use k256::elliptic_curve::point::AffineCoordinates;
use rayon::prelude::*;
use speedbitcrack::utils::bias::{generate_preseed_pos, blend_proxy_preseed, analyze_preseed_cascade};
use speedbitcrack::math::bigint::BigInt256;

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

    /// Number of cascade levels for POS analysis (1-5)
    #[arg(long, default_value = "3")]
    cascade_levels: usize,
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
    let range_width = Scalar::from(u64::MAX); // Full keyspace proxy
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
    let cascades = analyze_preseed_cascade(&proxy, args.cascade_levels.min(5)); // Clamp max 5

    println!("🎯 Computing bias scores for all {} pubkeys...", pubkeys.len());

    // Parallel computation of residues and local scores
    let residues: Vec<u64> = pubkeys.par_iter()
        .map(|point| compute_mod_residue(point, args.mod_level))
        .collect();

    // Compute overall cascade score
    let overall_score = cascades.last().map(|(_, bias)| *bias).unwrap_or(1.0);

    // Parallel scoring with detailed computation
    let scores: Vec<(String, f64)> = pubkeys.par_iter().zip(residues.par_iter()).map(|(point, &residue)| {
        let pos_score = compute_pos_score(point, &cascades);
        let gold_bonus = if is_gold_cluster(residue, args.mod_level) { 1.2 } else { 1.0 };
        let mod_bonus = match args.mod_level {
            81 => if residue % 9 == 0 { 1.1 } else { 1.0 }, // Mod9 alignment bonus
            27 => if residue % 9 == 0 { 1.05 } else { 1.0 },
            9 => 1.0,
            _ => 1.0,
        };
        let total_score = overall_score * pos_score * gold_bonus * mod_bonus;
        (hex::encode(point.to_encoded_point(false).as_bytes()), total_score)
    }).collect();

    // Sort by score descending for priority ordering
    let mut sorted_scores = scores;
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Extract high bias targets
    println!("🔍 Extracting high-bias targets (threshold: {:.2}x)...", args.threshold);
    let high_bias: Vec<String> = sorted_scores.iter()
        .filter(|&(_, score)| *score > args.threshold)
        .map(|(hex, score)| {
            println!("  🎯 Score: {:.2}x - {}", score, &hex[..16]); // Truncate for readability
            hex.clone()
        })
        .collect();

    // Detailed statistics
    let avg_score = sorted_scores.iter().map(|(_, s)| s).sum::<f64>() / sorted_scores.len() as f64;
    let max_score = sorted_scores.first().map(|(_, s)| *s).unwrap_or(0.0);
    let min_score = sorted_scores.last().map(|(_, s)| *s).unwrap_or(0.0);
    let high_percentage = (high_bias.len() as f64 / sorted_scores.len() as f64) * 100.0;

    println!("📊 Analysis Statistics:");
    println!("  📈 Average score: {:.2}x", avg_score);
    println!("  🎯 Max score: {:.2}x", max_score);
    println!("  📉 Min score: {:.2}x", min_score);
    println!("  🎪 High priority: {} targets ({:.1}%)", high_bias.len(), high_percentage);

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
                        AffinePoint::from_bytes(bytes.as_slice().try_into().unwrap()).into_option()
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
    let x_bytes = point.x().as_bytes();
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
    let x_bytes = point.x().as_bytes();
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

    #[test]
    fn test_bias_analysis_workflow() {
        // Test the core bias analysis workflow with mock data
        let mock_pubkeys = vec![
            AffinePoint::GENERATOR,
            // Add more mock points if needed
        ];

        let cascades = vec![(2.0, 1.5), (1.8, 1.3)]; // Mock cascades

        for point in &mock_pubkeys {
            let residue = compute_mod_residue(point, 81);
            let pos_score = compute_pos_score(point, &cascades);
            let gold_bonus = if is_gold_cluster(residue, 81) { 1.2 } else { 1.0 };
            let mod_bonus = if residue % 9 == 0 { 1.1 } else { 1.0 };
            let total_score = 1.5 * pos_score * gold_bonus * mod_bonus; // Mock overall

            assert!(total_score >= 1.0, "Bias score should be at least 1.0: {}", total_score);
        }
    }
}
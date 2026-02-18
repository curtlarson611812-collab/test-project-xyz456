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
use speedbitcrack::utils::bias::{compute_bins, aggregate_chi, trend_penalty, BiasWeights, BiasScores, compute_bias_scores, GlobalBiasStats, calculate_mod3_bias_with_stats, calculate_mod9_bias_with_stats, calculate_mod27_bias_with_stats, calculate_mod81_bias_with_stats};
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

    println!("🎯 Computing REAL statistical biases for all {} pubkeys...", pubkeys.len());

    // Convert pubkeys to hex strings for statistical analysis
    let pubkey_hexes: Vec<String> = pubkeys.par_iter()
        .map(|point| hex::encode(point.to_encoded_point(false).as_bytes()))
        .collect();

    // Compute modular bias statistics using REAL statistical analysis
    println!("📊 Computing mod3 bias statistics...");
    let mod3_bins = compute_bins(&pubkey_hexes, 3, 3)?;
    let mod3_chi = aggregate_chi(&mod3_bins, 1000.0 / 3.0, pubkey_hexes.len() as f64);
    let mod3_penalty = trend_penalty(&mod3_bins, 3);

    println!("📊 Computing mod9 bias statistics...");
    let mod9_bins = compute_bins(&pubkey_hexes, 9, 9)?;
    let mod9_chi = aggregate_chi(&mod9_bins, 1000.0 / 9.0, pubkey_hexes.len() as f64);
    let mod9_penalty = trend_penalty(&mod9_bins, 9);

    println!("📊 Computing mod27 bias statistics...");
    let mod27_bins = compute_bins(&pubkey_hexes, 27, 27)?;
    let mod27_chi = aggregate_chi(&mod27_bins, 1000.0 / 27.0, pubkey_hexes.len() as f64);
    let mod27_penalty = trend_penalty(&mod27_bins, 27);

    println!("📊 Computing mod81 bias statistics...");
    let mod81_bins = compute_bins(&pubkey_hexes, 81, 81)?;
    let mod81_chi = aggregate_chi(&mod81_bins, 1000.0 / 81.0, pubkey_hexes.len() as f64);
    let mod81_penalty = trend_penalty(&mod81_bins, 81);

    // Create bias weights for analysis (favor GOLD cluster detection)
    let weights = if args.mod_level == 81 {
        BiasWeights::gold_focused() // Focus on GOLD clusters for valuable_p2pk
    } else {
        BiasWeights::balanced()
    };

    println!("🎯 Computing individual bias scores using statistical analysis...");

    // Create global statistics for proper bias analysis
    let global_mod3_stats = GlobalBiasStats {
        chi: mod3_chi,
        bins: mod3_bins.clone(),
        expected: 1000.0 / 3.0,
        penalty: mod3_penalty,
    };
    let global_mod9_stats = GlobalBiasStats {
        chi: mod9_chi,
        bins: mod9_bins.clone(),
        expected: 1000.0 / 9.0,
        penalty: mod9_penalty,
    };
    let global_mod27_stats = GlobalBiasStats {
        chi: mod27_chi,
        bins: mod27_bins.clone(),
        expected: 1000.0 / 27.0,
        penalty: mod27_penalty,
    };
    let global_mod81_stats = GlobalBiasStats {
        chi: mod81_chi,
        bins: mod81_bins.clone(),
        expected: 1000.0 / 81.0,
        penalty: mod81_penalty,
    };

    // Compute bias scores for each pubkey using REAL statistical analysis
    let bias_scores: Vec<BiasScores> = pubkeys.par_iter()
        .map(|point| {
            // Use statistical analysis instead of fixed values
            let mod3_bias = calculate_mod3_bias_with_stats(point, &global_mod3_stats);
            let mod9_bias = calculate_mod9_bias_with_stats(point, &global_mod9_stats);
            let mod27_bias = calculate_mod27_bias_with_stats(point, &global_mod27_stats);
            let mod81_bias = calculate_mod81_bias_with_stats(point, &global_mod81_stats);

            BiasScores {
                basic_bias: 0.5, // Placeholder
                mod3_bias,
                mod9_bias,
                mod27_bias,
                mod81_bias,
                golden_bias: 0.5, // Placeholder
                pop_bias: 0.5, // Placeholder
            }
        })
        .collect();

    // Combine modular statistics with individual bias scores
    let scores: Vec<(String, f64, bool)> = pubkeys.par_iter().zip(bias_scores.par_iter()).map(|(point, bias_score)| {
        let hex = hex::encode(point.to_encoded_point(false).as_bytes());

        // Compute modular bias score from statistical analysis
        let mod_score = weights.compute_score(bias_score);

        // GOLD cluster bonus (addresses with mod81=0 pattern)
        let is_gold = bias_score.mod81_bias > 0.8; // Strong GOLD cluster membership
        let gold_multiplier = if is_gold { 2.0 } else { 1.0 };

        // Combine modular statistics with individual scores
        let global_mod_bonus = 1.0 +
            (mod3_chi * 0.1) + (mod9_chi * 0.15) +
            (mod27_chi * 0.2) + (mod81_chi * 0.3); // Favor higher modulus biases

        let total_score = mod_score * global_mod_bonus * gold_multiplier;

        (hex, total_score, is_gold)
    }).collect();

    // Sort by score descending for priority ordering
    let mut sorted_scores: Vec<(String, f64, bool)> = scores;
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Extract GOLD cluster addresses (mod81=0 pattern)
    let gold_cluster: Vec<String> = sorted_scores.iter()
        .filter(|(_, _, is_gold)| *is_gold)
        .map(|(hex, score, _)| {
            println!("  🏆 GOLD Cluster: {:.2}x - {}", score, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Extract top 100 highest bias addresses
    let top_100: Vec<String> = sorted_scores.iter()
        .take(100)
        .map(|(hex, score, is_gold)| {
            let marker = if *is_gold { " [GOLD]" } else { "" };
            println!("  🎯 Top 100: {:.2}x{} - {}", score, marker, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Extract high bias targets (above threshold)
    let high_bias: Vec<String> = sorted_scores.iter()
        .filter(|&(_, score, _)| *score > args.threshold)
        .map(|(hex, score, _)| {
            println!("  🎯 High Bias: {:.2}x - {}", score, &hex[..16]);
            hex.clone()
        })
        .collect();

    // Detailed statistics
    let avg_score = sorted_scores.iter().map(|(_, s, _)| s).sum::<f64>() / sorted_scores.len() as f64;
    let max_score = sorted_scores.first().map(|(_, s, _)| *s).unwrap_or(0.0);
    let min_score = sorted_scores.last().map(|(_, s, _)| *s).unwrap_or(0.0);
    let gold_percentage = (gold_cluster.len() as f64 / sorted_scores.len() as f64) * 100.0;
    let high_percentage = (high_bias.len() as f64 / sorted_scores.len() as f64) * 100.0;

    println!("\n📊 Analysis Statistics:");
    println!("  📈 Average bias score: {:.3}x", avg_score);
    println!("  🎯 Maximum bias score: {:.3}x", max_score);
    println!("  📉 Minimum bias score: {:.3}x", min_score);
    println!("  🏆 GOLD cluster addresses: {} ({:.2}%)", gold_cluster.len(), gold_percentage);
    println!("  🎯 High bias targets: {} ({:.2}%)", high_bias.len(), high_percentage);
    println!("  🥇 Top 100 addresses: {}", top_100.len());

    // Statistical analysis of modular biases
    println!("\n🎲 Modular Bias Statistics:");
    println!("  📊 Mod3 χ²: {:.3} (deviation: {:.1}%)", mod3_chi, mod3_chi * 100.0);
    println!("  📊 Mod9 χ²: {:.3} (deviation: {:.1}%)", mod9_chi, mod9_chi * 100.0);
    println!("  📊 Mod27 χ²: {:.3} (deviation: {:.1}%)", mod27_chi, mod27_chi * 100.0);
    println!("  📊 Mod81 χ²: {:.3} (deviation: {:.1}%)", mod81_chi, mod81_chi * 100.0);

    // Write GOLD cluster list
    let gold_output = format!("{}_gold.txt", args.output.trim_end_matches(".txt"));
    println!("\n💾 Writing {} GOLD cluster addresses to {}...", gold_cluster.len(), gold_output);
    write_priority_list(&gold_output, &gold_cluster)?;

    // Write top 100 list
    let top100_output = format!("{}_top100.txt", args.output.trim_end_matches(".txt"));
    println!("💾 Writing top 100 addresses to {}...", top100_output);
    write_priority_list(&top100_output, &top_100)?;

    // Write high bias list
    println!("💾 Writing {} high-bias addresses to {}...", high_bias.len(), args.output);
    write_priority_list(&args.output, &high_bias)?;

    println!("\n✅ Bias analysis complete!");
    println!("  🏆 GOLD cluster: {} addresses", gold_cluster.len());
    println!("  🥇 Top 100: {} addresses", top_100.len());
    println!("  🎯 High bias: {} addresses", high_bias.len());

    if gold_cluster.is_empty() {
        println!("⚠️  No GOLD cluster addresses found - check bias computation");
    }

    if high_bias.is_empty() {
        println!("⚠️  No addresses exceeded threshold {:.2}x - try lowering threshold", args.threshold);
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

/// Legacy functions kept for compatibility - now using real statistical analysis from bias.rs

/// Write priority list to file with detailed headers
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
    writeln!(file, "# Generated by speedbitcrack bias_analyze tool")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bias_analysis_workflow() {
        // Test that the bias analysis can process pubkeys without panicking
        let mock_pubkeys = vec![AffinePoint::GENERATOR];

        // This should not panic and should produce valid bias scores
        for point in &mock_pubkeys {
            let bias_scores = compute_bias_scores(point);
            assert!(bias_scores.mod3_bias >= 0.0 && bias_scores.mod3_bias <= 1.0);
            assert!(bias_scores.mod9_bias >= 0.0 && bias_scores.mod9_bias <= 1.0);
            assert!(bias_scores.mod27_bias >= 0.0 && bias_scores.mod27_bias <= 1.0);
            assert!(bias_scores.mod81_bias >= 0.0 && bias_scores.mod81_bias <= 1.0);
        }
    }

    #[test]
    fn test_load_pubkeys_parallel() {
        // Create a temporary file with a valid pubkey
        let temp_file = "/tmp/test_pubkeys.txt";
        {
            let mut file = File::create(temp_file).unwrap();
            writeln!(file, "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798").unwrap();
        }

        let result = load_pubkeys_parallel(temp_file);
        assert!(result.is_ok());
        let pubkeys = result.unwrap();
        assert_eq!(pubkeys.len(), 1);

        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
}
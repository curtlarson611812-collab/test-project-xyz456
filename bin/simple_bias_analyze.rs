use clap::Parser;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use k256::AffinePoint;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::group::GroupEncoding;
use k256::elliptic_curve::point::AffineCoordinates;
use rayon::prelude::*;
use speedbitcrack::math::bigint::BigInt256;

/// Simple standalone bias analysis tool for valuable_p2pk_pubkeys.txt
/// Computes proper statistical bias analysis without complex dependencies
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

    println!("🔍 Loading pubkeys from {}...", args.input);
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
    let mod3_counts = compute_modular_counts(&pubkey_hexes, 3);
    let _mod3_bias = compute_bias_score(&mod3_counts, pubkey_hexes.len());

    println!("📊 Computing mod9 bias statistics...");
    let mod9_counts = compute_modular_counts(&pubkey_hexes, 9);
    let _mod9_bias = compute_bias_score(&mod9_counts, pubkey_hexes.len());

    println!("📊 Computing mod27 bias statistics...");
    let mod27_counts = compute_modular_counts(&pubkey_hexes, 27);
    let _mod27_bias = compute_bias_score(&mod27_counts, pubkey_hexes.len());

    println!("📊 Computing mod81 bias statistics...");
    let mod81_counts = compute_modular_counts(&pubkey_hexes, 81);
    let _mod81_bias = compute_bias_score(&mod81_counts, pubkey_hexes.len());

    println!("🎯 Computing individual bias scores...");

    // Compute individual bias scores for each pubkey
    let bias_scores: Vec<(String, f64, usize)> = pubkeys.par_iter().enumerate().map(|(_i, point)| {
        let hex = hex::encode(point.to_encoded_point(false).as_bytes());

        // Calculate modular residues
        let x_bytes = point.x();
        let x_u64 = u64::from_be_bytes(x_bytes[..8].try_into().unwrap());
        let x = BigInt256 { limbs: [x_u64, 0, 0, 0] };
        let mod3_residue = (x.clone() % BigInt256::from_u64(3)).to_u64() as usize;
        let mod9_residue = (x.clone() % BigInt256::from_u64(9)).to_u64() as usize;
        let mod27_residue = (x.clone() % BigInt256::from_u64(27)).to_u64() as usize;
        let mod81_residue = (x.clone() % BigInt256::from_u64(81)).to_u64() as usize;

        // Compute bias score based on deviation from uniform distribution
        let mut score = 0.0;

        // Mod3 contribution
        let mod3_expected = pubkey_hexes.len() as f64 / 3.0;
        let mod3_deviation = (mod3_counts[mod3_residue] as f64 - mod3_expected) / mod3_expected;
        score += mod3_deviation.abs() * 0.25;

        // Mod9 contribution
        let mod9_expected = pubkey_hexes.len() as f64 / 9.0;
        let mod9_deviation = (mod9_counts[mod9_residue] as f64 - mod9_expected) / mod9_expected;
        score += mod9_deviation.abs() * 0.25;

        // Mod27 contribution
        let mod27_expected = pubkey_hexes.len() as f64 / 27.0;
        let mod27_deviation = (mod27_counts[mod27_residue] as f64 - mod27_expected) / mod27_expected;
        score += mod27_deviation.abs() * 0.25;

        // Mod81 contribution (GOLD cluster detection)
        let mod81_expected = pubkey_hexes.len() as f64 / 81.0;
        let mod81_deviation = (mod81_counts[mod81_residue] as f64 - mod81_expected) / mod81_expected;
        score += mod81_deviation.abs() * 0.25;

        // GOLD cluster bonus
        let is_gold = mod81_residue == 0;
        if is_gold {
            score *= 2.0; // Double score for GOLD cluster addresses
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
fn compute_modular_counts(pubkeys: &[String], modulus: u64) -> Vec<usize> {
    let mut counts = vec![0; modulus as usize];

    for pubkey_hex in pubkeys {
        // Decode the compressed pubkey and get x coordinate
        if let Ok(bytes) = hex::decode(pubkey_hex) {
            if bytes.len() == 33 {
                if let Some(point) = AffinePoint::from_bytes(bytes.as_slice().try_into().unwrap()).into_option() {
                    let x_bytes = point.x();
                    let x_u64 = u64::from_be_bytes(x_bytes[..8].try_into().unwrap());
                    let x = BigInt256 { limbs: [x_u64, 0, 0, 0] };
                    let residue = (x % BigInt256::from_u64(modulus)).to_u64() as usize;
                    if residue < counts.len() {
                        counts[residue] += 1;
                    }
                }
            }
        }
    }

    counts
}

/// Compute bias score from counts
fn compute_bias_score(counts: &[usize], total: usize) -> f64 {
    let expected = total as f64 / counts.len() as f64;
    let mut chi_squared = 0.0;

    for &count in counts {
        let diff = count as f64 - expected;
        chi_squared += diff * diff / expected;
    }

    chi_squared
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
            if hex_str.is_empty() {
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
    writeln!(file, "# Generated by speedbitcrack simple_bias_analyze tool")?;
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
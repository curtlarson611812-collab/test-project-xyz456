//! Real-Time Boxed Terminal Output for SpeedBitCrackV3
//!
//! Provides live progress updates during Pollard kangaroo runs with:
//! - Header: Version, start time, run time
//! - Options: All CLI args and config defaults for reproducibility
//! - Bias: Score and weights with utilization status
//! - Verbose: Scrollable log of collisions, cycles, progress

use crossterm::{
    terminal::{Clear, ClearType},
    cursor::MoveTo,
    execute,
};
use std::io::{stdout, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Simplified args struct for display (to avoid circular dependencies)
#[derive(Debug, Clone)]
pub struct DisplayArgs {
    pub puzzle: Option<u32>,
    pub valuable: bool,
    pub test_puzzles: bool,
    pub gpu: bool,
    pub laptop: bool,
    pub verbose: bool,
    pub max_cycles: u64,
    pub num_kangaroos: usize,
    pub bias_mod: u64,
}

/// Simplified config struct for display
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    pub dp_bits: usize,
    pub herd_size: usize,
    pub jump_mean: u64,
    pub near_threshold: u64,
}

/// Start real-time boxed output in a background thread
/// Updates every 1 second until stop_flag is set
pub fn start_real_time_output(
    version: String,
    start_time: Instant,
    args: DisplayArgs,
    default_config: DisplayConfig,
    address: String,
    bias_score: f64,
    biases: HashMap<u32, f64>,
    stop_flag: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        while !stop_flag.load(Ordering::Relaxed) {
            let run_time = start_time.elapsed();
            let mut out = stdout();

            // Clear screen and move cursor to top
            execute!(&mut out, Clear(ClearType::All), MoveTo(0, 0)).ok();

            // Print all boxed sections
            print_boxed_header(&version, start_time, run_time);
            print_boxed_options(&args, &default_config);
            print_boxed_bias(&address, bias_score, &biases);
            print_boxed_verbose();

            // Flush output
            out.flush().ok();

            // Update every 1 second
            thread::sleep(Duration::from_secs(1));
        }
    });
}

/// Print boxed header with version, start time, run time
fn print_boxed_header(version: &str, start_time: Instant, run_time: Duration) {
    let start_formatted = format_duration(start_time.elapsed());
    let run_formatted = format_duration(run_time);

    println!("┌─────────────────────────────────────┐");
    println!("│ SpeedBitCrackV3 v{}                │", version);
    println!("│ Started: {}                       │", start_formatted);
    println!("│ Run Time: {}                       │", run_formatted);
    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed options showing key args and config defaults
fn print_boxed_options(args: &DisplayArgs, default_config: &DisplayConfig) {
    println!("┌─────────────────────────────────────┐");
    println!("│ Run Options:                        │");

    // Key arguments
    if let Some(puzzle) = args.puzzle {
        println!("│ Puzzle: #{}                        │", puzzle);
    } else if args.valuable {
        println!("│ Mode: Valuable P2PK                │");
    } else if args.test_puzzles {
        println!("│ Mode: Test Puzzles                 │");
    } else {
        println!("│ Mode: Custom/Default               │");
    }

    println!("│ GPU: {}                             │", if args.gpu { "Enabled" } else { "Disabled" });
    println!("│ Laptop Opt: {}                     │", if args.laptop { "Yes" } else { "No" });
    println!("│ Verbose: {}                        │", if args.verbose { "Yes" } else { "No" });
    println!("│ Max Cycles: {}                    │", if args.max_cycles > 0 { args.max_cycles.to_string() } else { "Unlimited".to_string() });

    // Config defaults
    println!("│ DP Bits: {}                        │", default_config.dp_bits);
    println!("│ Herd Size: {}                     │", default_config.herd_size);
    println!("│ Jump Mean: {}                     │", default_config.jump_mean);
    println!("│ Near Threshold: {}                │", default_config.near_threshold);

    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed bias information with score and weights
fn print_boxed_bias(address: &str, bias_score: f64, biases: &HashMap<u32, f64>) {
    println!("┌─────────────────────────────────────┐");
    println!("│ Target Address:                     │");

    // Truncate address for display
    let display_addr = if address.len() > 35 {
        format!("{}...", &address[..32])
    } else {
        address.to_string()
    };
    println!("│ {} │", display_addr);

    println!("│                                     │");
    println!("│ Bias Analysis:                      │");
    println!("│ Score: {:.3} ({})                │",
             bias_score,
             if bias_score > 1.2 { "ACTIVE" } else { "UNIFORM" });

    if !biases.is_empty() {
        println!("│ Biases Applied:                     │");
        // Show top 3 bias weights
        let mut sorted_biases: Vec<_> = biases.iter().collect();
        sorted_biases.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (i, (res, weight)) in sorted_biases.iter().take(3).enumerate() {
            println!("│  res {}: {:.2}                      │", res, weight);
        }
    } else {
        println!("│ No biases applied (uniform)         │");
    }

    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed verbose logs (placeholder - would pull from actual log buffer)
fn print_boxed_verbose() {
    println!("┌─────────────────────────────────────┐");
    println!("│ Run Progress & Events:              │");
    println!("│ Steps: 0 / Est: 4.0B (0.0%)       │");
    println!("│ Hashrate: 0.0 M ops/sec            │");
    println!("│ ETA: --:--:--                       │");
    println!("│                                     │");
    println!("│ Recent Events:                      │");
    println!("│ • Run started                       │");
    println!("│ • Biases loaded                     │");
    println!("│ • GPU initialized                   │");
    println!("│ • Kangaroos deployed               │");
    println!("│                                     │");
    println!("│ Waiting for collisions...           │");
    println!("└─────────────────────────────────────┘");
}

/// Format duration as HH:MM:SS
fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Calculate hashrate in M ops/sec
fn calculate_hashrate(total_ops: u64, run_time: Duration) -> f64 {
    if run_time.as_secs_f64() > 0.0 {
        total_ops as f64 / run_time.as_secs_f64() / 1_000_000.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_format_duration() {
        let duration = Duration::from_secs(3661); // 1h 1m 1s
        assert_eq!(format_duration(duration), "01:01:01");
    }

    #[test]
    fn test_calculate_hashrate() {
        let total_ops = 150_000_000; // 150M ops
        let run_time = Duration::from_secs(1);
        assert!((calculate_hashrate(total_ops, run_time) - 150.0).abs() < 0.1);
    }
}
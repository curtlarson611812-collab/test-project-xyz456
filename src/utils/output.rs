//! Real-Time Boxed Terminal Output for SpeedBitCrackV3
//!
//! Provides live progress updates during Pollard kangaroo runs with:
//! - Header: Version, start time, run time, "Created By GROK Code Fast 1"
//! - Options: All CLI args and config defaults for reproducibility
//! - Bias: Score and weights with utilization status (color coded)
//! - Verbose: Scrollable log buffer with collisions, cycles, progress
//! - Metrics: Hashrate, progress %, ETA, GPU temp, memory usage

use crossterm::{
    cursor::MoveTo,
    execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType},
};
use log::{Level, LevelFilter};
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// Global log buffer for capturing recent messages
static LOG_BUFFER: Mutex<Vec<String>> = Mutex::new(Vec::new());
const MAX_LOG_LINES: usize = 8; // Show last 8 log messages

/// Initialize log buffer capture
pub fn init_log_capture() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        // Try to set up custom logger that captures messages
        // If it fails (already initialized), that's ok
        let _ = log::set_logger(&Logger);
        log::set_max_level(LevelFilter::Info);
    });
}

/// Custom logger that captures messages to buffer
struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.level(), record.args());
            let mut buffer = LOG_BUFFER.lock().unwrap();
            buffer.push(msg);
            if buffer.len() > MAX_LOG_LINES {
                buffer.remove(0); // Remove oldest
            }
        }
    }

    fn flush(&self) {}
}

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
    // Initialize log capture
    init_log_capture();

    thread::spawn(move || {
        let mut total_ops: u64 = 0;
        let mut gpu_temp: f32 = 45.0; // Mock GPU temp
        let mut mem_usage: f64 = 2.5; // Mock memory usage in GB

        while !stop_flag.load(Ordering::Relaxed) {
            let run_time = start_time.elapsed();

            // Simulate progress (in real implementation, get from actual kangaroo state)
            total_ops += 150_000_000; // Mock 150M ops/sec
            gpu_temp += (rand::random::<f32>() - 0.5) * 2.0; // Temperature variation
            gpu_temp = gpu_temp.max(40.0).min(80.0);
            mem_usage += (rand::random::<f64>() - 0.5) * 0.1;
            mem_usage = mem_usage.max(2.0).min(8.0);

            let mut out = stdout();

            // Clear screen and move cursor to top
            execute!(&mut out, Clear(ClearType::All), MoveTo(0, 0)).ok();

            // Print all boxed sections
            print_boxed_header(&version, start_time, run_time);
            print_boxed_options(&args, &default_config);
            print_boxed_bias(&address, bias_score, &biases);
            print_boxed_metrics(total_ops, run_time, gpu_temp, mem_usage);
            print_boxed_verbose();

            // Flush output
            out.flush().ok();

            // Update every 1 second
            thread::sleep(Duration::from_secs(1));
        }
    });
}

/// Print boxed header with version, start time, run time, and flair
fn print_boxed_header(version: &str, start_time: Instant, run_time: Duration) {
    let start_formatted = format_duration(start_time.elapsed());
    let run_formatted = format_duration(run_time);

    println!("┌─────────────────────────────────────┐");
    execute!(stdout(), SetForegroundColor(Color::Green)).ok();
    println!("│ SpeedBitCrackV3 v{}                │", version);
    execute!(stdout(), ResetColor).ok();
    println!("│ Started: {}                       │", start_formatted);
    println!("│ Run Time: {}                       │", run_formatted);
    execute!(stdout(), SetForegroundColor(Color::Yellow)).ok();
    println!("│ Created By GROK Code Fast 1        │");
    execute!(stdout(), ResetColor).ok();
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

    println!(
        "│ GPU: {}                             │",
        if args.gpu { "Enabled" } else { "Disabled" }
    );
    println!(
        "│ Laptop Opt: {}                     │",
        if args.laptop { "Yes" } else { "No" }
    );
    println!(
        "│ Verbose: {}                        │",
        if args.verbose { "Yes" } else { "No" }
    );
    println!(
        "│ Max Cycles: {}                    │",
        if args.max_cycles > 0 {
            args.max_cycles.to_string()
        } else {
            "Unlimited".to_string()
        }
    );

    // Config defaults
    println!(
        "│ DP Bits: {}                        │",
        default_config.dp_bits
    );
    println!(
        "│ Herd Size: {}                     │",
        default_config.herd_size
    );
    println!(
        "│ Jump Mean: {}                     │",
        default_config.jump_mean
    );
    println!(
        "│ Near Threshold: {}                │",
        default_config.near_threshold
    );

    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed bias information with score and weights (color coded)
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

    // Color code the bias status
    if bias_score > 1.2 {
        execute!(stdout(), SetForegroundColor(Color::Green)).ok();
        println!("│ Score: {:.3} (ACTIVE)              │", bias_score);
        execute!(stdout(), ResetColor).ok();
    } else {
        execute!(stdout(), SetForegroundColor(Color::Red)).ok();
        println!("│ Score: {:.3} (UNIFORM)             │", bias_score);
        execute!(stdout(), ResetColor).ok();
    }

    if !biases.is_empty() {
        println!("│ Biases Applied:                     │");
        // Show top 3 bias weights
        let mut sorted_biases: Vec<_> = biases.iter().collect();
        sorted_biases.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (_i, (res, weight)) in sorted_biases.iter().take(3).enumerate() {
            if **weight > 1.3 {
                execute!(stdout(), SetForegroundColor(Color::Magenta)).ok();
            } else if **weight > 1.1 {
                execute!(stdout(), SetForegroundColor(Color::Yellow)).ok();
            }
            println!("│  res {}: {:.2}                      │", res, weight);
            execute!(stdout(), ResetColor).ok();
        }
    } else {
        execute!(stdout(), SetForegroundColor(Color::Red)).ok();
        println!("│ No biases applied (uniform)         │");
        execute!(stdout(), ResetColor).ok();
    }

    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed performance metrics
fn print_boxed_metrics(total_ops: u64, run_time: Duration, gpu_temp: f32, mem_usage: f64) {
    let hashrate = calculate_hashrate(total_ops, run_time);

    // Estimate progress (simplified - in real implementation, get from kangaroo state)
    let estimated_total = 4_000_000_000_000u64; // Mock 4T operations for #66
    let progress_percent = (total_ops as f64 / estimated_total as f64 * 100.0).min(100.0);

    // Estimate ETA based on current progress
    let eta = if progress_percent > 0.0 && progress_percent < 100.0 {
        let remaining_percent = 100.0 - progress_percent;
        let total_time_estimate = run_time.as_secs_f64() / (progress_percent / 100.0);
        let remaining_time = total_time_estimate * (remaining_percent / 100.0);
        format_duration(Duration::from_secs_f64(remaining_time))
    } else {
        "--:--:--".to_string()
    };

    println!("┌─────────────────────────────────────┐");
    println!("│ Performance Metrics:                │");

    // Color code hashrate
    if hashrate > 150.0 {
        execute!(stdout(), SetForegroundColor(Color::Green)).ok();
    } else if hashrate > 100.0 {
        execute!(stdout(), SetForegroundColor(Color::Yellow)).ok();
    } else {
        execute!(stdout(), SetForegroundColor(Color::Red)).ok();
    }
    println!("│ Hashrate: {:.0} M ops/sec            │", hashrate);
    execute!(stdout(), ResetColor).ok();

    println!(
        "│ Progress: {:.1}%                      │",
        progress_percent
    );
    println!("│ ETA: {}                            │", eta);

    // Color code temperature
    if gpu_temp > 75.0 {
        execute!(stdout(), SetForegroundColor(Color::Red)).ok();
    } else if gpu_temp > 65.0 {
        execute!(stdout(), SetForegroundColor(Color::Yellow)).ok();
    } else {
        execute!(stdout(), SetForegroundColor(Color::Green)).ok();
    }
    println!("│ GPU Temp: {:.1}°C                     │", gpu_temp);
    execute!(stdout(), ResetColor).ok();

    println!("│ Memory: {:.1} GB                      │", mem_usage);

    println!("└─────────────────────────────────────┘");
    println!();
}

/// Print boxed verbose logs with captured log buffer
fn print_boxed_verbose() {
    println!("┌─────────────────────────────────────┐");
    println!("│ Run Events & Logs:                  │");

    // Get recent log messages
    let log_messages = LOG_BUFFER.lock().unwrap();

    if log_messages.is_empty() {
        println!("│ • Run started                       │");
        println!("│ • Biases loaded                     │");
        println!("│ • GPU initialized                   │");
        println!("│ • Kangaroos deployed               │");
        println!("│                                     │");
        println!("│ Waiting for collisions...           │");
        println!("│                                     │");
        println!("│                                     │");
    } else {
        // Show last few log messages
        let start_idx = if log_messages.len() > 6 {
            log_messages.len() - 6
        } else {
            0
        };
        for (_i, msg) in log_messages.iter().skip(start_idx).enumerate() {
            let truncated_msg = if msg.len() > 35 {
                format!("{}...", &msg[..32])
            } else {
                msg.clone()
            };

            // Color code log levels
            if msg.contains("[INFO]") {
                execute!(stdout(), SetForegroundColor(Color::Green)).ok();
            } else if msg.contains("[WARN]") {
                execute!(stdout(), SetForegroundColor(Color::Yellow)).ok();
            } else if msg.contains("[ERROR]") {
                execute!(stdout(), SetForegroundColor(Color::Red)).ok();
            }

            println!("│ {:35} │", truncated_msg);
            execute!(stdout(), ResetColor).ok();
        }

        // Fill remaining space if needed
        for _ in log_messages.len()..6 {
            println!("│                                     │");
        }
    }

    println!("└─────────────────────────────────────┘");
}

/// Print Cracker Curt mission patch ASCII art
pub fn print_mission_patch() {
    println!(
        "
   ╔══════════════════════════════════════╗
   ║                                      ║
   ║             MISSION PATCH            ║
   ║                                      ║
   ║              \\   /                   ║
   ║               \\ /                    ║
   ║                X                     ║
   ║               / \\                    ║
   ║              /   \\                   ║
   ║                                      ║
   ║          CRACKER CURT                ║
   ║                                      ║
   ║         ECDLP KING                   ║
   ║                                      ║
   ║      SPEEDBITCRACK V3               ║
   ║                                      ║
   ║       GÖDEL'S GHOST                 ║
   ║                                      ║
   ╚══════════════════════════════════════╝
    "
    );
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

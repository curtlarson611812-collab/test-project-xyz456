//! Advanced CLI with progress bars and real-time monitoring for SpeedBitCrack V3
//!
//! Features:
//! - Real-time progress bars with ETA calculations
//! - Live statistics display (ops/sec, memory usage, GPU utilization)
//! - Interactive controls (pause/resume, status queries)
//! - Visual collision detection alerts
//! - Performance trend analysis

use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute, queue, terminal,
};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Performance metrics collected during hunt
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_ops: u64,
    pub ops_per_second: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f64,
    pub dp_found: u64,
    pub collisions_found: u64,
    pub elapsed_time: Duration,
    pub eta_seconds: Option<f64>,
}

/// Hunt progress information
#[derive(Debug, Clone)]
pub struct HuntProgress {
    pub current_cycle: u64,
    pub max_cycles: u64,
    pub targets_processed: usize,
    pub total_targets: usize,
    pub solutions_found: usize,
    pub current_range: Option<(u64, u64)>,
    pub status_message: String,
}

/// Advanced CLI with real-time monitoring
pub struct AdvancedCli {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    progress: Arc<Mutex<HuntProgress>>,
    start_time: Instant,
    update_interval: Duration,
    is_running: Arc<Mutex<bool>>,
}

impl AdvancedCli {
    /// Create new advanced CLI instance
    pub fn new() -> Self {
        let metrics = PerformanceMetrics {
            total_ops: 0,
            ops_per_second: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization: 0.0,
            dp_found: 0,
            collisions_found: 0,
            elapsed_time: Duration::ZERO,
            eta_seconds: None,
        };

        let progress = HuntProgress {
            current_cycle: 0,
            max_cycles: 0,
            targets_processed: 0,
            total_targets: 0,
            solutions_found: 0,
            current_range: None,
            status_message: "Initializing...".to_string(),
        };

        AdvancedCli {
            metrics: Arc::new(Mutex::new(metrics)),
            progress: Arc::new(Mutex::new(progress)),
            start_time: Instant::now(),
            update_interval: Duration::from_millis(500),
            is_running: Arc::new(Mutex::new(true)),
        }
    }

    /// Start the CLI monitoring loop
    pub fn start(&self) -> Result<()> {
        let metrics = Arc::clone(&self.metrics);
        let progress = Arc::clone(&self.progress);
        let is_running = Arc::clone(&self.is_running);

        thread::spawn(move || {
            if let Err(e) = Self::monitoring_loop(metrics, progress, is_running) {
                eprintln!("CLI monitoring error: {}", e);
            }
        });

        Ok(())
    }

    /// Stop the CLI monitoring
    pub fn stop(&self) {
        *self.is_running.lock().unwrap() = false;
        // Give the monitoring thread time to clean up
        thread::sleep(Duration::from_millis(100));
    }

    /// Update performance metrics
    pub fn update_metrics(&self, metrics: PerformanceMetrics) {
        *self.metrics.lock().unwrap() = metrics;
    }

    /// Update hunt progress
    pub fn update_progress(&self, progress: HuntProgress) {
        *self.progress.lock().unwrap() = progress;
    }

    /// Set status message
    pub fn set_status(&self, message: String) {
        self.progress.lock().unwrap().status_message = message;
    }

    /// Monitoring loop that runs in background thread
    fn monitoring_loop(
        metrics: Arc<Mutex<PerformanceMetrics>>,
        progress: Arc<Mutex<HuntProgress>>,
        is_running: Arc<Mutex<bool>>,
    ) -> Result<()> {
        // Enable raw mode for keyboard input
        let mut stdout = io::stdout();
        execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

        while *is_running.lock().unwrap() {
            // Clear screen and draw interface
            Self::draw_interface(&metrics, &progress)?;

            // Check for keyboard input (non-blocking)
            if event::poll(Duration::from_millis(10))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            *is_running.lock().unwrap() = false;
                            break;
                        }
                        KeyCode::Char('p') => {
                            // Pause/resume functionality could be added here
                        }
                        _ => {}
                    }
                }
            }

            thread::sleep(Duration::from_millis(500));
        }

        // Clean up terminal
        execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show)?;
        Ok(())
    }

    /// Draw the complete CLI interface
    fn draw_interface(
        metrics: &Arc<Mutex<PerformanceMetrics>>,
        progress: &Arc<Mutex<HuntProgress>>,
    ) -> Result<()> {
        let mut stdout = io::stdout();
        queue!(
            stdout,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        let metrics = metrics.lock().unwrap();
        let progress = progress.lock().unwrap();

        // Title
        writeln!(stdout, "{}", "🚀 SpeedBitCrack V3 - Advanced Hunt Monitor")?;
        writeln!(stdout, "{}", "═".repeat(60))?;

        // Progress bar
        Self::draw_progress_bar(&progress)?;

        // Performance metrics
        Self::draw_metrics(&metrics)?;

        // Hunt status
        Self::draw_status(&progress)?;

        // Controls
        writeln!(stdout, "\n{}", "Controls:")?;
        writeln!(stdout, "  Q/Esc - Quit | P - Pause/Resume | S - Status")?;

        stdout.flush()?;
        Ok(())
    }

    /// Draw progress bar
    fn draw_progress_bar(progress: &HuntProgress) -> Result<()> {
        let mut stdout = io::stdout();

        let percentage = if progress.max_cycles > 0 {
            (progress.current_cycle as f64 / progress.max_cycles as f64 * 100.0) as u32
        } else {
            0
        };

        let bar_width = 40;
        let filled = (percentage as f64 / 100.0 * bar_width as f64) as usize;
        let empty = bar_width - filled;

        write!(stdout, "Progress: [")?;
        write!(stdout, "{}", "█".repeat(filled))?;
        write!(stdout, "{}", "░".repeat(empty))?;
        writeln!(
            stdout,
            "] {}% ({}/{})",
            percentage, progress.current_cycle, progress.max_cycles
        )?;

        // Targets progress
        let target_percentage = if progress.total_targets > 0 {
            (progress.targets_processed as f64 / progress.total_targets as f64 * 100.0) as u32
        } else {
            0
        };
        writeln!(
            stdout,
            "Targets:  {}% ({}/{}) processed",
            target_percentage, progress.targets_processed, progress.total_targets
        )?;

        Ok(())
    }

    /// Draw performance metrics
    fn draw_metrics(metrics: &PerformanceMetrics) -> Result<()> {
        let mut stdout = io::stdout();

        writeln!(stdout, "\n{}", "📊 Performance Metrics:")?;

        // Format numbers with appropriate units
        let ops_str = Self::format_large_number(metrics.total_ops);
        let ops_per_sec = metrics.ops_per_second;
        let memory_mb = metrics.memory_usage_mb;
        let gpu_util = metrics.gpu_utilization;

        writeln!(
            stdout,
            "  Operations: {} ({:.1}M ops/sec)",
            ops_str,
            ops_per_sec / 1_000_000.0
        )?;
        writeln!(stdout, "  Memory:     {:.1} MB", memory_mb)?;
        writeln!(stdout, "  GPU Util:   {:.1}%", gpu_util)?;
        writeln!(
            stdout,
            "  DPs Found:  {}",
            Self::format_large_number(metrics.dp_found)
        )?;
        writeln!(stdout, "  Collisions: {}", metrics.collisions_found)?;

        // ETA calculation
        if let Some(eta_secs) = metrics.eta_seconds {
            let eta = Self::format_duration(eta_secs as u64);
            writeln!(stdout, "  ETA:        {}", eta)?;
        }

        Ok(())
    }

    /// Draw hunt status
    fn draw_status(progress: &HuntProgress) -> Result<()> {
        let mut stdout = io::stdout();

        writeln!(stdout, "\n{}", "🎯 Hunt Status:")?;

        writeln!(stdout, "  Status:     {}", progress.status_message)?;

        if let Some((low, high)) = progress.current_range {
            writeln!(
                stdout,
                "  Range:      [{}, {}]",
                Self::format_large_number(low),
                Self::format_large_number(high)
            )?;
        }

        if progress.solutions_found > 0 {
            writeln!(
                stdout,
                "  {}",
                format!("🎉 Solutions Found: {}", progress.solutions_found)
            )?;
        }

        Ok(())
    }

    /// Format large numbers with appropriate suffixes
    fn format_large_number(num: u64) -> String {
        if num >= 1_000_000_000_000 {
            format!("{:.2}T", num as f64 / 1_000_000_000_000.0)
        } else if num >= 1_000_000_000 {
            format!("{:.2}B", num as f64 / 1_000_000_000.0)
        } else if num >= 1_000_000 {
            format!("{:.2}M", num as f64 / 1_000_000.0)
        } else if num >= 1_000 {
            format!("{:.2}K", num as f64 / 1_000.0)
        } else {
            num.to_string()
        }
    }

    /// Format duration in human-readable format
    fn format_duration(seconds: u64) -> String {
        let days = seconds / 86400;
        let hours = (seconds % 86400) / 3600;
        let minutes = (seconds % 3600) / 60;
        let secs = seconds % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, secs)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, secs)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }
}

impl Default for AdvancedCli {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AdvancedCli {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_large_number() {
        assert_eq!(AdvancedCli::format_large_number(500), "500");
        assert_eq!(AdvancedCli::format_large_number(1500), "1.50K");
        assert_eq!(AdvancedCli::format_large_number(2500000), "2.50M");
        assert_eq!(AdvancedCli::format_large_number(3500000000), "3.50B");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(AdvancedCli::format_duration(30), "30s");
        assert_eq!(AdvancedCli::format_duration(90), "1m 30s");
        assert_eq!(AdvancedCli::format_duration(3660), "1h 1m 0s");
        assert_eq!(
            AdvancedCli::format_duration(86400 + 3600 + 60 + 1),
            "1d 1h 1m 1s"
        );
    }
}

//! Performance monitoring and bottleneck analysis
//!
//! Comprehensive profiling, Nsight integration, and optimization
//! recommendations for hybrid GPU execution

// Types are defined in this module
use crate::gpu::HybridOperation;
use std::collections::HashMap;
use std::time::Instant;

/// Performance metric for hybrid operations
#[derive(Debug, Clone)]
pub struct HybridOperationMetrics {
    pub operation: String,
    pub backend: String,
    pub duration_ms: u128,
    pub data_size: usize,
    pub success: bool,
    pub timestamp: Instant,
}

/// Nsight rule result for GPU optimization
#[derive(Debug)]
pub struct NsightRuleResult {
    pub rule_name: String,
    pub score: f64,
    pub suggestion: String,
    pub severity: RuleSeverity,
}

/// Rule severity levels
#[derive(Debug)]
pub enum RuleSeverity {
    Info,
    Warning,
    Critical,
}

impl NsightRuleResult {
}


/// Pipeline performance summary
#[derive(Debug)]
pub struct PipelinePerformanceSummary {
    pub total_duration: std::time::Duration,
    pub stage_summaries: Vec<StagePerformanceSummary>,
    pub bottleneck_stage: Option<usize>,
    pub optimization_score: f64,
}

/// Stage performance summary
#[derive(Debug)]
pub struct StagePerformanceSummary {
    pub stage_id: usize,
    pub stage_name: String,
    pub average_duration: std::time::Duration,
    pub min_duration: std::time::Duration,
    pub max_duration: std::time::Duration,
    pub execution_count: usize,
    pub success_rate: f64,
}

impl NsightRuleResult {
    /// Create new Nsight rule result
    pub fn new(rule_name: &str, score: f64, suggestion: &str) -> Self {
        let severity = if score < 0.3 {
            RuleSeverity::Critical
        } else if score < 0.7 {
            RuleSeverity::Warning
        } else {
            RuleSeverity::Info
        };

        NsightRuleResult {
            rule_name: rule_name.to_string(),
            score,
            suggestion: suggestion.to_string(),
            severity,
        }
    }
}

impl Default for HybridOperationMetrics {
    fn default() -> Self {
        HybridOperationMetrics {
            operation: String::new(),
            backend: String::new(),
            duration_ms: 0,
            data_size: 0,
            success: true,
            timestamp: Instant::now(),
        }
    }
}

impl PipelinePerformanceSummary {
    /// Create performance summary from stage timings
    pub fn from_stage_timings(
        stage_durations: &HashMap<usize, Vec<std::time::Duration>>,
        stage_names: &HashMap<usize, String>,
    ) -> Self {
        let mut total_duration = std::time::Duration::ZERO;
        let mut stage_summaries = Vec::new();
        let mut max_duration = std::time::Duration::ZERO;
        let mut bottleneck_stage = None;

        for (stage_id, durations) in stage_durations {
            if durations.is_empty() {
                continue;
            }

            let sum: std::time::Duration = durations.iter().sum();
            let avg_duration = sum / durations.len() as u32;
            let min_duration = durations.iter().min().unwrap().clone();
            let max_duration_stage = durations.iter().max().unwrap().clone();

            total_duration += sum;

            if max_duration_stage > max_duration {
                max_duration = max_duration_stage;
                bottleneck_stage = Some(*stage_id);
            }

            let stage_name = stage_names.get(stage_id).cloned().unwrap_or_else(|| format!("Stage {}", stage_id));

            stage_summaries.push(StagePerformanceSummary {
                stage_id: *stage_id,
                stage_name,
                average_duration: avg_duration,
                min_duration,
                max_duration: max_duration_stage,
                execution_count: durations.len(),
                success_rate: 1.0, // Assume all successful for now
            });
        }

        // Calculate optimization score (0.0 = needs optimization, 1.0 = optimal)
        let optimization_score = if !stage_summaries.is_empty() {
            let avg_utilization = stage_summaries.iter()
                .map(|s| s.average_duration.as_secs_f64())
                .sum::<f64>() / stage_summaries.len() as f64;

            let variance = stage_summaries.iter()
                .map(|s| (s.average_duration.as_secs_f64() - avg_utilization).powi(2))
                .sum::<f64>() / stage_summaries.len() as f64;

            // Lower variance = higher optimization score
            (1.0 / (1.0 + variance.sqrt())).min(1.0)
        } else {
            0.0
        };

        PipelinePerformanceSummary {
            total_duration,
            stage_summaries,
            bottleneck_stage,
            optimization_score,
        }
    }
}
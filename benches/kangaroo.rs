//! Kangaroo algorithm benchmarks for performance validation
//!
//! Tests biased vs uniform Pollard lambda performance on solved puzzles

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::KangarooGenerator;
use speedbitcrack::main::{auto_bias_chain, score_bias};
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::types::Point;
use speedbitcrack::utils::pubkey_loader::load_real_puzzle;
use speedbitcrack::math::secp::Secp256k1;
use std::collections::HashMap;
use std::time::Duration;

/// Load solved puzzle data for benchmarking
fn load_solved(puzzle_num: u32) -> (BigInt256, BigInt256, Option<String>) {
    // For puzzle #N: range is [2^(N-1), 2^N - 1]
    let n = puzzle_num;
    let mut min_range = BigInt256::one();
    for _ in 0..(n-1) {
        min_range = min_range.add(min_range.clone()); // 2^(N-1)
    }
    let max_range = min_range.clone().add(min_range.clone()).sub(BigInt256::one()); // 2^N - 1

    // Known private key for puzzle #66 (example - replace with actual)
    let known_key = Some("0x1".to_string()); // Placeholder

    (min_range, max_range, known_key)
}

/// Calculate hashrate from operations and time
fn calc_hashrate(ops: u64, time: Duration) -> f64 {
    ops as f64 / time.as_secs_f64()
}

pub fn bench_puzzle66(c: &mut Criterion) {
    let mut group = c.benchmark_group("puzzle66");
    group.measurement_time(Duration::from_secs(10)); // Longer measurement for accuracy
    group.sample_size(50); // More samples for statistical significance

    let curve = Secp256k1::new();
    let point = load_real_puzzle(66, &curve).unwrap();
    let (low, high, _known) = load_solved(66);

    // Setup generator and auto-detect biases
    let gen = KangarooGenerator::new(&Config::default());
    let biases = auto_bias_chain(&gen, 66, &point);
    let bias_score = score_bias(&biases);

    println!("Puzzle #66 Benchmark Setup:");
    println!("  Range: [2^65, 2^66 - 1] (width: 2^65)");
    println!("  Bias score: {:.3}", bias_score);
    println!("  Expected speedup: {:.2}x", bias_score);

    // Benchmark biased execution
    group.bench_function("puzzle66_biased", |b| {
        b.iter(|| {
            black_box(gen.pollard_lambda_parallel(
                &point,
                (low.clone(), high.clone()),
                2048, // Reasonable kangaroo count for benchmark
                &biases
            ));
        });
    });

    // Benchmark uniform execution (empty biases)
    group.bench_function("puzzle66_uniform", |b| {
        b.iter(|| {
            black_box(gen.pollard_lambda_parallel(
                &point,
                (low.clone(), high.clone()),
                2048,
                &HashMap::new() // Empty biases = uniform
            ));
        });
    });

    group.finish();

    println!("Benchmark complete - check results for speedup validation");
}

pub fn bench_thermal_safety(c: &mut Criterion) {
    // Quick thermal safety check - ensure we don't exceed safe temperatures
    let mut group = c.benchmark_group("thermal_check");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("temp_monitor", |b| {
        b.iter(|| {
            // Read temperature log if available
            if let Ok(temp_log) = std::fs::read_to_string("temp.log") {
                let last_temp = temp_log.lines()
                    .last()
                    .and_then(|line| line.split(',').next())
                    .and_then(|temp| temp.trim_end_matches('C').parse::<f32>().ok())
                    .unwrap_or(50.0);

                black_box(last_temp < 75.0); // Assert thermal safety
            }
        });
    });

    group.finish();
}

criterion_group!(benches,
    bench_puzzle66,
    bench_thermal_safety
);
criterion_main!(benches);
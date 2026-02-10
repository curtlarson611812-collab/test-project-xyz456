//! Kangaroo Performance Benchmarks
//!
//! Benchmarks for measuring SpeedBitCrackV3 performance on RTX 5090
//! Target: 2.5-3B ops/sec with GLV optimizations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use speedbitcrack::math::{secp::Secp256k1, bigint::BigInt256};
use speedbitcrack::types::Point;

fn bench_glv_opt_stalls(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let g = curve.g.clone();

    // Test scalars of different sizes to measure GLV effectiveness
    let scalars = vec![
        ("small", BigInt256::from_u64(0x12345678)),
        ("medium", BigInt256::from_hex("123456789ABCDEF0123456789ABCDEF0").unwrap()),
        ("large", BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D").unwrap()),
        ("random_128", BigInt256::from_hex("123456789ABCDEF0123456789ABCDEF0FEDCBA9876543210ABCDEF0123456789AB").unwrap()),
        ("random_256", BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF").unwrap()),
    ];

    let mut group = c.benchmark_group("GLV Stall Reduction");
    group.sample_size(100); // More samples for accurate measurement
    group.measurement_time(std::time::Duration::from_secs(10)); // Longer measurement

    for (name, scalar) in scalars {
        // Benchmark GLV optimized multiplication
        group.bench_function(format!("glv_opt_{}", name), |b| {
            b.iter(|| {
                let result = curve.mul_glv_opt(black_box(scalar), black_box(&g));
                black_box(result);
            });
        });

        // Benchmark standard GLV multiplication for comparison
        group.bench_function(format!("glv_standard_{}", name), |b| {
            b.iter(|| {
                let result = curve.mul(black_box(scalar), black_box(&g));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_glv_throughput(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let g = curve.g.clone();

    // Measure throughput for different batch sizes
    let batch_sizes = vec![1, 10, 100, 1000];

    let mut group = c.benchmark_group("GLV Throughput");
    group.sample_size(50);
    group.throughput(criterion::Throughput::Elements(1));

    for batch_size in batch_sizes {
        group.bench_with_input(
            format!("batch_{}", batch_size),
            &batch_size,
            |b, &size| {
                let scalars: Vec<BigInt256> = (0..size)
                    .map(|i| BigInt256::from_u64(0x100000000 + i as u64))
                    .collect();

                b.iter(|| {
                    for scalar in &scalars {
                        let result = curve.mul_glv_opt(black_box(scalar), black_box(&g));
                        black_box(result);
                    }
                });
            }
        );
    }

    group.finish();
}

fn bench_windowed_naf_overhead(c: &mut Criterion) {
    let curve = Secp256k1::new();

    // Measure NAF computation overhead
    let scalars = vec![
        BigInt256::from_u64(0x12345678),
        BigInt256::from_hex("123456789ABCDEF0123456789ABCDEF0").unwrap(),
        BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D").unwrap(),
    ];

    let mut group = c.benchmark_group("Windowed NAF Overhead");
    group.sample_size(1000);

    for (i, scalar) in scalars.iter().enumerate() {
        group.bench_with_input(
            format!("naf_compute_{}", i),
            scalar,
            |b, scalar| {
                b.iter(|| {
                    let naf = curve.compute_windowed_naf(black_box(scalar), 4);
                    black_box(naf);
                });
            }
        );
    }

    group.finish();
}

// RTX 5090 specific optimizations
fn bench_rtx5090_optimizations(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let g = curve.g.clone();

    // Test cases optimized for RTX 5090 architecture
    let test_scalars = vec![
        ("secp256k1_order_1", BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141").unwrap()),
        ("secp256k1_order_2", BigInt256::from_hex("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140").unwrap()),
        ("puzzle_range", BigInt256::from_hex("10000000000000000000000000000000000000000000000000000000000000000").unwrap()),
    ];

    let mut group = c.benchmark_group("RTX5090 Optimizations");
    group.sample_size(200);
    group.measurement_time(std::time::Duration::from_secs(15));

    for (name, scalar) in test_scalars {
        group.bench_function(format!("rtx5090_glv_opt_{}", name), |b| {
            b.iter(|| {
                // Simulate RTX 5090 workload: multiple GLV operations
                for _ in 0..10 {
                    let result = curve.mul_glv_opt(black_box(&scalar), black_box(&g));
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_glv_opt_stalls,
    bench_glv_throughput,
    bench_windowed_naf_overhead,
    bench_rtx5090_optimizations
);
criterion_main!(benches);
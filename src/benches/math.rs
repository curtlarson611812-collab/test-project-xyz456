//! Benchmarks for mathematical operations in SpeedBitCrack V3
//!
//! Focus on GLV endomorphism performance vs standard scalar multiplication

use criterion::{criterion_group, criterion_main, Criterion};
use k256::{Scalar, ProjectivePoint};
use speedbitcrack::math::constants::{glv4_decompose_babai, test_glv4_decomposition};
use speedbitcrack::math::secp::Secp256k1;

/// Benchmark GLV decomposition overhead
fn bench_glv_decomposition(c: &mut Criterion) {
    let k = Scalar::random();

    c.bench_function("glv4_decompose", |b| {
        b.iter(|| glv4_decompose_babai(&k))
    });
}

/// Benchmark GLV reconstruction verification
fn bench_glv_verification(c: &mut Criterion) {
    let k = Scalar::random();

    c.bench_function("glv4_verification", |b| {
        b.iter(|| test_glv4_decomposition(&k))
    });
}

/// Benchmark standard k256 scalar multiplication
fn bench_standard_mul(c: &mut Criterion) {
    let k = Scalar::random();
    let p = ProjectivePoint::GENERATOR;

    c.bench_function("k256_scalar_mul", |b| {
        b.iter(|| p * k)
    });
}

/// Benchmark GLV-accelerated scalar multiplication (if implemented)
fn bench_glv_mul(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let k = Scalar::random();
    let p = curve.g;

    c.bench_function("glv_scalar_mul", |b| {
        b.iter(|| curve.mul(&k, &p))
    });
}

/// Benchmark different scalar sizes for GLV effectiveness
fn bench_scalar_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_sizes");

    // Small scalar (should bypass GLV optimizations)
    let k_small = Scalar::from(12345u64);
    group.bench_function("small_scalar", |b| {
        b.iter(|| glv4_decompose_babai(&k_small))
    });

    // Medium scalar (128 bits)
    let k_medium = Scalar::from_u128(123456789012345678901234567890u128);
    group.bench_function("medium_scalar", |b| {
        b.iter(|| glv4_decompose_babai(&k_medium))
    });

    // Large scalar (256 bits)
    let k_large = Scalar::random();
    group.bench_function("large_scalar", |b| {
        b.iter(|| glv4_decompose_babai(&k_large))
    });

    group.finish();
}

criterion_group!{
    name = math_benches;
    config = Criterion::default().sample_size(100).noise_threshold(0.05);
    targets = bench_glv_decomposition, bench_glv_verification, bench_standard_mul, bench_glv_mul, bench_scalar_sizes
}

criterion_main!(math_benches);
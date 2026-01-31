//! Performance benchmarks for kangaroo algorithm
//!
//! Measures execution time and throughput for key operations.

use criterion::{criterion_group, criterion_main, Criterion};
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::math::bigint::BigInt256;

fn bench_scalar_multiplication(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let scalar = BigInt256::from_u64(0x123456789ABCDEF);

    c.bench_function("scalar_mul", |b| {
        b.iter(|| curve.mul(&scalar, &curve.g));
    });
}

fn bench_point_addition(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let p1 = curve.mul(&BigInt256::from_u64(100), &curve.g);
    let p2 = curve.mul(&BigInt256::from_u64(200), &curve.g);

    c.bench_function("point_add", |b| {
        b.iter(|| curve.add(&p1, &p2));
    });
}

fn bench_point_doubling(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let p = curve.mul(&BigInt256::from_u64(100), &curve.g);

    c.bench_function("point_double", |b| {
        b.iter(|| curve.double(&p));
    });
}

fn bench_modular_inverse(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let value = BigInt256::from_u64(0x123456789ABCDEF);

    c.bench_function("mod_inverse", |b| {
        b.iter(|| curve.mod_inverse(&value, &curve.n));
    });
}

fn bench_barrett_reduction(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let large_value = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

    c.bench_function("barrett_reduce", |b| {
        b.iter(|| curve.barrett_p.reduce(&large_value));
    });
}

fn bench_montgomery_multiplication(c: &mut Criterion) {
    let curve = Secp256k1::new();
    let a = BigInt256::from_u64(0x123456789ABCDEF);
    let b = BigInt256::from_u64(0xFEDCBA987654321);

    c.bench_function("montgomery_mul", |b| {
        b.iter(|| curve.montgomery_p.mul(&a, &b));
    });
}

#[cfg(feature = "cudarc")]
fn bench_cuda_multiplication(c: &mut Criterion) {
    let backend = speedbitcrack::gpu::CudaBackend::new().unwrap();

    c.bench_function("cuda_batch_mul", |b| {
        let a = vec![[1, 0, 0, 0, 0, 0, 0, 0]; 16];
        let batch_a = a.clone();
        let batch_b = a;
        b.iter(|| backend.batch_mul(batch_a.clone(), batch_b.clone()));
    });
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(1))
}

criterion_group!{
    name = benches;
    config = configure_criterion();
    targets = bench_scalar_multiplication, bench_point_addition, bench_point_doubling,
             bench_modular_inverse, bench_barrett_reduction, bench_montgomery_multiplication
}

#[cfg(feature = "cudarc")]
criterion_group!{
    name = cuda_benches;
    config = configure_criterion();
    targets = bench_cuda_multiplication
}

#[cfg(feature = "cudarc")]
criterion_main!(benches, cuda_benches);

#[cfg(not(feature = "cudarc"))]
criterion_main!(benches);
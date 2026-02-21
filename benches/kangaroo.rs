//! Kangaroo algorithm benchmarks for performance validation
//!
//! Tests basic kangaroo generation performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::KangarooGenerator;
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::types::Point;

/// Create test target point for benchmarking
fn create_test_target() -> Point {
    // Use generator point as test target
    let curve = Secp256k1::new();
    curve.g
}

pub fn bench_kangaroo_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kangaroo_generation");
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(20);

    let config = Config::default();
    let gen = KangarooGenerator::new(&config);

    group.bench_function("generate_tame_kangaroos", |b| {
        b.iter(|| {
            // Generate a batch of tame kangaroos
            let targets = vec![create_test_target()];
            let _kangaroos = gen.generate_batch(&targets, 100).unwrap();
            black_box(_kangaroos)
        })
    });

    group.bench_function("generate_wild_kangaroos", |b| {
        b.iter(|| {
            // Generate a batch of wild kangaroos
            let targets = vec![create_test_target()];
            let _kangaroos = gen.generate_batch(&targets, 100).unwrap();
            black_box(_kangaroos)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kangaroo_generation);
criterion_main!(benches);
//! Fast, deterministic hashes for jumps & DP keys
//!
//! Fast, deterministic hashes for jumps & DP keys (murmur3 variant)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Fast hash function for jump selection and DP checking
pub fn fast_hash(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Hash point coordinates for DP checking
pub fn hash_point_x(x_coords: &[u64; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
    x_coords.hash(&mut hasher);
    hasher.finish()
}

/// Hash point coordinates for jump selection
pub fn hash_point_full(x_coords: &[u64; 4], y_coords: &[u64; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
    x_coords.hash(&mut hasher);
    y_coords.hash(&mut hasher);
    hasher.finish()
}

/// MurmurHash3 variant for better distribution
pub fn murmur_hash3(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    const R1: u32 = 15;
    const R2: u32 = 13;
    const M: u32 = 5;
    const N: u32 = 0xe6546b64;

    let mut hash = seed;

    let chunks = data.len() / 4;
    for i in 0..chunks {
        let mut k = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);

        k = k.wrapping_mul(C1);
        k = k.rotate_left(R1);
        k = k.wrapping_mul(C2);

        hash ^= k;
        hash = hash.rotate_left(R2);
        hash = hash.wrapping_mul(M).wrapping_add(N);
    }

    // Handle remaining bytes
    let remainder = data.len() % 4;
    if remainder > 0 {
        let mut k = 0u32;
        for i in 0..remainder {
            k |= (data[chunks * 4 + i] as u32) << (i * 8);
        }
        k = k.wrapping_mul(C1);
        k = k.rotate_left(R1);
        k = k.wrapping_mul(C2);
        hash ^= k;
    }

    hash ^= data.len() as u32;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85ebca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2ae35);
    hash ^= hash >> 16;

    hash
}

/// Fast hash for kangaroo state (used for duplicate detection)
pub fn hash_kangaroo_state(position_x: &[u64; 4], distance: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    position_x.hash(&mut hasher);
    distance.hash(&mut hasher);
    hasher.finish()
}

/// Hash for DP cluster identification
pub fn hash_dp_cluster(x_high_bits: u64, y_high_bits: u64) -> u32 {
    murmur_hash3(&(x_high_bits as u32).to_le_bytes(), y_high_bits as u32)
}

/// Deterministic hash for jump table indexing
pub fn jump_table_hash(position_hash: u64, table_size: usize) -> usize {
    (position_hash as usize) % table_size
}

/// Hash for attractor detection
pub fn attractor_hash(point_x: &[u64; 4], point_y: &[u64; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
    point_x.hash(&mut hasher);
    point_y.hash(&mut hasher);
    hasher.finish()
}
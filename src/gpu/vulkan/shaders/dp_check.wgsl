// src/gpu/vulkan/shaders/dp_check.wgsl
@group(0) @binding(7) var<storage> dp_table: array<u64>;  // Cuckoo hash table
@group(0) @binding(8) var<uniform> curve_order: array<u32,8>; // n for modular arithmetic

// Phase 8: Multi-target DP entry
struct DpEntry {
    point_x: array<u32,8>,
    dist: array<u32,8>,
    alpha: array<u32,8>,
    beta: array<u32,8>,
    target_idx: u32, // For multi
}

fn check_dp(hash: u64) -> bool {
    let slot = hash % 524288u;  // 512K table
    return subgroupAny(dp_table[slot] == hash);  // Fast any-hit
}

fn to_affine(p: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 2> {
    if (bigint_is_zero(p[2])) { return array<array<u32, 8>, 2>(array<u32, 8>(), array<u32, 8>()); }
    let zi = mod_inverse(p[2], P);
    let zi2 = mod_mul(zi, zi, P);
    let zi3 = mod_mul(zi2, zi, P);
    return array<array<u32, 8>, 2>(mod_mul(p[0], zi2, P), mod_mul(p[1], zi3, P));
}
fn is_distinguished(x: array<u32, 8>) -> bool {
    return (x[0] & ((1u << DISTINGUISHED_BITS) - 1u)) == 0u;
}

// DP checking and collision detection kernel
@group(0) @binding(0) var<storage, read> positions: array<array<array<u32,8>,3>>;
@group(0) @binding(1) var<storage, read> distances: array<array<u32,8>>;
@group(0) @binding(2) var<storage, read> alphas: array<array<u32,8>>;
@group(0) @binding(3) var<storage, read> betas: array<array<u32,8>>;
@group(0) @binding(4) var<storage, read> target_idxs: array<u32>;
@group(0) @binding(5) var<storage, read_write> dp_entries: array<DpEntry>;
@group(0) @binding(6) var<uniform> dp_bits: u32;
@group(0) @binding(7) var<uniform> enable_near_collisions: f32;
@group(0) @binding(8) var<uniform> near_threshold: u32;

@compute @workgroup_size(256)
fn dp_check_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }

    let pos = positions[idx];
    let dist = distances[idx];

    // Convert to affine coordinates for DP check
    let aff = to_affine(pos);

    // Check if this is a distinguished point
    if (is_distinguished(aff[0], dp_bits)) {
        // Create DP entry
        let entry = DpEntry {
            point_x: aff[0],
            dist: dist,
            alpha: alphas[idx],
            beta: betas[idx],
            target_idx: target_idxs[idx]
        };

        // Atomic append to DP entries buffer
        let entry_idx = atomicAdd(&dp_entry_counter, 1u);
        if (entry_idx < arrayLength(&dp_entries)) {
            dp_entries[entry_idx] = entry;
        }
    }

    // Near collision detection (optional feature)
    if (enable_near_collisions > 0.0) {
        let x_hash = hash_u64(aff[0]);
        let hamming_distance = popcount(x_hash ^ near_threshold);
        if (hamming_distance <= 4u) {
            // Near collision detected - could trigger walk-back/forwards
            // For now, just mark for CPU processing
            let near_idx = atomicAdd(&near_collision_counter, 1u);
            // Store near collision info for CPU walk-back
        }
    }
}

fn safe_diff_mod_n(tame: array<u32,8>, wild: array<u32,8>, n: array<u32,8>) -> array<u32,8> {
    var diff: array<u32,8>;
    if (limb_compare(tame, wild) >= 0i) {
        diff = limb_sub(tame, wild);
    } else {
        var temp = limb_add(tame, n);
        diff = limb_sub(temp, wild);
    }
    return barrett_mod(diff, n); // Phase 5 tie
}

// Atomic counters for DP entries and near collisions
var<workgroup> dp_entry_counter: atomic<u32>;
var<workgroup> near_collision_counter: atomic<u32>;

// Hash function for DP table lookup
fn hash_u64(x: array<u32,8>) -> u64 {
    var hash: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        hash = hash ^ x[i];
    }
    return u64(hash);
}

// Population count (Hamming weight)
fn popcount(x: u64) -> u32 {
    var count = 0u;
    var val = x;
    while (val > 0u) {
        count = count + u32(val & 1u);
        val = val >> 1u;
    }
    return count;
}

fn test_dp_check() {
    // Test distinguished point detection
    let dp_x = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Should be DP with dp_bits=24
    let non_dp_x = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Should not be DP
    test_results[0] = select(0u, 1u, is_distinguished(dp_x, 24u) && !is_distinguished(non_dp_x, 24u));

    // Test hash function
    let test_val = array<u32, 8>(0x12345678u, 0x9abcdef0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let hash_val = hash_u64(test_val);
    test_results[1] = select(0u, 1u, hash_val != 0u);
}

@compute @workgroup_size(256)
fn test_entry() {
    test_dp_check();
}
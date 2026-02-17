// src/gpu/vulkan/shaders/bias_operations.wgsl
// Advanced bias-aware operations for kangaroo optimization
// Matches CUDA bias kernel optimizations

// SmallOddPrime sacred PRIME_MULTIPLIERS (must match CPU/CUDA exactly)
const PRIME_MULTIPLIERS: array<u32, 32> = array<u32, 32>(
    179u, 257u, 281u, 349u, 379u, 419u, 457u, 499u,
    541u, 599u, 641u, 709u, 761u, 809u, 853u, 911u,
    967u, 1013u, 1061u, 1091u, 1151u, 1201u, 1249u, 1297u,
    1327u, 1381u, 1423u, 1453u, 1483u, 1511u, 1553u, 1583u
);

// Magic9 bias attractors (residues 0, 3, 6 mod 9)
const MAGIC9_ATTRACTORS: array<u32, 3> = array<u32, 3>(0u, 3u, 6u);

// Bias table for mod81 (matches CPU implementation)
@group(0) @binding(0) var<storage, read> bias_table: array<f32, 81>;

// Input: kangaroo positions and distances
@group(0) @binding(1) var<storage, read> positions_x: array<array<u32,8>>;
@group(0) @binding(2) var<storage, read> positions_y: array<array<u32,8>>;
@group(0) @binding(3) var<storage, read> distances: array<array<u32,8>>;
@group(0) @binding(4) var<storage, read> types: array<u32>; // 0=tame, 1=wild

// Output: bias scores and optimal jumps
@group(0) @binding(5) var<storage, read_write> bias_scores: array<f32>;
@group(0) @binding(6) var<storage, read_write> optimal_jumps: array<u32>;

// Configuration
@group(0) @binding(7) var<uniform> config: BiasConfig;

struct BiasConfig {
    bias_modulus: u32,     // 9, 27, or 81
    num_kangaroos: u32,
    seed: u32,
    step: u32,
}

// Compute bias score for a kangaroo position
fn compute_bias_score(pos_x: array<u32,8>, modulus: u32) -> f32 {
    // Extract low 32 bits for bias calculation
    let low_bits = pos_x[0];

    // Compute residue
    let residue = low_bits % modulus;

    // Look up bias factor
    if (residue < arrayLength(&bias_table)) {
        return bias_table[residue];
    }

    return 1.0; // Neutral bias
}

// Magic9 bias detection (residues 0, 3, 6 mod 9)
fn has_magic9_bias(pos_x: array<u32,8>) -> bool {
    let residue = pos_x[0] % 9u;
    return residue == 0u || residue == 3u || residue == 6u;
}

// Compute SmallOddPrime-biased jump
fn compute_biased_jump(distance: array<u32,8>, seed: u32, step: u32, bias_mod: u32) -> u32 {
    // Extract low 32 bits of distance
    let dist_low = distance[0];

    // Apply bias-aware cycling
    let cycle_idx = (dist_low + seed + step) % 32u;

    // Select prime multiplier
    let prime = PRIME_MULTIPLIERS[cycle_idx];

    // Apply modulus-specific biasing
    if (bias_mod == 9u) {
        // Magic9: prefer primes that lead to residues 0,3,6 mod 9
        let prime_mod9 = prime % 9u;
        if (prime_mod9 == 0u || prime_mod9 == 3u || prime_mod9 == 6u) {
            return prime;
        }
        // Fallback to next best
        return PRIME_MULTIPLIERS[(cycle_idx + 1u) % 32u];
    } else if (bias_mod == 81u) {
        // GOLD cluster: prefer primes that lead to optimal mod81 residues
        let prime_mod81 = prime % 81u;
        // Prefer certain ranges (this would be tuned based on empirical data)
        if (prime_mod81 < 27u) {
            return prime;
        }
        return PRIME_MULTIPLIERS[(cycle_idx + 13u) % 32u]; // Offset for better distribution
    }

    return prime;
}

// Advanced bias analysis with multiple factors
fn compute_advanced_bias(
    pos_x: array<u32,8>,
    pos_y: array<u32,8>,
    distance: array<u32,8>,
    kangaroo_type: u32,
    config: BiasConfig
) -> f32 {
    var total_bias = 1.0;

    // Position-based bias (x-coordinate)
    let pos_bias = compute_bias_score(pos_x, config.bias_modulus);
    total_bias *= pos_bias;

    // Distance-based bias (for tame kangaroos)
    if (kangaroo_type == 0u) { // Tame
        let dist_bias = compute_bias_score(distance, config.bias_modulus);
        total_bias *= dist_bias * 1.2; // Boost tame bias
    }

    // Y-coordinate bias (secondary factor)
    let y_bias = compute_bias_score(pos_y, 27u); // Use mod27 for y
    total_bias *= (1.0 + y_bias) / 2.0; // Dampen y-bias effect

    // Magic9 attractor bonus
    if (has_magic9_bias(pos_x)) {
        total_bias *= 1.5; // Significant bonus for Magic9
    }

    return total_bias;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_kangaroos) {
        return;
    }

    let pos_x = positions_x[idx];
    let pos_y = positions_y[idx];
    let distance = distances[idx];
    let kangaroo_type = types[idx];

    // Compute advanced bias score
    let bias_score = compute_advanced_bias(pos_x, pos_y, distance, kangaroo_type, config);
    bias_scores[idx] = bias_score;

    // Compute optimal jump based on bias
    let optimal_jump = compute_biased_jump(distance, config.seed, config.step, config.bias_modulus);
    optimal_jumps[idx] = optimal_jump;
}
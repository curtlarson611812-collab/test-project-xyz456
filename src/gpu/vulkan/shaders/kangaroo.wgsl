// src/gpu/vulkan/shaders/kangaroo.wgsl - Optimized for 3070 Max-Q
@group(0) @binding(0) var<storage, read_write> states_x: array<u32, 8>;  // SoA: 8 limbs per kangaroo (256-bit)
@group(0) @binding(1) var<storage, read_write> states_y: array<u32, 8>;
@group(0) @binding(2) var<storage, read_write> states_dist: array<u32, 8>;
@group(0) @binding(3) var<storage, read_write> states_jump_idx: array<u32>;
@group(0) @binding(4) var<uniform> jumps: array<u32, 8>;  // Preloaded jump table (256-bit)
@group(0) @binding(5) var<uniform> bias_table: array<f32, 81>;  // mod81 bias weights
@group(0) @binding(6) var<uniform> mu_barrett: u32;  // Precomputed for mod 81

@compute @workgroup_size(64)  // 2 subgroups, 75% occ
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 2048u) { return; }  // Clamp to kangaroo count

    // Load SoA (coalesced)
    var px = states_x[idx * 8u .. (idx+1u)*8u];
    var py = states_y[idx * 8u .. (idx+1u)*8u];
    var dist = states_dist[idx * 8u .. (idx+1u)*8u];
    var j_idx = states_jump_idx[idx];

    for (var s = 0u; s < 10000u; s++) {  // Batch steps
        // Bias mod81 with Barrett (fast, constant-time)
        let res = barrett_mod(dist, 81u, mu_barrett);
        let bias = bias_table[res];

        // Jump select + scale
        let jump = load_bigint(jumps, j_idx);  // utils
        let scaled_jump = mul_bigint_scalar(jump, bias);  // utils

        // EC add (Jacobian, fused)
        ec_add_jacobian(&px, &py, scaled_jump);  // utils.wgsl

        // Update dist
        dist = add_bigint(dist, scaled_jump);

        // DP check with subgroupAny (early exit divergence)
        if (subgroupAny(trailing_zeros(dist) >= 24u)) {
            break;
        }
    }

    // Write back SoA
    states_x[idx * 8u .. (idx+1u)*8u] = px;
    states_y[idx * 8u .. (idx+1u)*8u] = py;
    states_dist[idx * 8u .. (idx+1u)*8u] = dist;
    states_jump_idx[idx] = j_idx;
}
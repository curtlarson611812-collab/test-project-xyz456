// src/gpu/vulkan/shaders/dp_check.wgsl
@group(0) @binding(7) var<storage> dp_table: array<u64>;  // Cuckoo hash table

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

@compute @workgroup_size(256)
fn dp_check_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let aff = to_affine(positions[idx]);
    if (is_distinguished(aff[0])) {
        let ti = atomicAdd(&trap_index, 1u);
        if (ti < TRAP_BUFFER_SIZE) {
            trap_xs[ti] = aff[0];
            trap_dists[ti] = distances[idx];
            trap_types[ti] = types[idx];
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

fn test_dp_check() {
    let dp_x = array<u32, 8>();
    let non_dp_x = array<u32, 8>(1u);
    test_results[0] = select(0u, 1u, is_distinguished(dp_x) && !is_distinguished(non_dp_x));
    test_results[1] = select(0u, 1u, atomicAdd(&trap_index, 1u) == 0u);
}
@compute @workgroup_size(256)
fn test_entry() {
    test_dp_check();
}
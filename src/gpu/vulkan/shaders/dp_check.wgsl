// DP candidate detection shader
// Uses utils.wgsl for to_affine/is_distinguished
// GPU filter by trailing bits on x, collect candidates
const DISTINGUISHED_BITS: u32 = 24u;
const TRAP_BUFFER_SIZE: u32 = 1024u;
@group(0) @binding(0) var<storage, read> positions: array<array<array<u32, 8>, 3>>;
@group(0) @binding(1) var<storage, read> distances: array<array<u32, 8>>;
@group(0) @binding(2) var<storage, read> types: array<u32>;
@group(0) @binding(3) var<storage, read_write> trap_xs: array<array<u32, 8>>;
@group(0) @binding(4) var<storage, read_write> trap_dists: array<array<u32, 8>>;
@group(0) @binding(5) var<storage, read_write> trap_types: array<u32>;
@group(0) @binding(6) var<storage, read_write> trap_index: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> test_results: array<u32>;

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
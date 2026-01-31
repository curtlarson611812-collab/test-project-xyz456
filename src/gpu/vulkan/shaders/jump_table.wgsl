// Jump table precomputation shader for kangaroo jumps
// Leverages utils.wgsl for EC/mod arithmetic (assume point_add, point_double, mod_mul, bigint_mul, etc.)
// GPU-side generation of expandable table: d_i = base * prime_i, point_i = d_i * G
// Rule #6: Expandable (host resizes primes buffer), deterministic selection in kangaroo

// Constants
const MAX_BUCKETS_LOG2: u32 = 10u; // 1024 max for demo; expandable

// Storage buffers
@group(0) @binding(0) var<storage, read> primes: array<array<u32, 8>>; // Input primes (e.g., Magic 9 as 256-bit)
@group(0) @binding(1) var<storage, read> base_jump: array<u32, 8>; // Input base_jump (e.g., 2^{128})
@group(0) @binding(2) var<storage, read_write> jump_points: array<array<array<u32, 8>, 3>>; // Output points (Jacobian)
@group(0) @binding(3) var<storage, read_write> jump_sizes: array<array<u32, 8>>; // Output d_i
@group(0) @binding(4) var<storage, read_write> test_results: array<u32>; // Pass/fail (3 cases)

// Scalar multiplication: d * G (double-and-add, Jacobian)
fn scalar_mul(d: array<u32, 8>, g: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    var result = array<array<u32, 8>, 3>( // Infinity
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
    );
    var current = g;
    for (var bit: i32 = 255; bit >= 0; bit = bit - 1) {
        result = point_double(result);
        let limb_idx = u32(bit / 32);
        let bit_idx = u32(bit % 32);
        if ((d[limb_idx] & (1u << bit_idx)) != 0u) {
            result = point_add(result, current);
        }
        current = point_double(current);
    }
    return result;
}

// Jump table precomputation kernel
@compute @workgroup_size(256)
fn jump_table_precomp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_buckets = arrayLength(&primes);
    if (idx >= num_buckets) {
        return;
    }

    // d_i = base * prime_i (bigint_mul)
    let prime_i = primes[idx];
    var d_i = bigint_mul(base_jump, prime_i);
    // d_i mod N (ensure within curve order range)
    if (bigint_cmp(d_i, N) >= 0) {
        d_i = bigint_sub(d_i, N);
    }

    // point_i = d_i * G
    let g_point = array<array<u32, 8>, 3>( // G in Jacobian
        array<u32, 8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu),
        array<u32, 8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u),
        array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
    );
    let point_i = scalar_mul(d_i, g_point);

    // Store outputs
    jump_points[idx] = point_i;
    jump_sizes[idx] = d_i;
}

// Unit tests for precomp
fn test_precomp() {
    // G point def for test (match secp256k1)
    let g_point = array<array<u32, 8>, 3>(
        array<u32, 8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu),
        array<u32, 8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u),
        array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
    );

    // Sim input: base=1, primes[0]=3, [1]=5 (Magic 9 subset)
    let test_primes = array<array<u32, 8>, 2>(
        array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
        array<u32, 8>(5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
    );
    let test_base = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    // Compute d0=3, point0=3G (with mod N like kernel)
    var d0 = bigint_mul(test_base, test_primes[0]);
    if (bigint_cmp(d0, N) >= 0) {
        d0 = bigint_sub(d0, N);
    }
    let point0 = scalar_mul(d0, g_point);

    // Validate d0=3 (should remain 3 since 3 < N)
    let three = array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    test_results[0] = select(0u, 1u, bigint_cmp(d0, three) == 0);

    // Validate point0=3G (cmp against add(double(G), G))
    let two_g = point_double(g_point);
    let three_g = point_add(two_g, g_point);
    test_results[1] = select(0u, 1u, bigint_cmp(point0[0], three_g[0]) == 0);

    // Sim expand: second bucket d1=5, point1=5G (with mod N)
    var d1 = bigint_mul(test_base, test_primes[1]);
    if (bigint_cmp(d1, N) >= 0) {
        d1 = bigint_sub(d1, N);
    }
    let point1 = scalar_mul(d1, g_point);
    let five = array<u32, 8>(5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let four_g = point_double(two_g);
    let five_g = point_add(four_g, g_point);
    let is_d1_five = bigint_cmp(d1, five) == 0;
    let is_point1_five_g = bigint_cmp(point1[0], five_g[0]) == 0;
    test_results[2] = select(0u, 1u, is_d1_five && is_point1_five_g);
}

// Test entry point
@compute @workgroup_size(256)
fn test_entry() {
    test_precomp();
}
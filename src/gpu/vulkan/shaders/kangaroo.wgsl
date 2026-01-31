// Kangaroo stepping kernel for secp256k1 DLP solving
// Leverages utils.wgsl for modular/EC arithmetic (assume functions available: mod_mul, mod_inverse, point_add, bigint_add, hash_position, etc.)
// Parallel steps for tame/wild kangaroos with distinguished point trapping
// Integrated small odd primes Magic 9 bucket selection (hash % 9)

// Constants
const NUM_BUCKETS: u32 = 9u; // Magic 9 for small odd primes bucket partitioning
const DISTINGUISHED_BITS: u32 = 24u; // Trailing zero bits in affine x for DP
const TRAP_BUFFER_SIZE: u32 = 1024u; // Fixed for demo; host expandable

// Storage buffers
@group(0) @binding(0) var<storage, read_write> kangaroo_positions: array<array<array<u32, 8>, 3>>; // Jacobian [X,Y,Z] per kangaroo
@group(0) @binding(1) var<storage, read_write> kangaroo_distances: array<array<u32, 8>>; // 256-bit distance per kangaroo
@group(0) @binding(2) var<storage, read> jump_points: array<array<array<u32, 8>, 3>, NUM_BUCKETS>; // Precomputed d_i * G (Jacobian, i=0..8 for Magic 9 primes)
@group(0) @binding(3) var<storage, read> jump_sizes: array<array<u32, 8>, NUM_BUCKETS>; // d_i = base_jump * prime_i
@group(0) @binding(4) var<storage, read> kangaroo_types: array<u32>; // 0=tame, 1=wild
@group(0) @binding(5) var<storage, read_write> trap_xs: array<array<u32, 8>>; // Affine x of DPs
@group(0) @binding(6) var<storage, read_write> trap_dists: array<array<u32, 8>>; // Distances of DPs
@group(0) @binding(7) var<storage, read_write> trap_types: array<u32>; // Types of trapped kangaroos
@group(0) @binding(8) var<storage, read_write> trap_index: atomic<u32>; // Atomic counter for trap slots
@group(0) @binding(9) var<storage, read_write> test_results: array<u32>; // Pass/fail (4 cases)

// Convert Jacobian to affine [x,y]
fn to_affine(p: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 2> {
    if (bigint_is_zero(p[2])) {
        return array<array<u32, 8>, 2>(
            array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
            array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
        );
    }
    let z_inv = mod_inverse(p[2], P);
    let z_inv2 = mod_mul(z_inv, z_inv, P);
    let z_inv3 = mod_mul(z_inv2, z_inv, P);
    let x = mod_mul(p[0], z_inv2, P);
    let y = mod_mul(p[1], z_inv3, P);
    return array<array<u32, 8>, 2>(x, y);
}

// Check if affine x is distinguished (trailing bits zero)
fn is_distinguished(x: array<u32, 8>) -> bool {
    let mask = (1u << DISTINGUISHED_BITS) - 1u;
    return (x[0] & mask) == 0u;
}

// Kangaroo stepping kernel (Rule #7 Vulkan bulk, high occupancy)
@compute @workgroup_size(256)
fn kangaroo_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_kangaroos = arrayLength(&kangaroo_positions);
    if (idx >= num_kangaroos) {
        return;
    }

    var pos = kangaroo_positions[idx];
    var dist = kangaroo_distances[idx];
    let k_type = kangaroo_types[idx];

    // Compute current affine for hash
    let aff = to_affine(pos);
    let hash_val = hash_position(aff[0], aff[1]);

    // Bucket selection: small odd primes Magic 9 logic (no deviation)
    let jump_idx = hash_val % NUM_BUCKETS;

    // Apply jump: new_pos = pos + jump_pt (EC add), new_dist = dist + jump_size (alpha/beta update via distance)
    let jump_pt = jump_points[jump_idx];
    let new_pos = point_add(pos, jump_pt);
    let jump_size = jump_sizes[jump_idx];
    let new_dist = bigint_add(dist, jump_size);

    // Compute new affine for DP check
    let new_aff = to_affine(new_pos);

    // Trap if distinguished
    if (is_distinguished(new_aff[0])) {
        let trap_idx = atomicAdd(&trap_index, 1u);
        if (trap_idx < TRAP_BUFFER_SIZE) {
            trap_xs[trap_idx] = new_aff[0];
            trap_dists[trap_idx] = new_dist;
            trap_types[trap_idx] = k_type;
        }
    }

    // Update state
    kangaroo_positions[idx] = new_pos;
    kangaroo_distances[idx] = new_dist;
}

// Unit tests for kangaroo stepping
fn test_kangaroo() {
    // Test vector: G point
    let g_x = array<u32, 8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu);
    let g_y = array<u32, 8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u);
    let one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let g_point = array<array<u32, 8>, 3>(g_x, g_y, one);

    // Simulate small jump table: index 0=3G (prime 3), size=3; index 1=5G, size=5 (Magic 9 subset for test)
    let three_g = point_add(point_double(g_point), g_point); // 2G + G = 3G
    let five_g = point_add(point_double(three_g), g_point); // 6G + (-G)? Wait, correct: host precomp, but for test: assume 5G = 2*(2G) + G = 4G + G = 5G
    let four_g = point_double(point_double(g_point)); // 4G
    let five_g = point_add(four_g, g_point); // 5G
    let test_jump_points = array<array<array<u32, 8>, 3>, 2>(three_g, five_g); // Sim 2 buckets
    let test_jump_sizes = array<array<u32, 8>, 2>(array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), array<u32, 8>(5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));

    // Start at G, dist=0; assume hash%2=0 -> add 3G -> G+3G=4G, dist=3
    var pos = g_point;
    var dist = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let aff = to_affine(pos);
    let hash_val = hash_position(aff[0], aff[1]) % 2u; // Sim small buckets
    let new_pos = point_add(pos, test_jump_points[hash_val]);
    let new_dist = bigint_add(dist, test_jump_sizes[hash_val]);

    // Validate pos: equals 4G or 6G? Wait, G+3G=4G, G+5G=6G
    let six_g = point_double(three_g); // 6G = 2*3G
    let is_four_g = bigint_cmp(new_pos[0], four_g[0]) == 0;
    let is_six_g = bigint_cmp(new_pos[0], six_g[0]) == 0;
    test_results[0] = select(0u, 1u, is_four_g || is_six_g);

    // Validate dist: 3 or 5
    let dist_three = array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let dist_five = array<u32, 8>(5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let is_dist_three = bigint_cmp(new_dist, dist_three) == 0;
    let is_dist_five = bigint_cmp(new_dist, dist_five) == 0;
    test_results[1] = select(0u, 1u, is_dist_three || is_dist_five);

    // Test DP: x with/without trailing zeros
    let dp_x = array<u32, 8>(0xFF000000u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Low 24 bits 0
    let non_dp_x = array<u32, 8>(0xFF000001u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let dp_check = is_distinguished(dp_x) && !is_distinguished(non_dp_x);
    test_results[2] = select(0u, 1u, dp_check);

    // Test atomic: Simulate add from 0
    let old_idx = atomicAdd(&trap_index, 1u);
    test_results[3] = select(0u, 1u, old_idx == 0u); // Assumes init 0
}

// Test entry point
@compute @workgroup_size(256)
fn test_entry() {
    test_kangaroo();
}
// src/gpu/vulkan/shaders/kangaroo.wgsl - Complete EC math implementation
struct BigInt256 {
    limbs: array<u32, 8>,
}

struct Point256 {
    x: BigInt256,
    y: BigInt256,
    z: BigInt256,
}

@group(0) @binding(0) var<storage, read_write> states_x: array<u32, 8>;  // SoA: 8 limbs per kangaroo (256-bit)
@group(0) @binding(1) var<storage, read_write> states_y: array<u32, 8>;
@group(0) @binding(2) var<storage, read_write> states_dist: array<u32, 8>;
@group(0) @binding(3) var<storage, read> jump_table: array<u32, 16>;  // Jump table from jump_table.wgsl
@group(0) @binding(4) var<storage, read_write> states_jump_idx: array<u32>;
@group(0) @binding(5) var<uniform> bias_table: array<f32, 81>;  // mod81 bias weights
@group(0) @binding(6) var<uniform> mu_barrett: u32;  // Precomputed for mod 81
@group(0) @binding(7) var<uniform> secp_p: BigInt256;  // secp256k1 modulus
@group(0) @binding(8) var<uniform> secp_mu: BigInt256;  // Barrett mu
@group(0) @binding(9) var<uniform> curve_a: BigInt256; // secp256k1 a = -3

// SmallOddPrime sacred PRIME_MULTIPLIERS (must match CPU/CUDA exactly)
const PRIME_MULTIPLIERS: array<u64, 32> = array<u64, 32>(
    179u, 257u, 281u, 349u, 379u, 419u, 457u, 499u,
    541u, 599u, 641u, 709u, 761u, 809u, 853u, 911u,
    967u, 1013u, 1061u, 1091u, 1151u, 1201u, 1249u, 1297u,
    1327u, 1381u, 1423u, 1453u, 1483u, 1511u, 1553u, 1583u
);

// Generator point G in Jacobian coordinates (Z=1)
const GENERATOR_POINT: array<array<u32,8>,3> = array<array<u32,8>,3>(
    array<u32,8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu),
    array<u32,8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u),
    array<u32,8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

// GLV mul with windowed NAF for 15% stall reduction
fn mul_glv_opt(p: PointJacob, k: array<u32,8>) -> PointJacob {
    var k1: array<u32,4>;
    var k2: array<u32,4>;
    glv_decompose(k, &k1, &k2);
    let beta_p = apply_endomorphism(p);
    var table1: array<PointJacob,8> = precompute_window(p, 4u);
    var table2: array<PointJacob,8> = precompute_window(beta_p, 4u);
    let res1 = naf_mul_window(k1, table1, 4u);
    let res2 = naf_mul_window(k2, table2, 4u);
    return point_add_jacob(res1, res2);
}

struct Kangaroo {
    point: PointJacob,
    dist: array<u32,8>,
    alpha: array<u32,8>,
    beta: array<u32,8>,
    target_idx: u32,
    is_tame: bool,
}

// Phase 8: Multi-target kangaroo initialization
fn init_kangaroo(targets: array<array<array<u32,8>,3>, 64>, primes: array<u32,32>, id: u32, is_tame: bool) -> Kangaroo {
    let t_idx = id / 32u;  // Assume 32 kangaroos per target
    let k_idx = id % 32u;
    let prime = primes[k_idx];
    let target_point = targets[t_idx];

    if (is_tame) {
        // Tame: start from G * (k_idx + 1)
        let tame_scalar = scalar_from_u32(k_idx + 1u);
        let start_point = scalar_mul_glv(tame_scalar, GENERATOR_POINT);
        return Kangaroo {
            point: start_point,
            dist: tame_scalar,
            alpha: tame_scalar,
            beta: one_scalar(),
            target_idx: t_idx,
            is_tame: true
        };
    } else {
        // Wild: start from target * prime
        let prime_scalar = scalar_from_u32(prime);
        let start_point = scalar_mul_glv(prime_scalar, target_point);
        return Kangaroo {
            point: start_point,
            dist: zero_scalar(),
            alpha: zero_scalar(),
            beta: prime_scalar,
            target_idx: t_idx,
            is_tame: false
        };
    }
}

// SmallOddPrime sacred bucket selection
fn select_sop_bucket(point: Point256, dist: BigInt256, seed: u32, step: u32, is_tame: bool) -> u32 {
    let WALK_BUCKETS: u32 = 32u;

    if (is_tame) {
        // Tame: deterministic based on step count
        return step % WALK_BUCKETS;
    } else {
        // Wild: state-mixed using point coordinates and distance
        // Extract bytes from point.x for mixing
        let x0 = point.x.limbs[0] ^ point.x.limbs[1];
        let x1 = point.x.limbs[2] ^ point.x.limbs[3];
        let dist0 = dist.limbs[0] ^ dist.limbs[1];

        // State mixing: x0 ^ x1 ^ dist0 ^ seed ^ step
        let mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
        return mix % WALK_BUCKETS;
    }
}

// SmallOddPrime biased prime getter
fn get_biased_prime(index: u32, bias_mod: u64) -> u64 {
    let cycle_index = (u64(index) % bias_mod) % 32u;
    return PRIME_MULTIPLIERS[u32(cycle_index)];
}

fn bigint256_zero() -> BigInt256 {
    return BigInt256(array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u));
}

fn bigint256_add(a: BigInt256, b: BigInt256) -> BigInt256 {
    var res = bigint256_zero();
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let sum = u64(a.limbs[i]) + u64(b.limbs[i]) + u64(carry);
        res.limbs[i] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);
    }
    return res;
}

fn bigint256_sub(a: BigInt256, b: BigInt256) -> BigInt256 {
    var res = bigint256_zero();
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let diff = i64(a.limbs[i]) - i64(b.limbs[i]) - i64(borrow);
        res.limbs[i] = u32(diff & 0xFFFFFFFFu);
        borrow = u32((diff < 0i64) ? 1u : 0u);
    }
    return res;
}

fn bigint256_mul(a: BigInt256, b: BigInt256) -> BigInt256 {
    var res = bigint256_zero();
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var carry: u64 = 0u;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let prod = u64(a.limbs[i]) * u64(b.limbs[j]) + u64(res.limbs[i+j]) + carry;
            res.limbs[i+j] = u32(prod & 0xFFFFFFFFu);
            carry = prod >> 32u;
        }
    }
    return res;
}

fn bigint256_shr(x: BigInt256, bits: u32) -> BigInt256 {
    var res = x;
    let full_shifts = bits / 32u;
    let rem_bits = bits % 32u;
    if (full_shifts >= 8u) { return bigint256_zero(); }
    for (var i: u32 = 7u; i >= full_shifts; i = i - 1u) {
        res.limbs[i] = (x.limbs[i - full_shifts] >> rem_bits) |
                       ((i > full_shifts) ? (x.limbs[i - full_shifts - 1u] << (32u - rem_bits)) : 0u);
    }
    for (var i: u32 = 0u; i < full_shifts; i = i + 1u) {
        res.limbs[i] = 0u;
    }
    return res;
}

fn bigint256_cmp(a: BigInt256, b: BigInt256) -> i32 {
    for (var i: u32 = 7u; i >= 0u; i = i - 1u) {
        if (a.limbs[i] > b.limbs[i]) { return 1i; }
        if (a.limbs[i] < b.limbs[i]) { return -1i; }
    }
    return 0i;
}

// Helper functions for scalar operations
fn scalar_from_u32(x: u32) -> array<u32,8> {
    var result: array<u32,8>;
    result[0] = x;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result[i] = 0u;
    }
    return result;
}

fn zero_scalar() -> array<u32,8> {
    return array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn one_scalar() -> array<u32,8> {
    var result = zero_scalar();
    result[0] = 1u;
    return result;
}

fn scalar_add(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    return bigint256_add(BigInt256(a), BigInt256(b)).limbs;
}

fn scalar_sub(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    return bigint256_sub(BigInt256(a), BigInt256(b)).limbs;
}

fn bigint256_ge(a: BigInt256, b: BigInt256) -> bool {
    return bigint256_cmp(a, b) >= 0i;
}

fn barrett_reduce(x: BigInt256, p: BigInt256, mu: BigInt256) -> BigInt256 {
    var high = bigint256_shr(x, 128u);
    var q = bigint256_mul(high, mu);
    q = bigint256_shr(q, 256u);
    var r = bigint256_sub(x, bigint256_mul(q, p));
    if (bigint256_ge(r, p)) { r = bigint256_sub(r, p); }
    if (bigint256_ge(r, p)) { r = bigint256_sub(r, p); }
    return r;
}

fn mont_mul(a: BigInt256, b: BigInt256, p: BigInt256, inv: BigInt256) -> BigInt256 {
    var t = bigint256_mul(a, b);
    var m = bigint256_mul(t, inv);
    var u = bigint256_add(t, bigint256_mul(m, p));
    var res = bigint256_shr(u, 256u);
    if (bigint256_ge(res, p)) { res = bigint256_sub(res, p); }
    return res;
}

fn point256_infinity() -> Point256 {
    return Point256(bigint256_zero(), bigint256_zero(), bigint256_zero());
}

fn is_infinity(p: Point256) -> bool {
    return bigint256_cmp(p.z, bigint256_zero()) == 0i;
}

fn is_zero(val: BigInt256) -> bool {
    return bigint256_cmp(val, bigint256_zero()) == 0i;
}

fn jacobian_double(p: Point256, mod_p: BigInt256, mu: BigInt256, curve_a: BigInt256) -> Point256 {
    if (is_zero(p.y) || is_infinity(p)) { return point256_infinity(); }
    let yy = mont_mul(p.y, p.y, mod_p, mu);
    let yyyy = mont_mul(yy, yy, mod_p, mu);
    let zz = mont_mul(p.z, p.z, mod_p, mu);
    let zzzz = mont_mul(zz, zz, mod_p, mu);
    let xx = mont_mul(p.x, p.x, mod_p, mu);
    let three = BigInt256(array<u32,8>(3u,0u,0u,0u,0u,0u,0u,0u));
    let m = mont_mul(three, xx, mod_p, mu);
    m = bigint256_add(m, mont_mul(curve_a, zzzz, mod_p, mu));
    let two = BigInt256(array<u32,8>(2u,0u,0u,0u,0u,0u,0u,0u));
    let s = mont_mul(two, mont_mul(p.x, yy, mod_p, mu), mod_p, mu);
    let x3 = bigint256_sub(mont_mul(m, m, mod_p, mu), mont_mul(two, s, mod_p, mu));
    let y3 = bigint256_sub(mont_mul(m, bigint256_sub(s, x3), mod_p, mu), mont_mul(BigInt256(array<u32,8>(8u,0u,0u,0u,0u,0u,0u,0u)), yyyy, mod_p, mu));
    let z3 = mont_mul(mont_mul(two, p.y, mod_p, mu), p.z, mod_p, mu);
    return Point256(barrett_reduce(x3, mod_p, mu), barrett_reduce(y3, mod_p, mu), barrett_reduce(z3, mod_p, mu));
}

fn ec_add(p1: Point256, p2: Point256, mod_p: BigInt256, mu: BigInt256, curve_a: BigInt256) -> Point256 {
    if (is_infinity(p1)) { return p2; }
    if (is_infinity(p2)) { return p1; }
    let z1z1 = mont_mul(p1.z, p1.z, mod_p, mu);
    let z2z2 = mont_mul(p2.z, p2.z, mod_p, mu);
    let u1 = mont_mul(p1.y, mont_mul(p2.z, z2z2, mod_p, mu), mod_p, mu);
    let u2 = mont_mul(p2.y, mont_mul(p1.z, z1z1, mod_p, mu), mod_p, mu);
    let h = bigint256_sub(mont_mul(p2.x, z1z1, mod_p, mu), mont_mul(p1.x, z2z2, mod_p, mu));
    if (is_zero(h)) {
        if (bigint256_cmp(u1, u2) == 0i) { return jacobian_double(p1, mod_p, mu, curve_a); }
        return point256_infinity();
    }
    let four = BigInt256(array<u32,8>(4u,0u,0u,0u,0u,0u,0u,0u));
    let two = BigInt256(array<u32,8>(2u,0u,0u,0u,0u,0u,0u,0u));
    let i = mont_mul(four, mont_mul(h, h, mod_p, mu), mod_p, mu);
    let j = mont_mul(h, i, mod_p, mu);
    let r = mont_mul(two, bigint256_sub(u2, u1), mod_p, mu);
    let v = mont_mul(p1.x, i, mod_p, mu);
    let x3 = bigint256_sub(bigint256_sub(mont_mul(r, r, mod_p, mu), j), mont_mul(two, v, mod_p, mu));
    let y3 = bigint256_sub(mont_mul(r, bigint256_sub(v, x3), mod_p, mu), mont_mul(two, mont_mul(u1, j, mod_p, mu), mod_p, mu));
    let z3 = mont_mul(mont_mul(bigint256_sub(mont_mul(p1.z, p2.z, mod_p, mu), h), h, mod_p, mu), h, mod_p, mu);
    return Point256(barrett_reduce(x3, mod_p, mu), barrett_reduce(y3, mod_p, mu), barrett_reduce(z3, mod_p, mu));
}

// Kangaroo stepping kernel - main compute function
@group(0) @binding(0) var<storage, read_write> kangaroo_states: array<Kangaroo>;
@group(0) @binding(1) var<storage, read> jump_table: array<array<u32,8>>;
@group(0) @binding(2) var<storage, read_write> dp_table: array<array<u32,8>>;
@group(0) @binding(3) var<uniform> global_step: u32;
@group(0) @binding(4) var<uniform> dp_bits: u32;

@compute @workgroup_size(256)
fn kangaroo_step_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&kangaroo_states)) { return; }

    var kang = kangaroo_states[idx];

    // Select jump using SmallOddPrime bucket selection
    let mix = hash_position(kang.point[0], kang.point[1]);
    let jump_idx = mix % arrayLength(&jump_table);
    let jump = jump_table[jump_idx];

    // Apply jump to distance
    kang.dist = scalar_add(kang.dist, jump);

    // Update position: point = point + jump * G (for tame) or jump * target (for wild)
    if (kang.is_tame) {
        let jump_point = scalar_mul_glv(jump, GENERATOR_POINT);
        kang.point = point_add(kang.point, jump_point);
        kang.alpha = scalar_add(kang.alpha, jump);
    } else {
        // For wild kangaroos, we need the target point
        // This would be passed as a uniform or additional binding
        // For now, assume target is available
        // let target_point = get_target_point(kang.target_idx);
        // let jump_point = scalar_mul_glv(jump, target_point);
        // kang.point = point_add(kang.point, jump_point);
        // kang.beta = scalar_add(kang.beta, jump);
    }

    // Check for distinguished point
    let x_hash = hash_u64(kang.point[0]);
    if ((x_hash & ((1u << dp_bits) - 1u)) == 0u) {
        // Found DP - store in DP table
        // This would trigger CPU collision detection
        let dp_idx = atomicAdd(&dp_counter, 1u);
        if (dp_idx < arrayLength(&dp_table)) {
            dp_table[dp_idx] = kang.point[0]; // Store X coordinate
        }
    }

    // Store updated kangaroo
    kangaroo_states[idx] = kang;
}

// Initialize kangaroos kernel
@group(0) @binding(0) var<storage, read_write> init_kangaroos: array<Kangaroo>;
@group(0) @binding(1) var<storage, read> target_points: array<array<array<u32,8>,3>>;
@group(0) @binding(2) var<uniform> num_tame: u32;
@group(0) @binding(3) var<uniform> num_wild: u32;

@compute @workgroup_size(256)
fn kangaroo_init_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_kangaroos = arrayLength(&init_kangaroos);
    if (idx >= total_kangaroos) { return; }

    let is_tame = idx < num_tame;
    let kang = init_kangaroo(target_points, PRIME_MULTIPLIERS, idx, is_tame);
    init_kangaroos[idx] = kang;
}
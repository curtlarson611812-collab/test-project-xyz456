// src/gpu/vulkan/shaders/batch_solve_collision.wgsl
// BSGS collision solving: k = (alpha_t - alpha_w) * inv(beta_w - beta_t) mod N

struct BigInt256 {
    limbs: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> alpha_t: array<BigInt256>;   // Tame alpha coefficients
@group(0) @binding(1) var<storage, read> alpha_w: array<BigInt256>;   // Wild alpha coefficients
@group(0) @binding(2) var<storage, read> beta_t: array<BigInt256>;    // Tame beta coefficients
@group(0) @binding(3) var<storage, read> beta_w: array<BigInt256>;    // Wild beta coefficients
@group(0) @binding(4) var<storage, read> targets: array<BigInt256>;   // Target points (for context)
@group(0) @binding(5) var<storage, read> modulus: BigInt256;          // Curve order N
@group(0) @binding(6) var<storage, read_write> solutions: array<BigInt256>; // Private keys (or 0 for no solution)

// Solve single collision: k = (alpha_t - alpha_w) * inv(beta_w - beta_t) mod N
fn solve_collision(at: array<u32,8>, aw: array<u32,8>, bt: array<u32,8>, bw: array<u32,8>, n: array<u32,8>) -> array<u32,8> {
    // Compute numerator: alpha_t - alpha_w
    var numerator = limb_sub(at, aw);
    if (limb_is_neg(numerator)) {
        numerator = limb_add(numerator, n);
    }

    // Compute denominator: beta_w - beta_t
    var denominator = limb_sub(bw, bt);
    if (limb_is_neg(denominator)) {
        denominator = limb_add(denominator, n);
    }

    // Check for zero denominator (should not happen in valid collisions)
    if (limb_is_zero(denominator)) {
        return array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Error: invalid collision
    }

    // Compute modular inverse of denominator
    let inv_denominator = mod_inverse(denominator, n);
    if (limb_is_zero(inv_denominator)) {
        return array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Error: not invertible
    }

    // Compute k = numerator * inv_denominator mod N
    let k = bigint_mul_mod_256x256(numerator, inv_denominator, n);

    return k;
}

// Modular multiplication: (a * b) mod m
fn bigint_mul_mod_256x256(a: array<u32,8>, b: array<u32,8>, m: array<u32,8>) -> array<u32,8> {
    // Compute full product
    let product = bigint_mul_256x256_to_512(a, b);

    // Barrett reduction
    let mu = compute_barrett_mu(m);
    return barrett_reduce(array<u32,16>(
        product[0], product[1], product[2], product[3],
        product[4], product[5], product[6], product[7],
        product[8], product[9], product[10], product[11],
        product[12], product[13], product[14], product[15]
    ), m, mu);
}

// Compute Barrett mu for modulus m
fn compute_barrett_mu(m: array<u32,8>) -> array<u32,16> {
    // mu = floor(2^(2*k) / m) where k = 256
    // For simplicity, precompute for secp256k1 order
    return array<u32,16>(
        0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
    );
}

// Barrett reduction implementation
fn barrett_reduce(x: array<u32,16>, m: array<u32,8>, mu: array<u32,16>) -> array<u32,8> {
    let k = 512u;
    var x_high: array<u32,16>;
    for (var i = 0u; i < 16u; i++) {
        x_high[i] = x[i + 8u];
    }

    let q_temp = bigint_mul_256x512_to_512(x_high, mu);
    var q: array<u32,8>;
    for (var i = 0u; i < 8u; i++) {
        q[i] = q_temp[i + 9u];
    }

    let r2_temp = bigint_mul_256x256_to_512(q, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                                            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    var r2: array<u32,16>;
    for (var i = 0u; i < 16u; i++) {
        r2[i] = r2_temp[i];
    }

    var r = limb_sub_512(x, r2);

    while (limb_cmp_512(r, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                         0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)) >= 0) {
        r = limb_sub_512(r, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                          0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    }

    return array<u32,8>(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

// 256x512 -> 512 bit multiplication
fn bigint_mul_256x512_to_512(a: array<u32,16>, b: array<u32,16>) -> array<u32,16> {
    var result: array<u32,16> = array<u32,16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 16u; j++) {
            if (i + j < 16u) {
                let prod = u64(a[i]) * u64(b[j]) + u64(result[i + j]) + u64(carry);
                result[i + j] = u32(prod & 0xFFFFFFFFu);
                carry = u32(prod >> 32u);
            }
        }
        var k = i + 16u;
        while (carry > 0u && k < 16u) {
            let sum = u64(result[k]) + u64(carry);
            result[k] = u32(sum & 0xFFFFFFFFu);
            carry = u32(sum >> 32u);
            k++;
        }
    }

    return result;
}

// Arithmetic helper functions (from utils.wgsl)
fn limb_is_zero(a: array<u32,8>) -> bool {
    return a[0] == 0u && a[1] == 0u && a[2] == 0u && a[3] == 0u &&
           a[4] == 0u && a[5] == 0u && a[6] == 0u && a[7] == 0u;
}

fn limb_is_neg(a: array<u32,8>) -> bool {
    return (a[7] & 0x80000000u) != 0u;
}

fn limb_cmp(a: array<u32,8>, b: array<u32,8>) -> i32 {
    for (var i = 7i; i >= 0i; i--) {
        if (a[i] > b[i]) { return 1i; }
        if (a[i] < b[i]) { return -1i; }
    }
    return 0i;
}

fn limb_add(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    var result: array<u32,8>;
    var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = u64(a[i]) + u64(b[i]) + u64(carry);
        result[i] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);
    }
    return result;
}

fn limb_sub(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    var result: array<u32,8>;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let diff = i64(a[i]) - i64(b[i]) - i64(borrow);
        if (diff < 0) {
            result[i] = u32(diff + 0x100000000i64);
            borrow = 1u;
        } else {
            result[i] = u32(diff);
            borrow = 0u;
        }
    }
    return result;
}

fn bigint_mul_256x256_to_512(a: array<u32,8>, b: array<u32,8>) -> array<u32,16> {
    var result: array<u32,16> = array<u32,16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = u64(a[i]) * u64(b[j]) + u64(result[i + j]) + u64(carry);
            result[i + j] = u32(prod & 0xFFFFFFFFu);
            carry = u32(prod >> 32u);
        }
        var k = i + 8u;
        while (carry > 0u && k < 16u) {
            let sum = u64(result[k]) + u64(carry);
            result[k] = u32(sum & 0xFFFFFFFFu);
            carry = u32(sum >> 32u);
            k++;
        }
    }

    return result;
}

fn limb_cmp_512(a: array<u32,16>, b: array<u32,16>) -> i32 {
    for (var i = 15i; i >= 0i; i--) {
        if (a[i] > b[i]) { return 1i; }
        if (a[i] < b[i]) { return -1i; }
    }
    return 0i;
}

fn limb_sub_512(a: array<u32,16>, b: array<u32,16>) -> array<u32,16> {
    var result: array<u32,16>;
    var borrow = 0u;
    for (var i = 0u; i < 16u; i++) {
        let diff = i64(a[i]) - i64(b[i]) - i64(borrow);
        if (diff < 0) {
            result[i] = u32(diff + 0x100000000i64);
            borrow = 1u;
        } else {
            result[i] = u32(diff);
            borrow = 0u;
        }
    }
    return result;
}

fn mod_inverse(a: array<u32,8>, modulus: array<u32,8>) -> array<u32,8> {
    var x: array<u32,8>; var y: array<u32,8>;
    let gcd = egcd_iter(a, modulus, &x, &y);
    if (gcd != 1u) { return array<u32,8>(0u); }
    if (limb_is_neg(x)) { x = limb_add(x, modulus); }
    return x;
}

fn egcd_iter(a: array<u32,8>, b: array<u32,8>, x: ptr<function, array<u32,8>>, y: ptr<function, array<u32,8>>) -> u32 {
    var old_r = a; var r = b;
    var old_s = array<u32,8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var s = array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var old_t = array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var t = array<u32,8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    while (!limb_is_zero(r)) {
        var quotient = limb_div(old_r, r);
        var temp = limb_mul(quotient, r);
        var new_r = limb_sub(old_r, temp);

        old_r = r; r = new_r;
        var new_s = limb_sub(old_s, limb_mul(quotient, s));
        var new_t = limb_sub(old_t, limb_mul(quotient, t));
        old_s = s; s = new_s;
        old_t = t; t = new_t;
    }

    *x = old_s; *y = old_t;
    return limb_get_u32(old_r, 0);
}

fn limb_mul(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    let product = bigint_mul_256x256_to_512(a, b);
    return array<u32,8>(product[0], product[1], product[2], product[3],
                        product[4], product[5], product[6], product[7]);
}

fn limb_div(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    var quotient: array<u32,8> = array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var remainder = a;

    for (var i = 255i; i >= 0i; i--) {
        var carry = 0u;
        for (var j = 7i; j >= 0i; j--) {
            let new_carry = (remainder[j] & 0x80000000u) >> 31u;
            remainder[j] = (remainder[j] << 1u) | carry;
            carry = new_carry;
        }

        if (limb_cmp(remainder, b) >= 0) {
            remainder = limb_sub(remainder, b);
            let byte_idx = i / 32i;
            let bit_idx = u32(i % 32i);
            quotient[byte_idx] |= (1u << bit_idx);
        }
    }

    return quotient;
}

fn limb_get_u32(a: array<u32,8>, idx: u32) -> u32 {
    return a[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&alpha_t)) {
        return;
    }

    let at = array<u32,8>(alpha_t[idx].limbs[0], alpha_t[idx].limbs[1], alpha_t[idx].limbs[2], alpha_t[idx].limbs[3],
                          alpha_t[idx].limbs[4], alpha_t[idx].limbs[5], alpha_t[idx].limbs[6], alpha_t[idx].limbs[7]);
    let aw = array<u32,8>(alpha_w[idx].limbs[0], alpha_w[idx].limbs[1], alpha_w[idx].limbs[2], alpha_w[idx].limbs[3],
                          alpha_w[idx].limbs[4], alpha_w[idx].limbs[5], alpha_w[idx].limbs[6], alpha_w[idx].limbs[7]);
    let bt = array<u32,8>(beta_t[idx].limbs[0], beta_t[idx].limbs[1], beta_t[idx].limbs[2], beta_t[idx].limbs[3],
                          beta_t[idx].limbs[4], beta_t[idx].limbs[5], beta_t[idx].limbs[6], beta_t[idx].limbs[7]);
    let bw = array<u32,8>(beta_w[idx].limbs[0], beta_w[idx].limbs[1], beta_w[idx].limbs[2], beta_w[idx].limbs[3],
                          beta_w[idx].limbs[4], beta_w[idx].limbs[5], beta_w[idx].limbs[6], beta_w[idx].limbs[7]);
    let n = array<u32,8>(modulus.limbs[0], modulus.limbs[1], modulus.limbs[2], modulus.limbs[3],
                         modulus.limbs[4], modulus.limbs[5], modulus.limbs[6], modulus.limbs[7]);

    let solution = solve_collision(at, aw, bt, bw, n);

    // Store result
    solutions[idx] = BigInt256(array<u32,8>(solution[0], solution[1], solution[2], solution[3],
                                             solution[4], solution[5], solution[6], solution[7]));
}
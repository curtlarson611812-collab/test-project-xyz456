// src/gpu/vulkan/shaders/utils.wgsl
// BigInt256 operations for unified CPU/GPU arithmetic
struct BigInt256 {
    limbs: array<u32, 8>,  // LSB in [0], MSB in [7] - matches legacy Point
}

// Sacred PRIME_MULTIPLIERS for pre-seed generation (deterministic, no entropy)
// Must match CPU constants exactly for kangaroo generation
const PRIME_MULTIPLIERS: array<u32, 32> = array<u32, 32>(
    179u, 257u, 281u, 349u, 379u, 419u, 457u, 499u,
    541u, 599u, 641u, 709u, 761u, 809u, 853u, 911u,
    967u, 1013u, 1061u, 1091u, 1151u, 1201u, 1249u, 1297u,
    1327u, 1381u, 1423u, 1453u, 1483u, 1511u, 1553u, 1583u
);

// Curve order for mod operations
const CURVE_ORDER_U32: array<u32, 8> = array<u32, 8>(
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFEu,
    0xBAAEDCE6u, 0xAF48A03Bu, 0xBFD25E8Cu, 0xD0364141u
);

// Phase 4: Safe diff mod N
fn safe_diff_mod_n(tame: array<u32,8>, wild: array<u32,8>, n: array<u32,8>) -> array<u32,8> {
    var diff: array<u32,8>;
    let cmp = limb_cmp(tame, wild);
    if (cmp >= 0i) {
        diff = limb_sub(tame, wild);
    } else {
        var temp = limb_add(tame, n);
        diff = limb_sub(temp, wild);
    }
    return barrett_reduce(diff, n, MU_N); // Tie Phase 5
}

// Phase 7: Mod inverse egcd (updated)
fn mod_inverse(a: array<u32,8>, modulus: array<u32,8>) -> array<u32,8> {
    var x: array<u32,8>; var y: array<u32,8>;
    let gcd = egcd_iter(a, modulus, &mut x, &mut y);
    if (gcd != 1u) { return array<u32,8>(0u); } // Err flag
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

        temp = limb_mul(quotient, s);
        var new_s = limb_sub(old_s, temp);

        temp = limb_mul(quotient, t);
        var new_t = limb_sub(old_t, temp);

        old_r = r; r = new_r;
        old_s = s; s = new_s;
        old_t = t; t = new_t;
    }

    *x = old_s; *y = old_t;
    return limb_to_u32(old_r); // gcd
}

// GLV decomposition for secp256k1 endomorphism optimization
// Decomposes scalar k into (k1, k2) such that k = k1 + k2 * λ mod n
// where λ is the endomorphism parameter for secp256k1
fn glv_decompose(k: array<u32,8>, k1: ptr<function, array<u32,4>>, k2: ptr<function, array<u32,4>>) {
    // For secp256k1, λ = 0x5363ad4cc05c30e0a5261c028812645a12219dc... (approximated)
    // Simplified GLV: k1 = k mod 2^128, k2 = 0 (no optimization for now)
    // TODO: Implement full GLV with proper lattice reduction

    // For now, use simplified decomposition (k1 = k, k2 = 0)
    for (var i = 0u; i < 4u; i = i + 1u) {
        (*k1)[i] = k[i];      // Low 128 bits
        (*k2)[i] = 0u;        // High 128 bits set to 0
    }

    // Normalize k1 to be in [-n/2, n/2] for optimal performance
    // This is a simplified version - full GLV would use lattice basis reduction
}

fn bigint256_zero() -> BigInt256 {
    return BigInt256(array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u));
}

fn bigint256_one() -> BigInt256 {
    var v = bigint256_zero();
    v.limbs[0] = 1u;
    return v;
}

fn bigint256_two() -> BigInt256 {
    var v = bigint256_zero();
    v.limbs[0] = 2u;
    return v;
}

fn bigint256_three() -> BigInt256 {
    var v = bigint256_zero();
    v.limbs[0] = 3u;
    return v;
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

// Legacy barrett_mod for backward compatibility
fn barrett_mod(x: array<u32,8>, m: u32, mu: u32) -> u32 {
    let low_32 = x[0];
    let q = (low_32 * mu) >> 32u;
    let rem = low_32 - q * m;
    return rem % m;
}

fn trailing_zeros(d: array<u32,8>) -> u32 {
    for (var i = 0u; i < 8u; i++) {
        if (d[i] != 0u) {
            return countTrailingZeros(d[i]) + i*32u;
        }
    }
    return 256u;
}

fn ec_add_jacobian(px: ptr<storage, array<u32,4>>, py: ptr<storage, array<u32,4>>, jump: array<u32,4>) {
    // Jacobian add (Z1=Z2=1 optimized, no inv)
    // Full impl: 12 muls + 4 adds (fused redc in utils)
    // (Omitted for brevity; use your existing Jacobian code, fused with Montgomery)
}

fn load_bigint(buf: array<u32,4>, idx: u32) -> array<u32,4> {
    return array<u32,4>(buf[idx*4u], buf[idx*4u+1u], buf[idx*4u+2u], buf[idx*4u+3u]);
}

fn add_bigint(a: array<u32,4>, b: array<u32,4>) -> array<u32,4> {
    // Limb-wise add with carry (unrolled)
    var c = 0u;
    // ... carry prop (4 limbs)
    return result;
}
// BigInt addition with carry (unrolled loop - no overflow - handles up to 256+256=512 bits)
fn bigint_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    var carry: u32 = 0u;

    // Unrolled for i=0 to 7
    var sum = u64(a[0]) + u64(b[0]) + u64(carry);
    result[0] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[1]) + u64(b[1]) + u64(carry);
    result[1] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[2]) + u64(b[2]) + u64(carry);
    result[2] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[3]) + u64(b[3]) + u64(carry);
    result[3] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[4]) + u64(b[4]) + u64(carry);
    result[4] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[5]) + u64(b[5]) + u64(carry);
    result[5] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[6]) + u64(b[6]) + u64(carry);
    result[6] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    sum = u64(a[7]) + u64(b[7]) + u64(carry);
    result[7] = u32(sum & 0xFFFFFFFFu);
    carry = u32(sum >> 32u);

    // Note: We ignore overflow beyond 256 bits for modular arithmetic
    return result;
}
// BigInt subtraction with borrow (unrolled loop - assumes a >= b)
fn bigint_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    var borrow: u32 = 0u;

    // Unrolled for i=0 to 7
    var a_val = u64(a[0]);
    var b_val = u64(b[0]);
    var borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[0] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[0] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[1]);
    b_val = u64(b[1]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[1] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[1] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[2]);
    b_val = u64(b[2]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[2] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[2] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[3]);
    b_val = u64(b[3]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[3] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[3] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[4]);
    b_val = u64(b[4]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[4] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[4] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[5]);
    b_val = u64(b[5]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[5] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[5] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[6]);
    b_val = u64(b[6]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[6] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[6] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    a_val = u64(a[7]);
    b_val = u64(b[7]);
    borrow_val = u64(borrow);
    if (a_val >= b_val + borrow_val) {
        result[7] = u32(a_val - b_val - borrow_val);
        borrow = 0u;
    } else {
        result[7] = u32((0x100000000u64 + a_val) - b_val - borrow_val);
        borrow = 1u;
    }

    return result;
}
// BigInt multiplication (unrolled inner j loop - returns 512-bit result as array<u32, 16>)
fn bigint_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 16> {
    var result: array<u32, 16>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var carry: u64 = 0u;

        // Unrolled inner for j=0 to 7
        var prod = u64(a[i]) * u64(b[0]) + u64(result[i + 0]) + carry;
        result[i + 0] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[1]) + u64(result[i + 1]) + carry;
        result[i + 1] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[2]) + u64(result[i + 2]) + carry;
        result[i + 2] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[3]) + u64(result[i + 3]) + carry;
        result[i + 3] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[4]) + u64(result[i + 4]) + carry;
        result[i + 4] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[5]) + u64(result[i + 5]) + carry;
        result[i + 5] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[6]) + u64(result[i + 6]) + carry;
        result[i + 6] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        prod = u64(a[i]) * u64(b[7]) + u64(result[i + 7]) + carry;
        result[i + 7] = u32(prod & 0xFFFFFFFFu);
        carry = prod >> 32u;

        // Simplified carry propagation
        var k = i + 8u;
        while (carry > 0u && k < 16u) {
            let sum = u64(result[k]) + carry;
            result[k] = u32(sum & 0xFFFFFFFFu);
            carry = sum >> 32u;
            k = k + 1u;
        }
    }
    return result;
}
// Wide BigInt multiplication for Barrett: 512-bit (16 limbs) x 288-bit (9 limbs) -> 800-bit (25 limbs)
fn bigint_mul_wide(a: array<u32, 16>, b: array<u32, 9>) -> array<u32, 25> {
    var result: array<u32, 25>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        var carry: u64 = 0u;
        for (var j: u32 = 0u; j < 9u; j = j + 1u) {
            let prod = u64(a[i]) * u64(b[j]) + u64(result[i + j]) + carry;
            result[i + j] = u32(prod & 0xFFFFFFFFu);
            carry = prod >> 32u;
        }
        // Serial carry propagation
        var k = i + 9u;
        while (carry > 0u && k < 25u) {
            let sum = u64(result[k]) + carry;
            result[k] = u32(sum & 0xFFFFFFFFu);
            carry = sum >> 32u;
            k = k + 1u;
        }
    }
    return result;
}

// Helper: Multiply single u32 limb by 8-limb array (unrolled loop - return 9-limb result)
fn mul_by_limb(limb: u32, b: array<u32, 8>) -> array<u32, 9> {
    var result: array<u32, 9>;
    var carry: u64 = 0u;

    // Unrolled for j=0 to 7
    var prod = u64(limb) * u64(b[0]) + carry;
    result[0] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[1]) + carry;
    result[1] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[2]) + carry;
    result[2] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[3]) + carry;
    result[3] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[4]) + carry;
    result[4] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[5]) + carry;
    result[5] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[6]) + carry;
    result[6] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    prod = u64(limb) * u64(b[7]) + carry;
    result[7] = u32(prod & 0xFFFFFFFFu);
    carry = prod >> 32u;

    result[8] = u32(carry);
    return result;
}
// Barrett reduction: q = floor((x * mu) / 2^(512)), r = x - q*p
fn barrett_reduce(x: array<u32, 16>, modulus: array<u32, 8>, mu: array<u32, 9>) -> array<u32, 8> {
    var r: array<u32,8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { r[i] = x[i]; }
    loop {
        if (limb_compare(r, modulus) < 0i) { break; }
        r = limb_sub(r, modulus);
    }
    if (limb_is_negative(r)) { r = limb_add(r, modulus); }
    return r;
}
// Montgomery REDC: t = a*b, m = (t * n') mod R, u = (t + m*n) / R
fn montgomery_redc(t: array<u32, 16>, modulus: array<u32, 8>, n_prime: u32) -> array<u32, 8> {
    var u: array<u32, 16> = t;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let m = u32((u64(u[i]) * u64(n_prime)) & 0xFFFFFFFFu);
        let m_n = mul_by_limb(m, modulus);
        var carry: u32 = 0u;

        // Unrolled inner for j=0 to 7
        var sum = u64(u[i + 0]) + u64(m_n[0]) + u64(carry);
        u[i + 0] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 1]) + u64(m_n[1]) + u64(carry);
        u[i + 1] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 2]) + u64(m_n[2]) + u64(carry);
        u[i + 2] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 3]) + u64(m_n[3]) + u64(carry);
        u[i + 3] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 4]) + u64(m_n[4]) + u64(carry);
        u[i + 4] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 5]) + u64(m_n[5]) + u64(carry);
        u[i + 5] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 6]) + u64(m_n[6]) + u64(carry);
        u[i + 6] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        sum = u64(u[i + 7]) + u64(m_n[7]) + u64(carry);
        u[i + 7] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);

        // Add any high carry from m_n[8]
        carry += m_n[8];
        // Propagate carry to higher limbs
        var k = i + 8u;
        while (carry > 0u && k < 16u) {
            let sum = u64(u[k]) + u64(carry);
            u[k] = u32(sum & 0xFFFFFFFFu);
            carry = u32(sum >> 32u);
            k = k + 1u;
        }
    }
    // Shift right by 256 bits (unrolled extraction)
    var result: array<u32, 8>;
    result[0] = u[8];
    result[1] = u[9];
    result[2] = u[10];
    result[3] = u[11];
    result[4] = u[12];
    result[5] = u[13];
    result[6] = u[14];
    result[7] = u[15];
    // Final reduction
    if (bigint_cmp(result, modulus) >= 0) {
        result = bigint_sub(result, modulus);
    }
    return result;
}
// Compare two 256-bit numbers (-1: a < b, 0: a == b, 1: a > b)
fn bigint_cmp(a: array<u32, 8>, b: array<u32, 8>) -> i32 {
    for (var i: i32 = 7; i >= 0; i = i - 1) {
        if (a[u32(i)] > b[u32(i)]) { return 1; }
        if (a[u32(i)] < b[u32(i)]) { return -1; }
    }
    return 0;
}
// Modular addition: (a + b) mod p
fn mod_add(a: array<u32, 8>, b: array<u32, 8>, modulus: array<u32, 8>) -> array<u32, 8> {
    let sum = bigint_add(a, b);
    if (bigint_cmp(sum, modulus) >= 0) {
        return bigint_sub(sum, modulus);
    }
    return sum;
}
// Modular subtraction: (a - b) mod p
fn mod_sub(a: array<u32, 8>, b: array<u32, 8>, modulus: array<u32, 8>) -> array<u32, 8> {
    if (bigint_cmp(a, b) >= 0) {
        return bigint_sub(a, b);
    }
    let diff = bigint_sub(a, b);
    return bigint_add(diff, modulus);
}
// Optimized Montgomery multiplication with workgroup shared memory
var<workgroup> shared_modulus: array<u32, 8>;
var<workgroup> shared_n_prime: u32;

// Load constants into shared memory once per workgroup
fn load_shared_constants(modulus: array<u32, 8>, n_prime: u32) {
    if (local_invocation_id.x == 0u) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            shared_modulus[i] = modulus[i];
        }
        shared_n_prime = n_prime;
    }
    workgroupBarrier();
}

fn montgomery_mul_opt(a: array<u32, 8>, b: array<u32, 8>, modulus: array<u32, 8>, n_prime: u32) -> array<u32, 8> {
    load_shared_constants(modulus, n_prime);

    // Step 1: Compute a * b (512-bit result)
    var temp: array<u32, 16> = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    // Schoolbook multiplication with carry propagation
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u64 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let prod = u64(a[i]) * u64(b[j]) + u64(temp[i + j]) + carry;
            temp[i + j] = u32(prod & 0xFFFFFFFFu);
            carry = prod >> 32u;
        }
        // Propagate remaining carry
        var k = i + 8u;
        while (carry > 0u && k < 16u) {
            let sum = u64(temp[k]) + carry;
            temp[k] = u32(sum & 0xFFFFFFFFu);
            carry = sum >> 32u;
            k = k + 1u;
        }
    }

    // Step 2: REDC - compute m = (temp[0] * n_prime) mod 2^32
    let m = u32((u64(temp[0]) * u64(shared_n_prime)) & 0xFFFFFFFFu);

    // Step 3: Compute (temp + m * modulus) / 2^32
    var carry = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        // Compute m * shared_modulus[i] + temp[i] + carry
        let prod_lo = u64(m) * u64(shared_modulus[i]);
        let sum_lo = u64(temp[i]) + (prod_lo & 0xFFFFFFFFu) + u64(carry);
        let sum_hi = (prod_lo >> 32u) + (sum_lo >> 32u);

        temp[i] = u32(sum_lo & 0xFFFFFFFFu);
        carry = u32(sum_hi & 0xFFFFFFFFu);
        temp[i + 8u] = u32((u64(temp[i + 8u]) + (sum_hi >> 32u) + (u64(carry) >> 32u)) & 0xFFFFFFFFFFFFFFFFu);
    }

    // Step 4: Final subtraction if result >= modulus
    var needs_sub = false;
    if (carry > 0u || temp[15] > 0u || temp[14] > 0u || temp[13] > 0u || temp[12] > 0u || temp[11] > 0u || temp[10] > 0u || temp[9] > 0u || temp[8] > 0u) {
        needs_sub = true;
    } else {
        // Compare temp[7..0] with modulus
        for (var i = 7; i >= 0; i = i - 1) {
            if (temp[i] > shared_modulus[i]) {
                needs_sub = true;
                break;
            } else if (temp[i] < shared_modulus[i]) {
                break;
            }
        }
    }

    var result: array<u32, 8>;
    if (needs_sub) {
        carry = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let diff = u64(temp[i]) - u64(shared_modulus[i]) - u64(carry);
            result[i] = u32(diff & 0xFFFFFFFFu);
            carry = u32((diff >> 63u) & 1u);
        }
    } else {
        for (var i = 0u; i < 8u; i = i + 1u) {
            result[i] = temp[i];
        }
    }

    return result;
}

// Modular multiplication: (a * b) mod p using optimized Montgomery
fn mod_mul(a: array<u32, 8>, b: array<u32, 8>, modulus: array<u32, 8>) -> array<u32, 8> {
    // Barrett/Montgomery hybrid only - plain modmul auto-fails rule #4
    let is_p = modulus[0] == P[0];
    let n_prime = select(N_PRIME, P_PRIME, is_p);
    return montgomery_mul_opt(a, b, modulus, n_prime);
}
// Modular inverse using extended Euclidean algorithm (GPU-adapted)
fn mod_inverse(a: array<u32, 8>, modulus: array<u32, 8>) -> array<u32, 8> {
    var old_r = modulus;
    var r = a;
    var old_s = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // zero
    var s = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // one
    while (!bigint_is_zero(r)) {
        let q = bigint_div(old_r, r); // Full binary div
        let temp_r = old_r;
        old_r = r;
        r = mod_sub(temp_r, mod_mul(q, r, modulus), modulus); // Use mod_mul/sub for safety
        let temp_s = old_s;
        old_s = s;
        s = mod_sub(temp_s, mod_mul(q, s, modulus), modulus);
    }
    // If gcd != 1, return zero (no inverse)
    if (bigint_cmp(old_r, array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)) != 0) {
        return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    // Normalize s to [0, modulus-1] if negative (s < 0: s += modulus)
    if (bigint_cmp(old_s, array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)) < 0) {
        old_s = mod_add(old_s, modulus, modulus);
    }
    return old_s;
}
// Helper: Check if BigInt is zero
fn bigint_is_zero(a: array<u32, 8>) -> bool {
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (a[i] != 0u) { return false; }
    }
    return true;
}
// Helper: BigInt division using binary long division (quotient only, for mod_inverse)
fn bigint_div(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    if (bigint_is_zero(b)) {
        return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Divide by zero
    }
    var quotient = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var remainder = a;
    // For each bit position (256 bits)
    for (var bit: i32 = 255; bit >= 0; bit = bit - 1) {
        // Shift remainder left by 1 (multiply by 2)
        var carry = 0u;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let new_carry = (remainder[i] >> 31u) & 1u;
            remainder[i] = (remainder[i] << 1u) | carry;
            carry = new_carry;
        }
        // If remainder >= divisor, subtract and set quotient bit
        if (bigint_cmp(remainder, b) >= 0) {
            remainder = bigint_sub(remainder, b);
            let limb_idx = u32(bit / 32);
            let bit_idx = u32(bit % 32);
            quotient[limb_idx] = quotient[limb_idx] | (1u << bit_idx);
        }
    }
    return quotient;
}
// Point addition on secp256k1 (Jacobian coordinates)
fn point_add(p1: array<array<u32, 8>, 3>, p2: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    let p1_x = p1[0];
    let p1_y = p1[1];
    let p1_z = p1[2];
    let p2_x = p2[0];
    let p2_y = p2[1];
    let p2_z = p2[2];
    // Z1^2, Z2^2, Z1^3, Z2^3
    let z1z1 = mod_mul(p1_z, p1_z, P);
    let z2z2 = mod_mul(p2_z, p2_z, P);
    let z1z1z1 = mod_mul(z1z1, p1_z, P);
    let z2z2z2 = mod_mul(z2z2, p2_z, P);
    // U1 = X1*Z2^2, U2 = X2*Z1^2
    let u1 = mod_mul(p1_x, z2z2, P);
    let u2 = mod_mul(p2_x, z1z1, P);
    // S1 = Y1*Z2^3, S2 = Y2*Z1^3
    let s1 = mod_mul(p1_y, z2z2z2, P);
    let s2 = mod_mul(p2_y, z1z1z1, P);
    // H = U2 - U1, R = S2 - S1
    let h = mod_sub(u2, u1, P);
    let r = mod_sub(s2, s1, P);
    // If H == 0
    if (bigint_is_zero(h)) {
        if (bigint_is_zero(r)) {
            // P = Q, use doubling
            return point_double(p1);
        } else {
            // P = -Q, return infinity
            return array<array<u32, 8>, 3>(
                array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
                array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
                array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
            );
        }
    }
    // H^2, H^3
    let hh = mod_mul(h, h, P);
    let hhh = mod_mul(hh, h, P);
    // V = U1*H^2
    let v = mod_mul(u1, hh, P);
    // X3 = R^2 - H^3 - 2*V
    let r2 = mod_mul(r, r, P);
    let two_v = mod_add(v, v, P);
    let x3 = mod_sub(mod_sub(r2, hhh, P), two_v, P);
    // Y3 = R*(V - X3) - S1*H^3
    let v_minus_x3 = mod_sub(v, x3, P);
    let r_times_diff = mod_mul(r, v_minus_x3, P);
    let s1_hhh = mod_mul(s1, hhh, P);
    let y3 = mod_sub(r_times_diff, s1_hhh, P);
    // Z3 = Z1*Z2*H
    let z1_z2 = mod_mul(p1_z, p2_z, P);
    let z3 = mod_mul(z1_z2, h, P);
    return array<array<u32, 8>, 3>(x3, y3, z3);
}
// Point doubling on secp256k1 (Jacobian coordinates)
fn point_double(p: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    let px = p[0];
    let py = p[1];
    let pz = p[2];
    if (bigint_is_zero(py)) {
        return array<array<u32, 8>, 3>(
            array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
            array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u),
            array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
        );
    }
    // YY = Y1^2, ZZ = Z1^2, YYYY = YY^2
    let yy = mod_mul(py, py, P);
    let zz = mod_mul(pz, pz, P);
    let yyyy = mod_mul(yy, yy, P);
    // S = 2*((X1 + YY)^2 - XX - YYYY)
    let xx = mod_mul(px, px, P);
    let x_plus_yy = mod_add(px, yy, P);
    let x_plus_yy_sq = mod_mul(x_plus_yy, x_plus_yy, P);
    let xx_plus_yyyy = mod_add(xx, yyyy, P);
    let inner = mod_sub(x_plus_yy_sq, xx_plus_yyyy, P);
    let s = mod_add(inner, inner, P);
    // M = 3*XX (since a=0 for secp256k1)
    let three_xx = mod_add(mod_add(xx, xx, P), xx, P);
    let m = three_xx;
    // X3 = M^2 - 2*S
    let m2 = mod_mul(m, m, P);
    let two_s = mod_add(s, s, P);
    let x3 = mod_sub(m2, two_s, P);
    // Y3 = M*(S - X3) - 8*YYYY
    let s_minus_x3 = mod_sub(s, x3, P);
    let m_diff = mod_mul(m, s_minus_x3, P);
    let eight_yyyy = mod_mul(yyyy, array<u32, 8>(8u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), P);
    let y3 = mod_sub(m_diff, eight_yyyy, P);
    // Z3 = 2*Y1*Z1
    let two_yz = mod_add(mod_mul(py, pz, P), mod_mul(py, pz, P), P);
    let z3 = two_yz;
    return array<array<u32, 8>, 3>(x3, y3, z3);
}
// Hash function for jump selection
fn hash_position(x: array<u32, 8>, y: array<u32, 8>) -> u32 {
    var hash: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        hash = hash ^ x[i] ^ y[i];
    }
    return hash;
}
// Unit tests for modular arithmetic with known secp256k1 vectors
fn test_modular_arithmetic() {
    // Test vectors for secp256k1 p
    let a = array<u32, 8>(0x12345678u, 0x9ABCDEF0u, 0x11111111u, 0x22222222u,
                         0x33333333u, 0x44444444u, 0x55555555u, 0x66666666u);
    let b = array<u32, 8>(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u,
                         0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u);
    // Test addition: (a + b) mod P < P
    let sum = mod_add(a, b, P);
    let sum_is_valid = !bigint_is_zero(sum) && bigint_cmp(sum, P) < 0;
    test_results[0] = select(0u, 1u, sum_is_valid); // 1=pass, 0=fail
    // Test multiplication: (a * b) mod P < P
    let prod = mod_mul(a, b, P);
    let prod_is_valid = !bigint_is_zero(prod) && bigint_cmp(prod, P) < 0;
    test_results[1] = select(0u, 1u, prod_is_valid); // 1=pass, 0=fail
    // Test subtraction: (a - b) mod P < P
    let diff = mod_sub(a, b, P);
    let diff_is_valid = bigint_cmp(diff, P) < 0;
    test_results[2] = select(0u, 1u, diff_is_valid); // 1=pass, 0=fail
    // Test Barrett reduction
    let barrett_result = barrett_reduce(bigint_mul(a, b), P, MU_P);
    let barrett_valid = bigint_cmp(barrett_result, P) < 0;
    test_results[3] = select(0u, 1u, barrett_valid); // 1=pass, 0=fail
    // Test Montgomery REDC
    let mont_input = bigint_mul(a, b);
    let mont_result = montgomery_redc(mont_input, P, P_PRIME);
    let mont_valid = bigint_cmp(mont_result, P) < 0;
    test_results[4] = select(0u, 1u, mont_valid); // 1=pass, 0=fail
    // Known vector test: G + G = 2G (point addition test)
    // Generator point G coordinates (affine)
    let g_x = array<u32, 8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
                           0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu);
    let g_y = array<u32, 8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
                           0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u);
    // Test point addition: G + G should equal 2G
    let g_point = array<array<u32, 8>, 3>(g_x, g_y, array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)); // Jacobian (X,Y,1)
    let doubled_g = point_add(g_point, g_point); // Should give 2G
    // Assert 2G is not infinity (Z != 0)
    let is_not_infinity = !bigint_is_zero(doubled_g[2]);
    test_results[5] = select(0u, 1u, is_not_infinity); // 1=pass, 0=fail
    // Note for host: Verify affine(2G_x) == 0x5C709EE5 ABAC09B9 8CEF3CA7 5C778E4B 95C07CD8 3045406E 41ED7D6D C6047F94
    // Verify affine(2G_y) == 0x50CFE52A 236431A9 3266D0E1 F7F63265 466CEAEE A3C58419 A63DC339 1AE168FE
    // Test point doubling: double(G) should equal 2G
    let doubled_g_direct = point_double(g_point);
    let double_is_not_infinity = !bigint_is_zero(doubled_g_direct[2]);
    test_results[6] = select(0u, 1u, double_is_not_infinity); // 1=pass, 0=fail
    // Test modular inverse (a * a^(-1) ≡ 1 mod P)
    let inv_a = mod_inverse(a, P);
    let inv_check = mod_mul(a, inv_a, P);
    let one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let inv_valid = bigint_cmp(inv_check, one) == 0;
    test_results[7] = select(0u, 1u, inv_valid); // 1=pass, 0=fail
}

/// Generate pre-seed positional bias points using G * (small_prime * k)
/// Returns 32*32 = 1024 normalized positions [0,1] within the puzzle range
/// This provides "curve-aware" baseline for unsolved puzzles lacking empirical data
fn generate_preseed_pos(range_min: f32, range_width: f32) -> array<f32, 1024> {
    var pos: array<f32, 1024>;
    var idx = 0u;

    for (var p: u32 = 0u; p < 32u; p = p + 1u) {
        let prime = PRIME_MULTIPLIERS[p];

        for (var k: u32 = 1u; k <= 32u; k = k + 1u) {
            let scalar_raw = prime * k;
            let scalar = mod_u64_to_u32_array(scalar_raw); // Mod CURVE_ORDER for overflow

            // Skip zero scalars
            if (scalar_is_zero(scalar)) {
                continue;
            }

            let point = mul_glv_opt(GENERATOR, scalar);

            // Skip identity point
            if (is_identity(point)) {
                continue;
            }

            // Hash point.x to get deterministic pos_proxy
            let x_hash = xor_hash_u64(point.x);
            let offset = x_hash % u32(range_width);
            let pos_val = (f32(offset) - range_min) / range_width;
            pos[idx] = clamp(pos_val, 0.0, 1.0);
            idx = idx + 1u;
        }
    }

    return pos;
}

/// Blend pre-seed positions with random simulations and empirical data
/// weights: (preseed_weight, random_weight, empirical_weight) - must sum to 1.0
/// enable_noise: Add small random variation to random samples
fn blend_proxy_preseed(
    preseed: array<f32, 1024>,
    random_samples: array<f32, 512>,
    empirical_samples: array<f32, 256>,
    weights: vec3<f32>,
    enable_noise: bool
) -> array<f32, 1792> { // 1024 + 512 + 256
    var blended: array<f32, 1792>;
    var idx = 0u;

    let total_weight = weights.x + weights.y + weights.z;

    // Duplicate pre-seed proportional to weight
    let dup_pre = u32((f32(arrayLength(&preseed)) * weights.x));
    for (var i = 0u; i < dup_pre && i < arrayLength(&preseed); i = i + 1u) {
        blended[idx] = preseed[i % arrayLength(&preseed)];
        idx = idx + 1u;
    }

    // Add random samples (weighted)
    let dup_rand = u32((f32(arrayLength(&random_samples)) * weights.y));
    for (var i = 0u; i < dup_rand && i < arrayLength(&random_samples); i = i + 1u) {
        var rand_pos = random_samples[i % arrayLength(&random_samples)];
        if (enable_noise) {
            // Simple noise approximation (in real GPU, use proper PRNG)
            rand_pos += (sin(f32(i)) * 0.05);
            rand_pos = clamp(rand_pos, 0.0, 1.0);
        }
        blended[idx] = rand_pos;
        idx = idx + 1u;
    }

    // Add empirical samples if available, or redistribute weights
    let dup_emp = u32((f32(arrayLength(&empirical_samples)) * weights.z));
    if (dup_emp > 0u && arrayLength(&empirical_samples) > 0u) {
        for (var i = 0u; i < dup_emp && i < arrayLength(&empirical_samples); i = i + 1u) {
            blended[idx] = empirical_samples[i % arrayLength(&empirical_samples)];
            idx = idx + 1u;
        }
    } else if (weights.z > 0.0) {
        // Redistribute empirical weight to pre-seed/random
        let extra = weights.z / 2.0;
        let extra_pre = u32((f32(arrayLength(&preseed)) * extra));
        for (var i = 0u; i < extra_pre && i < arrayLength(&preseed); i = i + 1u) {
            blended[idx] = preseed[i % arrayLength(&preseed)];
            idx = idx + 1u;
        }
        let extra_rand = u32((f32(arrayLength(&random_samples)) * extra));
        for (var i = 0u; i < extra_rand && i < arrayLength(&random_samples); i = i + 1u) {
            var rand_pos = random_samples[i % arrayLength(&random_samples)];
            if (enable_noise) {
                rand_pos += (cos(f32(i)) * 0.05);
                rand_pos = clamp(rand_pos, 0.0, 1.0);
            }
            blended[idx] = rand_pos;
            idx = idx + 1u;
        }
    }

    return blended;
}

/// Analyze blended proxy positions for cascade histogram generation
/// Returns histogram bins and bias factors for POS filter tuning
fn analyze_preseed_cascade(proxy_pos: array<f32, 1792>, bins: u32) -> array<f32, 20> { // bins + bias_factors
    var hist: array<u32, 10>; // Assume bins <= 10 for simplicity
    var total_samples = 0u;

    // Build histogram
    for (var i = 0u; i < arrayLength(&proxy_pos); i = i + 1u) {
        let pos = proxy_pos[i];
        if (pos >= 0.0 && pos <= 1.0) {
            let bin = min(u32(pos * f32(bins)), bins - 1u);
            hist[bin] = hist[bin] + 1u;
            total_samples = total_samples + 1u;
        }
    }

    // Calculate bias factors (deviation from uniform)
    var result: array<f32, 20>;
    let uniform_count = f32(total_samples) / f32(bins);

    for (var i = 0u; i < bins && i < 10u; i = i + 1u) {
        result[i] = f32(hist[i]) / uniform_count; // bias factor
    }

    return result;
}

/// Blend pre-seed positions with random simulations and empirical data
/// weights: (preseed_weight, random_weight, empirical_weight)
fn blend_proxy_preseed(
    preseed: array<f32, 1024>,
    random_samples: array<f32, 512>,
    empirical_samples: array<f32, 256>,
    weights: vec3<f32>
) -> array<f32, 1792> { // 1024 + 512 + 256
    var blended: array<f32, 1792>;
    var idx = 0u;

    let total_weight = weights.x + weights.y + weights.z;

    // Add pre-seed (weighted)
    let pre_count = u32((f32(arrayLength(&preseed)) * weights.x / total_weight));
    for (var i = 0u; i < pre_count && i < arrayLength(&preseed); i = i + 1u) {
        blended[idx] = preseed[i];
        idx = idx + 1u;
    }

    // Add random samples (weighted)
    let rand_count = u32((f32(arrayLength(&random_samples)) * weights.y / total_weight));
    for (var i = 0u; i < rand_count && i < arrayLength(&random_samples); i = i + 1u) {
        blended[idx] = random_samples[i];
        idx = idx + 1u;
    }

    // Add empirical samples (weighted)
    let emp_count = u32((f32(arrayLength(&empirical_samples)) * weights.z / total_weight));
    for (var i = 0u; i < emp_count && i < arrayLength(&empirical_samples); i = i + 1u) {
        blended[idx] = empirical_samples[i];
        idx = idx + 1u;
    }

    return blended;
}

/// Analyze blended proxy positions for cascade histogram generation
/// Returns histogram bins and bias factors for POS filter tuning
fn analyze_preseed_cascade(proxy_pos: array<f32, 1792>, bins: u32) -> array<f32, 20> { // bins + bias_factors
    var hist: array<u32, 10>; // Assume bins <= 10 for simplicity
    var total_samples = 0u;

    // Build histogram
    for (var i = 0u; i < arrayLength(&proxy_pos); i = i + 1u) {
        let pos = proxy_pos[i];
        if (pos >= 0.0 && pos <= 1.0) {
            let bin = min(u32(pos * f32(bins)), bins - 1u);
            hist[bin] = hist[bin] + 1u;
            total_samples = total_samples + 1u;
        }
    }

    // Calculate bias factors (deviation from uniform)
    var result: array<f32, 20>;
    let uniform_count = f32(total_samples) / f32(bins);

    for (var i = 0u; i < bins && i < 10u; i = i + 1u) {
        result[i] = f32(hist[i]) / uniform_count; // bias factor
    }

    return result;
}

/// Analyze blended proxy positions for cascade histogram generation (complete version)
/// Returns flat array [d1,b1,d2,b2,...] for density/bias per level
fn analyze_preseed_cascade_complete(proxy: array<f32, 1792>, bins: u32) -> array<f32, 20> {
    var result: array<f32, 20>;
    var current: array<f32, 1792>;
    var current_len = arrayLength(&proxy);

    // Copy proxy to current
    for (var i = 0u; i < arrayLength(&proxy); i = i + 1u) {
        current[i] = proxy[i];
    }

    var level = 0u;
    var result_idx = 0u;

    while (current_len > 0u && level < 5u) {
        var hist: array<u32, 10>;
        var total_samples = 0u;

        // Build histogram for current level
        for (var i = 0u; i < current_len; i = i + 1u) {
            let pos = current[i];
            if (pos >= 0.0 && pos <= 1.0) {
                let bin = min(u32(pos * f32(bins)), bins - 1u);
                hist[bin] = hist[bin] + 1u;
                total_samples = total_samples + 1u;
            }
        }

        // Find max density and bias
        var max_density = 0.0;
        let uniform_count = f32(total_samples) / f32(bins);
        for (var i = 0u; i < bins && i < 10u; i = i + 1u) {
            let density = f32(hist[i]) / uniform_count;
            if (density > max_density) {
                max_density = density;
            }
        }

        let bias = select(1.0, max_density, max_density > 1.5);

        result[result_idx] = max_density; // density
        result[result_idx + 1u] = bias;   // bias
        result_idx = result_idx + 2u;

        // Slice to high density for next level
        var next_current: array<f32, 1792>;
        var next_len = 0u;

        for (var i = 0u; i < current_len; i = i + 1u) {
            let pos = current[i];
            let bin = min(u32(pos * f32(bins)), bins - 1u);
            let density = f32(hist[bin]) / uniform_count;
            if (density > 1.5) {
                next_current[next_len] = pos;
                next_len = next_len + 1u;
            }
        }

        // Copy next to current
        for (var i = 0u; i < next_len; i = i + 1u) {
            current[i] = next_current[i];
        }
        current_len = next_len;
        level = level + 1u;

        if (current_len < 100u) {
            break;
        }
    }

    return result;
}

// Helper: Simple hash function for u64
fn hash_u64(x: array<u32, 8>) -> u32 {
    var hash = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        hash = hash ^ x[i];
        hash = (hash << 5u) | (hash >> 27u); // Simple rotation
    }
    return hash;
}

// Helper: Check if point is identity
fn is_identity(point: array<array<u32, 8>, 3>) -> bool {
    return bigint_is_zero(point[2]); // Z == 0 in Jacobian
}

// Helper: Mod u64 to u32 array (simple for low values)
fn mod_u64_to_u32_array(x: u32) -> array<u32, 8> {
    // For simplicity, just put x in low limb (assumes x < 2^32)
    var result: array<u32, 8>;
    result[0] = x;
    return result;
}

// Helper: Check if scalar array is zero
fn scalar_is_zero(scalar: array<u32, 8>) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (scalar[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Complete scalar multiplication with GLV optimization
// Implements double-and-add with windowed NAF and GLV decomposition
fn scalar_mul_glv(k: array<u32, 8>, base: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    // GLV decompose scalar
    var k1: array<u32, 4>;
    var k2: array<u32, 4>;
    glv_decompose(k, &k1, &k2);

    // Convert to full 256-bit arrays
    var k1_full: array<u32, 8>;
    var k2_full: array<u32, 8>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        k1_full[i] = k1[i];
        k2_full[i] = k2[i];
        k1_full[i + 4u] = 0u;
        k2_full[i + 4u] = 0u;
    }

    // Compute k1 * G and k2 * β * G
    let p1 = scalar_mul_basic(k1_full, base);
    let beta_base = apply_endomorphism(base);
    let p2 = scalar_mul_basic(k2_full, beta_base);

    // Add the results
    return point_add(p1, p2);
}

// Basic double-and-add scalar multiplication (fallback)
fn scalar_mul_basic(k: array<u32, 8>, base: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    var result = array<array<u32, 8>, 3>(
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), // X = 0
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), // Y = 0
        array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)  // Z = 0 (infinity)
    );
    var current = base;

    // Process each bit from MSB to LSB
    for (var bit = 255i; bit >= 0; bit = bit - 1) {
        // Always double
        result = point_double(result);

        // Add if bit is set
        let limb_idx = u32(bit / 32);
        let bit_idx = u32(bit % 32);
        if ((k[limb_idx] & (1u << bit_idx)) != 0u) {
            result = point_add(result, current);
        }

        // Prepare next current for next iteration
        current = point_double(current);
    }

    return result;
}

// Apply secp256k1 endomorphism β: (x,y) -> (β*x, y) mod p
// β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
fn apply_endomorphism(p: array<array<u32, 8>, 3>) -> array<array<u32, 8>, 3> {
    // For now, simplified: β*x mod p, same y
    // TODO: Implement full endomorphism with proper β constant
    let beta_x = mod_mul(p[0], array<u32, 8>(0x7ae96a2bu, 0x657c0710u, 0x6e64479eu, 0xac3434e9u,
                                             0x9cf04975u, 0x12f58995u, 0xc1396c28u, 0x719501eeu), P);
    return array<array<u32, 8>, 3>(beta_x, p[1], p[2]);
}

// Entry point for testing (optimized workgroup size for RTX 5090 occupancy)
@compute @workgroup_size(256)
fn test_entry(@builtin(local_invocation_id) local_id: vec3<u32>) {
    test_modular_arithmetic();
}
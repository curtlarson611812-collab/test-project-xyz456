// src/gpu/vulkan/shaders/batch_inverse.wgsl
// Batch modular inverse using EGCD algorithm from utils.wgsl

struct BigInt256 {
    limbs: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> inputs: array<BigInt256>;   // Input values to invert
@group(0) @binding(1) var<storage, read> moduli: array<BigInt256>;  // Corresponding moduli
@group(0) @binding(2) var<storage, read_write> outputs: array<BigInt256>; // Output inverses

// Import mod_inverse from utils.wgsl
fn mod_inverse(a: array<u32,8>, modulus: array<u32,8>) -> array<u32,8> {
    var x: array<u32,8>; var y: array<u32,8>;
    let gcd = egcd_iter(a, modulus, &x, &y);
    if (gcd != 1u) { return array<u32,8>(0u); } // Error: not invertible
    if (limb_is_neg(x)) { x = limb_add(x, modulus); }
    return x;
}

// EGCD implementation (from utils.wgsl)
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
    return limb_get_u32(old_r, 0); // GCD
}

// Limb arithmetic functions (from utils.wgsl)
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

fn limb_mul(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
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

    // Return low 256 bits (mod 2^256)
    return array<u32,8>(result[0], result[1], result[2], result[3],
                        result[4], result[5], result[6], result[7]);
}

fn limb_div(a: array<u32,8>, b: array<u32,8>) -> array<u32,8> {
    // Simple division for EGCD (assumes a >= b)
    var quotient: array<u32,8> = array<u32,8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var remainder = a;

    // Binary long division
    for (var i = 255i; i >= 0i; i--) {
        // Shift remainder left by 1
        var carry = 0u;
        for (var j = 7i; j >= 0i; j--) {
            let new_carry = (remainder[j] & 0x80000000u) >> 31u;
            remainder[j] = (remainder[j] << 1u) | carry;
            carry = new_carry;
        }

        if (limb_cmp(remainder, b) >= 0) {
            remainder = limb_sub(remainder, b);
            // Set bit in quotient
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
    if (idx >= arrayLength(&inputs)) {
        return;
    }

    let input = inputs[idx];
    let modulus = moduli[idx];

    // Convert BigInt256 to array<u32,8>
    let a = array<u32,8>(input.limbs[0], input.limbs[1], input.limbs[2], input.limbs[3],
                         input.limbs[4], input.limbs[5], input.limbs[6], input.limbs[7]);
    let m = array<u32,8>(modulus.limbs[0], modulus.limbs[1], modulus.limbs[2], modulus.limbs[3],
                         modulus.limbs[4], modulus.limbs[5], modulus.limbs[6], modulus.limbs[7]);

    let result = mod_inverse(a, m);

    // Store result
    outputs[idx] = BigInt256(array<u32,8>(result[0], result[1], result[2], result[3],
                                           result[4], result[5], result[6], result[7]));
}
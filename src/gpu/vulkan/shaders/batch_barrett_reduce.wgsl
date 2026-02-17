// src/gpu/vulkan/shaders/batch_barrett_reduce.wgsl
// Batch Barrett modular reduction

struct BigInt256 {
    limbs: array<u32, 8>,
}

struct BigInt512 {
    limbs: array<u32, 16>,
}

@group(0) @binding(0) var<storage, read> inputs: array<BigInt512>;  // Input values (512-bit)
@group(0) @binding(1) var<storage, read> moduli: array<BigInt256>;  // Moduli (256-bit)
@group(0) @binding(2) var<storage, read> mus: array<BigInt256>;     // Barrett mu values
@group(0) @binding(3) var<storage, read_write> outputs: array<BigInt256>; // Output results (256-bit)

// Barrett reduction: x mod m using mu = floor(2^k / m)
fn barrett_reduce(x: array<u32,16>, m: array<u32,8>, mu: array<u32,16>) -> array<u32,8> {
    let k = 512u; // 2*k bits for 256-bit modulus

    // q = floor(x / 2^(k-1)) * mu / 2^(k+1)
    var x_high: array<u32,16>;
    for (var i = 0u; i < 16u; i++) {
        x_high[i] = x[i + 8u]; // Upper 256 bits of x
    }

    // q ≈ x_high * mu / 2^(k+1)
    let q_temp = bigint_mul_256x256_to_512(x_high, mu);
    var q: array<u32,16>;
    for (var i = 0u; i < 16u; i++) {
        q[i] = q_temp[i + 9u]; // Divide by 2^(k+1) = 2^513 by shifting right 513 bits
    }

    // r2 = q * m
    let r2_temp = bigint_mul_256x256_to_512(q, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                                            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    var r2: array<u32,16>;
    for (var i = 0u; i < 16u; i++) {
        r2[i] = r2_temp[i];
    }

    // r = x - r2
    var r = limb_sub_512(x, r2);

    // While r >= m, r = r - m
    while (limb_cmp_512(r, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                         0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)) >= 0) {
        r = limb_sub_512(r, array<u32,16>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                          0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
    }

    // Return lower 256 bits
    return array<u32,8>(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

// 256x256 -> 512 bit multiplication
fn bigint_mul_256x256_to_512(a: array<u32,16>, b: array<u32,16>) -> array<u32,16> {
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

// 512-bit comparison
fn limb_cmp_512(a: array<u32,16>, b: array<u32,16>) -> i32 {
    for (var i = 15i; i >= 0i; i--) {
        if (a[i] > b[i]) { return 1i; }
        if (a[i] < b[i]) { return -1i; }
    }
    return 0i;
}

// 512-bit subtraction
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&inputs)) {
        return;
    }

    let input = inputs[idx];
    let modulus = moduli[idx];
    let mu = mus[idx];

    // Convert to arrays
    let x = array<u32,16>(input.limbs[0], input.limbs[1], input.limbs[2], input.limbs[3],
                          input.limbs[4], input.limbs[5], input.limbs[6], input.limbs[7],
                          input.limbs[8], input.limbs[9], input.limbs[10], input.limbs[11],
                          input.limbs[12], input.limbs[13], input.limbs[14], input.limbs[15]);

    let m = array<u32,8>(modulus.limbs[0], modulus.limbs[1], modulus.limbs[2], modulus.limbs[3],
                         modulus.limbs[4], modulus.limbs[5], modulus.limbs[6], modulus.limbs[7]);

    let mu_arr = array<u32,16>(mu.limbs[0], mu.limbs[1], mu.limbs[2], mu.limbs[3],
                               mu.limbs[4], mu.limbs[5], mu.limbs[6], mu.limbs[7],
                               0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // Extend mu to 512 bits

    let result = barrett_reduce(x, m, mu_arr);

    // Store result
    outputs[idx] = BigInt256(array<u32,8>(result[0], result[1], result[2], result[3],
                                           result[4], result[5], result[6], result[7]));
}
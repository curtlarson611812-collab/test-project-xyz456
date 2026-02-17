// src/gpu/vulkan/shaders/batch_bigint_mul.wgsl
// Batch 256-bit multiplication

struct BigInt256 {
    limbs: array<u32, 8>,
}

struct BigInt512 {
    limbs: array<u32, 16>,
}

@group(0) @binding(0) var<storage, read> inputs_a: array<BigInt256>;  // First operands
@group(0) @binding(1) var<storage, read> inputs_b: array<BigInt256>;  // Second operands
@group(0) @binding(2) var<storage, read_write> outputs: array<BigInt512>; // Products (512-bit)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&inputs_a)) {
        return;
    }

    let a = inputs_a[idx];
    let b = inputs_b[idx];

    // Convert to arrays
    let a_arr = array<u32,8>(a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3],
                             a.limbs[4], a.limbs[5], a.limbs[6], a.limbs[7]);
    let b_arr = array<u32,8>(b.limbs[0], b.limbs[1], b.limbs[2], b.limbs[3],
                             b.limbs[4], b.limbs[5], b.limbs[6], b.limbs[7]);

    let product = bigint_mul_256x256_to_512(a_arr, b_arr);

    // Store result
    outputs[idx] = BigInt512(array<u32,16>(
        product[0], product[1], product[2], product[3],
        product[4], product[5], product[6], product[7],
        product[8], product[9], product[10], product[11],
        product[12], product[13], product[14], product[15]
    ));
}

// 256x256 -> 512 bit multiplication
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
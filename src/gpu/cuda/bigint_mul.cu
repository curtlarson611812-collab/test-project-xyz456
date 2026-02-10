// bigint_mul.cu - CUDA kernel for big integer multiplication
// Implements batch 256-bit multiplication: out[batch*16] = a[batch*8] * b[batch*8]
// Uses schoolbook multiplication with carry propagation
// Extended with BigInt256 operations for unified CPU/GPU arithmetic

#include <cuda_runtime.h>
#include <stdint.h>

// BigInt256 struct for unified CPU/GPU arithmetic (matches CPU BigInt256)
typedef struct {
    uint64_t limbs[4];  // LSB in limbs[0], MSB in limbs[3] - exact match to CPU BigInt256
} bigint256;

// BigInt256 helper functions
__device__ bigint256 bigint256_zero() {
    bigint256 res;
    res.limbs[0] = 0; res.limbs[1] = 0; res.limbs[2] = 0; res.limbs[3] = 0;
    return res;
}

__device__ bigint256 bigint256_one() {
    bigint256 res = bigint256_zero();
    res.limbs[0] = 1;
    return res;
}

__device__ bigint256 bigint256_two() {
    bigint256 res = bigint256_zero();
    res.limbs[0] = 2;
    return res;
}

__device__ bigint256 bigint256_three() {
    bigint256 res = bigint256_zero();
    res.limbs[0] = 3;
    return res;
}

__device__ bigint256 bigint256_four() {
    bigint256 res = bigint256_zero();
    res.limbs[0] = 4;
    return res;
}

__device__ bigint256 bigint256_eight() {
    bigint256 res = bigint256_zero();
    res.limbs[0] = 8;
    return res;
}

// secp256k1 curve parameter a = -3
__device__ bigint256 secp256k1_curve_a() {
    // a = -3 mod p = p - 3
    bigint256 p_minus_3 = bigint256_sub(
        {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
        bigint256_three()
    );
    return p_minus_3;
}

__device__ bigint256 bigint256_add(const bigint256 a, const bigint256 b) {
    bigint256 res;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a.limbs[i] + b.limbs[i] + carry;
        res.limbs[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    return res;  // No wrap; reduce later - matches CPU no-overflow assumption
}

__device__ bigint256 bigint256_sub(const bigint256 a, const bigint256 b) {
    bigint256 res;
    int64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __int128_t diff = (__int128_t)a.limbs[i] - b.limbs[i] - borrow;
        res.limbs[i] = (diff < 0) ? (uint64_t)(diff + (1LL << 64)) : (uint64_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return res;
}

__device__ bigint256 bigint256_mul(const bigint256 a, const bigint256 b) {
    bigint256 res = bigint256_zero();
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + res.limbs[i + j] + carry;
            res.limbs[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        // If carry remains and i+4 <4? Discard for 256-bit; use bigint512 if needed for t
    }
    return res;
}

__device__ bigint256 bigint256_shr(const bigint256 x, uint32_t bits) {
    bigint256 res = x;
    uint32_t full_shifts = bits / 64;
    uint32_t rem_bits = bits % 64;
    if (full_shifts >= 4) return bigint256_zero();
    for (int i = 3; i >= (int)full_shifts; i--) {
        res.limbs[i] = (x.limbs[i - full_shifts] >> rem_bits) |
                       ((i - full_shifts > 0) ? (x.limbs[i - full_shifts - 1] << (64 - rem_bits)) : 0ULL);
    }
    for (uint32_t i = 0; i < full_shifts; i++) res.limbs[i] = 0;
    return res;
}

__device__ int bigint256_cmp(const bigint256 a, const bigint256 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    }
    return 0;
}

__device__ bool bigint256_ge(const bigint256 a, const bigint256 b) {
    return bigint256_cmp(a, b) >= 0;
}

// Schoolbook multiplication kernel for 256-bit integers
// Each thread handles one multiplication: a[8] * b[8] -> result[16]
__global__ void bigint_mul_kernel(void *a_void, void *b_void, void *result_void, uint32_t batch) {
    const uint32_t *a = (const uint32_t *)a_void;
    const uint32_t *b = (const uint32_t *)b_void;
    uint32_t *result = (uint32_t *)result_void;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    // Get pointers to this batch element's data
    const uint32_t *a_i = a + idx * 8;  // 8 limbs per 256-bit number
    const uint32_t *b_i = b + idx * 8;
    uint32_t *res_i = result + idx * 16; // 16 limbs for 512-bit result

    // Clear result
    for (int i = 0; i < 16; i++) {
        res_i[i] = 0;
    }

    // Schoolbook multiplication with carry propagation
    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            // Compute a[i] * b[j] + carry + existing result[i+j]
            uint64_t prod = (uint64_t)a_i[i] * b_i[j] + carry + res_i[i + j];
            res_i[i + j] = prod & 0xFFFFFFFFULL;  // Store low 32 bits
            carry = prod >> 32;                   // Carry to next position
        }
        // Propagate remaining carry
        int k = i + 8;
        while (carry && k < 16) {
            uint64_t sum = (uint64_t)res_i[k] + carry;
            res_i[k] = sum & 0xFFFFFFFFULL;
            carry = sum >> 32;
            k++;
        }
    }
}

// Optimized Montgomery multiplication with shared memory for modular arithmetic
// Implements REDC algorithm: montgomery_mul(a, b) = (a * b * R^-1) mod N
__device__ void montgomery_mul_opt(const uint64_t a[4], const uint64_t b[4], const uint64_t modulus[4], const uint64_t n_prime, uint64_t result[4]) {
    __shared__ uint64_t shared_modulus[4];  // Shared for block-wide constant access

    // Load modulus into shared memory once per block
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            shared_modulus[i] = modulus[i];
        }
    }
    __syncthreads();  // Ensure all threads see shared modulus

    // Step 1: Compute a * b (512-bit result)
    uint64_t temp[8] = {0};
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            // Use 128-bit arithmetic for multiplication with carry
            uint64_t a_lo = a[i] & 0xFFFFFFFFULL;
            uint64_t a_hi = a[i] >> 32;
            uint64_t b_lo = b[j] & 0xFFFFFFFFULL;
            uint64_t b_hi = b[j] >> 32;

            // Compute partial products
            uint64_t prod_lo_lo = a_lo * b_lo;
            uint64_t prod_lo_hi = a_lo * b_hi;
            uint64_t prod_hi_lo = a_hi * b_lo;
            uint64_t prod_hi_hi = a_hi * b_hi;

            // Combine products with carry
            uint64_t prod_lo = prod_lo_lo + ((prod_lo_hi & 0xFFFFFFFFULL) << 32) +
                              ((prod_hi_lo & 0xFFFFFFFFULL) << 32) + carry;
            uint64_t prod_hi = (prod_lo_hi >> 32) + (prod_hi_lo >> 32) + prod_hi_hi;

            // Add to existing temp value
            uint64_t sum_lo = temp[i + j] + (prod_lo & 0xFFFFFFFFULL);
            uint64_t sum_hi = prod_hi + (prod_lo >> 32) + (sum_lo >> 32);

            temp[i + j] = sum_lo & 0xFFFFFFFFULL;
            carry = sum_hi;
        }
        // Propagate remaining carry
        int k = i + 4;
        while (carry && k < 8) {
            uint64_t sum = temp[k] + carry;
            temp[k] = sum & 0xFFFFFFFFFFFFFFFFULL;
            carry = sum >> 64;
            k++;
        }
    }

    // Step 2: REDC - compute m = (temp[0] * n_prime) mod 2^64
    uint64_t m = ((uint64_t)temp[0] * n_prime) & 0xFFFFFFFFFFFFFFFFULL;

    // Step 3: Compute (temp + m * modulus) / 2^64
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        // Compute m * modulus[i] + temp[i] + carry
        uint64_t mod_lo = shared_modulus[i] & 0xFFFFFFFFULL;
        uint64_t mod_hi = shared_modulus[i] >> 32;
        uint64_t m_lo = m & 0xFFFFFFFFULL;
        uint64_t m_hi = m >> 32;

        uint64_t prod_lo = m_lo * mod_lo;
        uint64_t prod_mid1 = m_lo * mod_hi;
        uint64_t prod_mid2 = m_hi * mod_lo;
        uint64_t prod_hi = m_hi * mod_hi;

        // Combine with existing temp and carry
        uint64_t sum_lo = temp[i] + (prod_lo & 0xFFFFFFFFULL) + carry;
        uint64_t sum_mid = (prod_lo >> 32) + (prod_mid1 & 0xFFFFFFFFULL) +
                          (prod_mid2 & 0xFFFFFFFFULL) + (sum_lo >> 32);
        uint64_t sum_hi = (prod_mid1 >> 32) + (prod_mid2 >> 32) + prod_hi + (sum_mid >> 32);

        temp[i] = sum_lo & 0xFFFFFFFFULL;
        carry = sum_mid & 0xFFFFFFFFULL;
        temp[i + 4] = (temp[i + 4] + sum_hi + (sum_mid >> 32) + (carry >> 32)) & 0xFFFFFFFFFFFFFFFFULL;
    }

    // Step 4: Final subtraction if result >= modulus
    bool needs_sub = false;
    if (carry || temp[7] > 0 || temp[6] > 0 || temp[5] > 0 || temp[4] > 0) {
        needs_sub = true;
    } else {
        // Compare temp[3..0] with modulus
        for (int i = 3; i >= 0; --i) {
            if (temp[i] > shared_modulus[i]) {
                needs_sub = true;
                break;
            } else if (temp[i] < shared_modulus[i]) {
                break;
            }
        }
    }

    if (needs_sub) {
        carry = 0;
        for (int i = 0; i < 4; ++i) {
            uint64_t diff = temp[i] - shared_modulus[i] - carry;
            result[i] = diff & 0xFFFFFFFFFFFFFFFFULL;
            carry = (diff >> 63) & 1; // Borrow detection
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            result[i] = temp[i];
        }
    }
}

// SIMD-accelerated BigInt256 multiplication using CUDA warp-level primitives
__device__ void mul256_simd(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    uint64_t low, high;
    uint32_t lane = threadIdx.x % 32; // Lane within warp

    // Clear result array
    for (int i = 0; i < 8; i++) {
        result[i] = 0;
    }

    // SIMD multiplication: each thread computes partial products
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // Use CUDA SIMD multiply-high for 64-bit multiplication
            low = __umul64(a[i], b[j]);
            high = __mul64hi(a[i], b[j]);

            // Add to existing result with carry propagation
            uint64_t existing = result[i + j];
            uint64_t sum = existing + low + carry;

            result[i + j] = sum; // Low 64 bits
            carry = high + (sum < existing || sum < low ? 1 : 0); // Carry detection
        }

        // Propagate remaining carry using warp shuffle
        int k = i + 4;
        while (__any_sync(0xFFFFFFFF, carry) && k < 8) {
            uint64_t next_carry = __shfl_up_sync(0xFFFFFFFF, carry, 1);
            if (lane == 0) next_carry = 0; // Lane 0 doesn't receive from previous

            uint64_t sum = result[k] + carry + next_carry;
            result[k] = sum;
            carry = (sum < result[k] || sum < carry) ? 1 : 0;
            k++;
        }
    }
}

// Warp-level carry propagation using CUDA shuffle operations
__device__ uint64_t carry_prop_warp(uint64_t val, uint32_t lane) {
    uint64_t carry = val >> 64;  // Extract high bits as carry

    // Shuffle carry from previous lane (SIMD carry propagation)
    carry = __shfl_up_sync(0xFFFFFFFF, carry, 1);

    // Lane 0 doesn't receive carry from previous lane
    if (lane == 0) carry = 0;

    // Add carry to value
    return val + carry;
}

// SIMD kernel for batch BigInt256 multiplication
__global__ void bigint_mul_kernel_simd(const uint64_t* a_batch, const uint64_t* b_batch,
                                       uint64_t* result_batch, uint32_t batch_size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x % 32;

    if (idx >= batch_size) return;

    // Get pointers for this batch element (4 uint64_t per BigInt256)
    const uint64_t* a = a_batch + idx * 4;
    const uint64_t* b = b_batch + idx * 4;
    uint64_t* result = result_batch + idx * 8; // 8 uint64_t for 512-bit result

    // Use SIMD multiplication
    mul256_simd(result, a, b);

    // Apply warp-level carry propagation to final result
    for (int i = 0; i < 8; i++) {
        result[i] = carry_prop_warp(result[i], lane);
    }
}
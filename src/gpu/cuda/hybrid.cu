// hybrid.cu - CUDA implementation of Barrett-Montgomery hybrid modular arithmetic
// Implements Rule #4: Montgomery for mul-heavy ops, Barrett for add/sub/reduce
// mod_mul: bigint_mul -> mont_redc -> mul by R_mod -> barrett_reduce

#include <cuda_runtime.h>
#include <stdint.h>

#define LIMBS 8
#define WIDE_LIMBS 16
#define MU_LIMBS 9

// secp256k1 prime (p) Montgomery constants
__constant__ uint32_t SECP_P[LIMBS] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// secp256k1 order (n) Montgomery constants
__constant__ uint32_t SECP_N[LIMBS] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// R = 2^256 mod p (Montgomery base)
__constant__ uint32_t R_P[LIMBS] = {
    0x000003D1u, 0x00000001u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
};

// R = 2^256 mod n
__constant__ uint32_t R_N[LIMBS] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Barrett mu = floor(2^512 / p) for p
__constant__ uint32_t MU_P[MU_LIMBS] = {
    0x9ED0D4F9u, 0xA9E34737u, 0x8F5E9C3Du, 0x7B2E0029u,
    0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u
};

// Barrett mu = floor(2^512 / n) for n
__constant__ uint32_t MU_N[MU_LIMBS] = {
    0xE89F2F9Eu, 0xED809F6Du, 0xCAA2B2BBu, 0xC2D2EAA9u,
    0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u
};

// Montgomery n' = -p^{-1} mod 2^32 for p
__constant__ uint32_t P_PRIME = 0xFFFFFFEDu;

// Montgomery n' = -n^{-1} mod 2^32 for n
__constant__ uint32_t N_PRIME = 0x747ED871u;

// Device helper: Parallel big integer comparison
__device__ int bigint_cmp_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    int local_cmp = 0;

    if (limb_idx < LIMBS) {
        if (a[limb_idx] > b[limb_idx]) local_cmp = 1;
        else if (a[limb_idx] < b[limb_idx]) local_cmp = -1;
    }

    // Find the most significant difference
    int msb_diff = 0;
    for (int i = LIMBS - 1; i >= 0; i--) {
        int cmp = __shfl_sync(0xFFFFFFFF, local_cmp, i);
        if (cmp != 0) {
            msb_diff = cmp;
            break;
        }
    }

    return msb_diff;
}

// Device helper: Parallel big integer subtraction with borrow
__device__ void bigint_sub_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t borrow_in = 0;

    // Get borrow from previous limb via warp shuffle
    if (limb_idx > 0) {
        borrow_in = __shfl_sync(0xFFFFFFFF, (result[limb_idx - 1] >> 31) & 1, limb_idx - 1);
    }

    uint64_t diff = (uint64_t)a[limb_idx] - b[limb_idx] - borrow_in;
    result[limb_idx] = diff & 0xFFFFFFFFULL;

    uint32_t borrow_out = (diff >> 63) & 1;

    // Propagate borrow to next limb
    if (limb_idx < LIMBS - 1) {
        __shfl_sync(0xFFFFFFFF, borrow_out, limb_idx + 1);
    }
    __syncthreads();
}

// Device helper: Parallel big integer multiplication (8x8 -> 16 limbs)
__device__ void bigint_mul_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[WIDE_LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;

    // Initialize result
    if (limb_idx < WIDE_LIMBS) result[limb_idx] = 0;
    __syncthreads();

    // Each limb of a multiplies with each limb of b
    for (int i = 0; i < LIMBS; i++) {
        if (limb_idx < LIMBS) {
            uint64_t prod = (uint64_t)a[i] * b[limb_idx];
            uint32_t low = prod & 0xFFFFFFFFULL;
            uint32_t high = (prod >> 32) & 0xFFFFFFFFULL;

            atomicAdd(&result[i + limb_idx], low);
            if (i + limb_idx + 1 < WIDE_LIMBS) {
                atomicAdd(&result[i + limb_idx + 1], high);
            }
        }
    }
    __syncthreads();

    // Carry propagation
    if (limb_idx < WIDE_LIMBS) {
        uint32_t carry = 0;
        uint64_t sum = result[limb_idx] + carry;
        result[limb_idx] = sum & 0xFFFFFFFFULL;
        carry = (sum >> 32) & 0xFFFFFFFFULL;

        if (limb_idx < WIDE_LIMBS - 1) {
            __shfl_sync(0xFFFFFFFF, carry, limb_idx + 1);
        }
    }
    __syncthreads();
}

// Device helper: Wide multiplication for Barrett (16x9 -> 25 limbs)
__device__ void bigint_wide_mul_par(const uint32_t a[WIDE_LIMBS], const uint32_t b[MU_LIMBS], uint32_t result[WIDE_LIMBS + MU_LIMBS]) {
    int limb_idx = threadIdx.x % WIDE_LIMBS;

    // Initialize result
    if (limb_idx < WIDE_LIMBS + MU_LIMBS) result[limb_idx] = 0;
    __syncthreads();

    // Each limb of a multiplies with each limb of b
    for (int i = 0; i < WIDE_LIMBS; i++) {
        if (limb_idx < MU_LIMBS) {
            uint64_t prod = (uint64_t)a[i] * b[limb_idx];
            uint32_t low = prod & 0xFFFFFFFFULL;
            uint32_t high = (prod >> 32) & 0xFFFFFFFFULL;

            atomicAdd(&result[i + limb_idx], low);
            atomicAdd(&result[i + limb_idx + 1], high);
        }
    }
    __syncthreads();

    // Carry propagation
    if (limb_idx < WIDE_LIMBS + MU_LIMBS) {
        uint32_t carry = 0;
        uint64_t sum = result[limb_idx] + carry;
        result[limb_idx] = sum & 0xFFFFFFFFULL;
        carry = (sum >> 32) & 0xFFFFFFFFULL;

        if (limb_idx < WIDE_LIMBS + MU_LIMBS - 1) {
            __shfl_sync(0xFFFFFFFF, carry, limb_idx + 1);
        }
    }
    __syncthreads();
}

// Device helper: Montgomery reduction (REDC algorithm)
__device__ void montgomery_redc_par(const uint32_t t[WIDE_LIMBS], const uint32_t mod_[LIMBS], uint32_t n_prime, uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t temp[WIDE_LIMBS];

    // Copy t to temp
    if (limb_idx < WIDE_LIMBS) temp[limb_idx] = t[limb_idx];
    __syncthreads();

    // REDC loop
    for (int i = 0; i < LIMBS; i++) {
        if (limb_idx == 0) {
            uint32_t m = ((uint64_t)temp[i] * n_prime) & 0xFFFFFFFFULL;
            uint32_t carry = 0;

            for (int j = 0; j < LIMBS; j++) {
                uint64_t prod = (uint64_t)m * mod_[j] + temp[i + j] + carry;
                temp[i + j] = prod & 0xFFFFFFFFULL;
                carry = (prod >> 32) & 0xFFFFFFFFULL;
            }

            // Propagate remaining carry
            for (int j = LIMBS; j < WIDE_LIMBS - i && carry > 0; j++) {
                uint64_t sum = (uint64_t)temp[i + j] + carry;
                temp[i + j] = sum & 0xFFFFFFFFULL;
                carry = (sum >> 32) & 0xFFFFFFFFULL;
            }
        }
        __syncthreads();
    }

    // Extract upper half
    if (limb_idx < LIMBS) result[limb_idx] = temp[LIMBS + limb_idx];

    // Conditional subtraction
    if (bigint_cmp_par(result, mod_) >= 0) {
        bigint_sub_par(result, mod_, result);
    }
}

// Device helper: Barrett reduction
__device__ void barrett_reduce_par(const uint32_t x[WIDE_LIMBS], const uint32_t mu[MU_LIMBS], const uint32_t mod_[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t x_mu[WIDE_LIMBS + MU_LIMBS];

    // Compute x * mu (wide multiplication)
    bigint_wide_mul_par(x, mu, x_mu);

    // Extract q from upper bits (approximation of x * mu >> 512)
    __shared__ uint32_t q[LIMBS];
    if (limb_idx < LIMBS) {
        q[limb_idx] = x_mu[WIDE_LIMBS + limb_idx];  // Upper 256 bits of product
    }
    __syncthreads();

    // Compute q * mod
    uint32_t qp[WIDE_LIMBS];
    bigint_mul_par(q, mod_, qp);

    // Compute r = x - q * mod (lower 256 bits)
    if (limb_idx < LIMBS) {
        uint64_t diff = (uint64_t)x[limb_idx] - qp[limb_idx];
        result[limb_idx] = diff & 0xFFFFFFFFULL;
        uint32_t borrow = (diff >> 63) & 1;

        // Propagate borrow
        if (limb_idx < LIMBS - 1) {
            __shfl_sync(0xFFFFFFFF, borrow, limb_idx + 1);
        }
    }
    __syncthreads();

    // Handle borrow propagation properly
    if (limb_idx < LIMBS) {
        uint32_t borrow = 0;
        if (limb_idx > 0) {
            borrow = __shfl_sync(0xFFFFFFFF, (result[limb_idx - 1] >> 31) & 1, limb_idx - 1);
        }
        uint64_t corrected = (uint64_t)result[limb_idx] - borrow;
        result[limb_idx] = corrected & 0xFFFFFFFFULL;
    }
    __syncthreads();

    // Conditional subtraction: if r >= mod, r -= mod
    if (bigint_cmp_par(result, mod_) >= 0) {
        bigint_sub_par(result, mod_, result);
    }
}

// Core hybrid modular multiplication: a * b mod mod
__device__ void mod_mul_hybrid(const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t mod_[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t t[WIDE_LIMBS];

    // Step 1: Standard multiplication -> 512-bit product
    bigint_mul_par(a, b, t);

    // Step 2: Montgomery reduction -> (a*b)/R mod mod
    uint32_t n_prime = (mod_[0] == SECP_P[0]) ? P_PRIME : N_PRIME;
    uint32_t redc_result[LIMBS];
    montgomery_redc_par(t, mod_, n_prime, redc_result);

    // Step 3: Convert back from Montgomery form: redc_result * R mod mod
    uint32_t r_mod[LIMBS];
    if (mod_[0] == SECP_P[0]) {
        for (int i = 0; i < LIMBS; i++) r_mod[i] = R_P[i];
    } else {
        for (int i = 0; i < LIMBS; i++) r_mod[i] = R_N[i];
    }

    uint32_t adjust[WIDE_LIMBS];
    bigint_mul_par(redc_result, r_mod, adjust);

    // Step 4: Barrett reduction for final result
    uint32_t mu[MU_LIMBS];
    if (mod_[0] == SECP_P[0]) {
        for (int i = 0; i < MU_LIMBS; i++) mu[i] = MU_P[i];
    } else {
        for (int i = 0; i < MU_LIMBS; i++) mu[i] = MU_N[i];
    }

    barrett_reduce_par(adjust, mu, mod_, result);
}

// Batch hybrid modular multiplication kernel
__global__ void batch_mod_mul_hybrid(uint32_t *a, uint32_t *b, uint32_t *mod, uint32_t *result, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    // Each thread handles one modular multiplication
    mod_mul_hybrid(&a[id * LIMBS], &b[id * LIMBS], mod, &result[id * LIMBS]);
}

// Batch hybrid modular squaring kernel (optimization for a*a)
__global__ void batch_mod_sqr_hybrid(uint32_t *a, uint32_t *mod, uint32_t *result, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    // Squaring using the same hybrid multiplication
    mod_mul_hybrid(&a[id * LIMBS], &a[id * LIMBS], mod, &result[id * LIMBS]);
}

// Host function for batch hybrid multiplication
extern "C" cudaError_t batch_mod_mul_hybrid_cuda(
    uint32_t *d_a, uint32_t *d_b, uint32_t *d_mod, uint32_t *d_result,
    int batch, cudaStream_t stream
) {
    dim3 grid((batch + 255) / 256);
    dim3 block(256);  // 256 threads per block, 32 limbs per warp

    batch_mod_mul_hybrid<<<grid, block, 0, stream>>>(
        d_a, d_b, d_mod, d_result, batch
    );

    return cudaGetLastError();
}

// Host function for batch hybrid squaring
extern "C" cudaError_t batch_mod_sqr_hybrid_cuda(
    uint32_t *d_a, uint32_t *d_mod, uint32_t *d_result,
    int batch, cudaStream_t stream
) {
    dim3 grid((batch + 255) / 256);
    dim3 block(256);

    batch_mod_sqr_hybrid<<<grid, block, 0, stream>>>(
        d_a, d_mod, d_result, batch
    );

    return cudaGetLastError();
}
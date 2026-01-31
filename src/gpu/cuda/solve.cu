// solve.cu - CUDA kernels for batch collision equation solving and Barrett reduction
// Implements priv = alpha_t - alpha_w + (beta_t - beta_w) * target mod n
// Barrett reduction: q = (x * mu)>>512, r = x - q*mod mod mod
// Hybrid with Montgomery for mul-heavy operations

#include <cuda_runtime.h>
#include <stdint.h>

#define LIMBS 8
#define WIDE_LIMBS 16

// secp256k1 order (n) and secp256k1 prime (p) constants
__constant__ uint32_t SECP_N[LIMBS] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

__constant__ uint32_t SECP_P[LIMBS] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Device helper: Parallel big integer subtraction with borrow propagation
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

// Device helper: Parallel big integer addition with carry propagation
__device__ void bigint_add_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t carry_in = 0;

    // Get carry from previous limb via warp shuffle
    if (limb_idx > 0) {
        carry_in = __shfl_sync(0xFFFFFFFF, result[limb_idx - 1] >> 31, limb_idx - 1);
    }

    uint64_t sum = (uint64_t)a[limb_idx] + b[limb_idx] + carry_in;
    result[limb_idx] = sum & 0xFFFFFFFFULL;

    uint32_t carry_out = (sum >> 32) & 0xFFFFFFFFULL;

    // Propagate carry to next limb
    if (limb_idx < LIMBS - 1) {
        __shfl_sync(0xFFFFFFFF, carry_out, limb_idx + 1);
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

            // Atomic add to result limbs
            atomicAdd(&result[i + limb_idx], low);
            if (i + limb_idx + 1 < WIDE_LIMBS) {
                atomicAdd(&result[i + limb_idx + 1], high);
            }
        }
    }
    __syncthreads();

    // Carry propagation for the result
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

// Device helper: Montgomery reduction (simplified)
__device__ void montgomery_redc_par(const uint32_t t[WIDE_LIMBS], const uint32_t mod_[LIMBS], uint32_t n_prime, uint32_t result[LIMBS]) {
    // Simplified REDC implementation - would need full CIOS/FIOS algorithm for production
    uint32_t m, carry;
    uint32_t temp[WIDE_LIMBS];

    // Copy t to temp
    for (int i = 0; i < WIDE_LIMBS; i++) temp[i] = t[i];

    // REDC loop
    for (int i = 0; i < LIMBS; i++) {
        m = ((uint64_t)temp[i] * n_prime) & 0xFFFFFFFFULL;
        carry = 0;

        for (int j = 0; j < LIMBS; j++) {
            uint64_t prod = (uint64_t)m * mod_[j] + temp[i + j] + carry;
            temp[i + j] = prod & 0xFFFFFFFFULL;
            carry = (prod >> 32) & 0xFFFFFFFFULL;
        }

        // Propagate remaining carry
        for (int j = LIMBS; j < WIDE_LIMBS - i; j++) {
            uint64_t sum = temp[i + j] + carry;
            temp[i + j] = sum & 0xFFFFFFFFULL;
            carry = (sum >> 32) & 0xFFFFFFFFULL;
        }
    }

    // Extract upper half
    for (int i = 0; i < LIMBS; i++) result[i] = temp[i + LIMBS];

    // Conditional subtraction
    if (bigint_cmp_par(result, mod_) >= 0) {
        bigint_sub_par(result, mod_, result);
    }
}

// Device helper: Wide multiplication for Barrett (16x9 -> 25 limbs, but we take upper)
__device__ void bigint_wide_mul_par(const uint32_t a[WIDE_LIMBS], const uint32_t b[LIMBS+1], uint32_t result[WIDE_LIMBS + LIMBS + 1]) {
    int limb_idx = threadIdx.x % WIDE_LIMBS;

    // Initialize result
    if (limb_idx < WIDE_LIMBS + LIMBS + 1) result[limb_idx] = 0;
    __syncthreads();

    // Each limb of a multiplies with each limb of b
    for (int i = 0; i < WIDE_LIMBS; i++) {
        if (limb_idx < LIMBS + 1) {
            uint64_t prod = (uint64_t)a[i] * b[limb_idx];
            uint32_t low = prod & 0xFFFFFFFFULL;
            uint32_t high = (prod >> 32) & 0xFFFFFFFFULL;

            atomicAdd(&result[i + limb_idx], low);
            atomicAdd(&result[i + limb_idx + 1], high);
        }
    }
    __syncthreads();

    // Carry propagation
    if (limb_idx < WIDE_LIMBS + LIMBS + 1) {
        uint32_t carry = 0;
        uint64_t sum = result[limb_idx] + carry;
        result[limb_idx] = sum & 0xFFFFFFFFULL;
        carry = (sum >> 32) & 0xFFFFFFFFULL;

        if (limb_idx < WIDE_LIMBS + LIMBS) {
            __shfl_sync(0xFFFFFFFF, carry, limb_idx + 1);
        }
    }
    __syncthreads();
}

// Batch collision equation solving kernel
// priv = alpha_t - alpha_w + (beta_t - beta_w) * target mod n
__global__ void batch_collision_solve(uint32_t *alpha_t, uint32_t *alpha_w, uint32_t *beta_t, uint32_t *beta_w, uint32_t *target, uint32_t *n, uint32_t *priv_out, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    // Shared memory for intermediate results (one set per block)
    __shared__ uint32_t alpha_diff[LIMBS];
    __shared__ uint32_t beta_diff[LIMBS];
    __shared__ uint32_t mul_part[WIDE_LIMBS];
    __shared__ uint32_t priv_temp[WIDE_LIMBS];

    int limb_idx = threadIdx.x % LIMBS;

    // Load inputs for this batch item
    uint32_t alpha_t_limb = alpha_t[id * LIMBS + limb_idx];
    uint32_t alpha_w_limb = alpha_w[id * LIMBS + limb_idx];
    uint32_t beta_t_limb = beta_t[id * LIMBS + limb_idx];
    uint32_t beta_w_limb = beta_w[id * LIMBS + limb_idx];
    uint32_t target_limb = target[id * LIMBS + limb_idx];  // Assuming target per batch item
    uint32_t n_limb = n[limb_idx];

    // Step 1: Compute alpha_diff = alpha_t - alpha_w
    bigint_sub_par(&alpha_t[id * LIMBS], &alpha_w[id * LIMBS], alpha_diff);

    // Step 2: Compute beta_diff = beta_t - beta_w
    bigint_sub_par(&beta_t[id * LIMBS], &beta_w[id * LIMBS], beta_diff);

    // Step 3: Compute mul_part = beta_diff * target
    bigint_mul_par(beta_diff, &target[id * LIMBS], mul_part);

    // Step 4: Compute priv_temp = alpha_diff + mul_part
    bigint_add_par(alpha_diff, mul_part, priv_temp);

    // Step 5: Reduce modulo n
    // First, check if priv_temp >= n and subtract if needed
    if (bigint_cmp_par(priv_temp, n) >= 0) {
        bigint_sub_par(priv_temp, n, priv_temp);
    }

    // Handle negative results (if alpha_diff was negative)
    // Check if priv_temp is negative (most significant bit set)
    if ((priv_temp[LIMBS-1] & 0x80000000) != 0) {
        bigint_add_par(priv_temp, n, priv_temp);
    }

    // Store result
    priv_out[id * LIMBS + limb_idx] = priv_temp[limb_idx];
}

// Batch Barrett reduction kernel with hybrid Montgomery option
__global__ void batch_barrett_reduce(uint32_t *x, uint32_t *mu, uint32_t *mod, uint32_t *out, bool use_mont, uint32_t n_prime, int batch, int limbs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    int limb_idx = threadIdx.x % limbs;

    if (use_mont) {
        // Use Montgomery reduction for mul-heavy operations
        montgomery_redc_par(&x[id * WIDE_LIMBS], mod, n_prime, &out[id * limbs]);
    } else {
        // Barrett reduction: q = (x * mu) >> (2*k), r = x - q * mod
        __shared__ uint32_t x_wide[WIDE_LIMBS];
        __shared__ uint32_t q[LIMBS];
        __shared__ uint32_t qp[WIDE_LIMBS];
        __shared__ uint32_t result[LIMBS];

        // Load x (512-bit input)
        if (limb_idx < WIDE_LIMBS) {
            x_wide[limb_idx] = x[id * WIDE_LIMBS + limb_idx];
        }
        __syncthreads();

        // Compute x * mu (wide multiplication)
        uint32_t x_mu[WIDE_LIMBS + LIMBS + 1];
        bigint_wide_mul_par(x_wide, mu, x_mu);

        // Extract q from upper bits (approximation of x * mu >> 512)
        // For 512-bit Barrett, we need bits 512 to 512+k-1
        if (limb_idx < limbs) {
            q[limb_idx] = x_mu[WIDE_LIMBS + limb_idx];  // Upper 256 bits of product
        }
        __syncthreads();

        // Compute q * mod
        bigint_mul_par(q, mod, qp);

        // Compute r = x - q * mod (lower 256 bits)
        if (limb_idx < limbs) {
            uint64_t diff = (uint64_t)x_wide[limb_idx] - qp[limb_idx];
            result[limb_idx] = diff & 0xFFFFFFFFULL;
            uint32_t borrow = (diff >> 63) & 1;

            // Propagate borrow
            if (limb_idx < limbs - 1) {
                __shfl_sync(0xFFFFFFFF, borrow, limb_idx + 1);
            }
        }
        __syncthreads();

        // Handle borrow propagation properly
        if (limb_idx < limbs) {
            uint32_t borrow = 0;
            if (limb_idx > 0) {
                borrow = __shfl_sync(0xFFFFFFFF, (result[limb_idx - 1] >> 31) & 1, limb_idx - 1);
            }
            uint64_t corrected = (uint64_t)result[limb_idx] - borrow;
            result[limb_idx] = corrected & 0xFFFFFFFFULL;
        }
        __syncthreads();

        // Conditional subtraction: if r >= mod, r -= mod
        if (bigint_cmp_par(result, mod) >= 0) {
            bigint_sub_par(result, mod, result);
        }

        // Store final result
        out[id * limbs + limb_idx] = result[limb_idx];
    }
}

// Fused batch collision solve with integrated Montgomery reduction
__global__ void batch_collision_solve_fused(uint32_t *alpha_t, uint32_t *alpha_w, uint32_t *beta_t, uint32_t *beta_w, uint32_t *target, uint32_t *n, uint32_t n_prime, uint32_t *priv_out, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    __shared__ uint32_t alpha_diff[LIMBS];
    __shared__ uint32_t beta_diff[LIMBS];
    __shared__ uint32_t mul_part[WIDE_LIMBS];
    __shared__ uint32_t priv_temp[WIDE_LIMBS];

    int limb_idx = threadIdx.x % LIMBS;

    // Load inputs
    uint32_t alpha_t_limb = alpha_t[id * LIMBS + limb_idx];
    uint32_t alpha_w_limb = alpha_w[id * LIMBS + limb_idx];
    uint32_t beta_t_limb = beta_t[id * LIMBS + limb_idx];
    uint32_t beta_w_limb = beta_w[id * LIMBS + limb_idx];
    uint32_t target_limb = target[id * LIMBS + limb_idx];
    uint32_t n_limb = n[limb_idx];

    // Compute alpha_diff = alpha_t - alpha_w
    bigint_sub_par(&alpha_t[id * LIMBS], &alpha_w[id * LIMBS], alpha_diff);

    // Compute beta_diff = beta_t - beta_w
    bigint_sub_par(&beta_t[id * LIMBS], &beta_w[id * LIMBS], beta_diff);

    // Compute mul_part = beta_diff * target
    bigint_mul_par(beta_diff, &target[id * LIMBS], mul_part);

    // Compute priv_temp = alpha_diff + mul_part
    bigint_add_par(alpha_diff, mul_part, priv_temp);

    // Fused Montgomery reduction instead of separate modulo
    montgomery_redc_par(priv_temp, n, n_prime, &priv_out[id * LIMBS]);
}

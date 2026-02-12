// solve.cu - CUDA kernels for batch collision solving and BSGS
#include <cuda_runtime.h>
#include <stdint.h>
#include "common_constants.h"
#include <stdio.h> // For printf

#define KANGS_PER_TARGET 4096
#define DEBUG 1
#define LIMBS 8
#define WIDE_LIMBS 16

// Point and KangarooState structures are now defined in common_constants.h

// Constants are defined in step.cu - extern declarations here

// Forward declarations for functions from step.cu (only those not implemented locally)
extern __device__ void bigint_sub(const uint32_t* a, const uint32_t* b, uint32_t* res);
extern __device__ void point_neg(const Point* p, Point* neg, const uint32_t* mod);
extern __device__ void bigint_sub_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]);
extern __device__ void bigint_mul_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[WIDE_LIMBS]);
extern __device__ void bigint_add_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]);
extern __device__ Point jacobian_add(Point p1, Point p2);
extern __device__ Point jacobian_double(Point p);
extern __device__ Point ec_mul_small(Point p, uint32_t scalar);
extern __device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* res, const uint32_t* mod);
extern __device__ void glv_decompose_scalar(const uint32_t k[8], uint32_t k1[8], uint32_t k2[8], int8_t* sign1, int8_t* sign2);
extern __device__ Point mul_glv_opt(Point p, const uint32_t k[8]);
extern __device__ int point_equal(Point p1, Point p2);

// Constants from step.cu
extern __constant__ uint64_t PRIME_MULTIPLIERS[32];
extern __device__ Point jacobian_add(Point p1, Point p2);
extern __device__ Point jacobian_double(Point p);
extern __device__ Point ec_mul_small(Point p, uint32_t scalar);
extern __device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* res, const uint32_t* mod);
extern __device__ void glv_decompose_scalar(const uint32_t k[8], uint32_t k1[8], uint32_t k2[8], int8_t* sign1, int8_t* sign2);
extern __device__ Point mul_glv_opt(Point p, const uint32_t k[8]);
extern __device__ int point_equal(Point p1, Point p2);

#define LIMBS 8
#define WIDE_LIMBS 16

// Use Point struct from step.cu

// secp256k1 order (n) as uint32_t[8]
__constant__ uint32_t SECP_N[LIMBS] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Alias for CURVE_ORDER
#define CURVE_ORDER SECP_N

// secp256k1 prime (p) as uint32_t[8]
__constant__ uint32_t SECP_P[LIMBS] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Precomputed Barrett mu for SECP_P (floor(2^{512} / p)) as uint32_t[9]
__constant__ uint32_t MU_P[9] = {
    0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u,
    0x00000000u  // Extra limb for mu
};

// secp256k1 generator G x/y coords as uint32_t[8] (affine, z=1)
__constant__ uint32_t GENERATOR_X[LIMBS] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};

__constant__ uint32_t GENERATOR_Y[LIMBS] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0DA388F4u, 0xFE5D4984u, 0x35203864u, 0x483ADA77u
};

// GLV constants are now defined in step.cu


// BSGS table entry for baby steps
struct BSGS_Entry {
    uint32_t point_x[LIMBS];  // x-coordinate of g^i
    uint32_t index;          // i
};

// BSGS table for giant steps (stored in global memory)
struct BSGS_Table {
    BSGS_Entry* baby_steps;
    int m;  // sqrt(order) size
};

// DP Entry structure for collision detection
struct DpEntry {
    Point point;
    uint32_t distance[LIMBS];
    uint32_t alpha[LIMBS];
    uint32_t beta[LIMBS];
    uint32_t cluster_id;
    uint64_t timestamp;
};

// Solved flags array (per target)
struct SolvedFlag {
    uint32_t solved; // 1 if solved
    uint32_t priv_key[LIMBS]; // Recovered key
};

// Utility functions for limb operations
static __device__ int limb_compare(const uint32_t* a, const uint32_t* b, int limbs) {
    for (int i = limbs - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

static __device__ void limb_add(const uint32_t* a, const uint32_t* b, uint32_t* res, int limbs) {
    uint32_t carry = 0;
    for (int i = 0; i < limbs; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        res[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
}

static __device__ void limb_sub(const uint32_t* a, const uint32_t* b, uint32_t* res, int limbs) {
    uint32_t borrow = 0;
    for (int i = 0; i < limbs; i++) {
        uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
        res[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
}

// Barrett modular reduction
static __device__ void barrett_mod(const uint32_t* x, const uint32_t* modulus, uint32_t* result, int limbs) {
    // Simplified Barrett reduction for now
    for (int i = 0; i < limbs; i++) {
        result[i] = x[i] % modulus[i];
    }
}

// Compare a ? b
static __device__ int bigint_cmp(const uint32_t* a, const uint32_t* b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// Parallel comparison for LIMBS arrays
static __device__ int bigint_cmp_par(const uint32_t* a, const uint32_t* b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}


// Add res = a + b
static __device__ void bigint_add(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    uint32_t carry = 0;
    for (int i = 0; i < LIMBS; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        res[i] = sum & 0xFFFFFFFF;
        carry = (sum >> 32) != 0 ? 1 : 0;
    }
    // Carry is ignored - function modifies res in place
}

// Mul res = a * b (wide)
static __device__ void bigint_mul(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    for (int i = 0; i < WIDE_LIMBS; i++) res[i] = 0;
    for (int i = 0; i < LIMBS; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j] + res[i+j] + carry;
            res[i+j] = prod & 0xFFFFFFFF;
            carry = (prod >> 32) != 0 ? 1 : 0;
        }
        res[i+LIMBS] += carry;
    }
}

// Full Barrett reduction for wide input
static __device__ void barrett_reduce_full(const uint32_t* x, const uint32_t* modulus, const uint32_t* mu, uint32_t* result) {
    // x is WIDE_LIMBS, modulus LIMBS, mu 9 limbs
    uint32_t q[WIDE_LIMBS] = {0};
    // Approximate q = (x >> (LIMBS*32 - 1)) * mu >> (LIMBS*32 + 1)
    // Shift x right by k-1 = 255 bits (approx with loops)
    for (int i = 0; i < WIDE_LIMBS; i++) q[i] = x[i] >> 1; // Simplified shift
    mul_mod(q, mu, q, modulus); // q * mu mod (approx)
    // r = x - q * modulus
    uint32_t q_mod[WIDE_LIMBS] = {0};
    mul_mod(q, modulus, q_mod, modulus); // Placeholder mod
    limb_sub(x, q_mod, result, LIMBS);
    // Final subtract if r >= modulus
    if (limb_compare(result, modulus, LIMBS) >= 0) limb_sub(result, modulus, result, LIMBS);
}

// Simple div for gcd (a / b = q)
static __device__ void bigint_div(const uint32_t* a, const uint32_t* b, uint32_t* q) {
    for (int i = 0; i < LIMBS; i++) q[i] = 0;
    uint32_t dividend[WIDE_LIMBS];
    for (int i = 0; i < LIMBS; i++) dividend[i] = a[i];
    for (int i = LIMBS; i < WIDE_LIMBS; i++) dividend[i] = 0;
    for (int bit = 256 - 1; bit >= 0; bit--) {
        uint32_t carry = 0;
        for (int i = 0; i < WIDE_LIMBS; i++) {
            uint32_t next_carry = dividend[i] >> 31;
            dividend[i] = (dividend[i] << 1) | carry;
            carry = next_carry;
        }
        if (bigint_cmp(dividend, b) >= 0) {
            bigint_sub(dividend, b, dividend);
            int word = bit / 32;
            int bbit = bit % 32;
            q[word] |= (1u << bbit);
        }
    }
}

// Mod res = a mod m (using Barrett)
static __device__ void bigint_mod(const uint32_t* a, const uint32_t* m, uint32_t* res) {
    uint32_t wide_a[WIDE_LIMBS];
    for (int i = 0; i < LIMBS; i++) wide_a[i] = a[i];
    for (int i = LIMBS; i < WIDE_LIMBS; i++) wide_a[i] = 0;
    barrett_reduce_full(wide_a, m, MU_P, res);
}

// Extended Euclidean mod inverse
static __device__ void mod_inverse(const uint32_t* a, const uint32_t* mod, uint32_t* res) {
    uint32_t t[LIMBS] = {0}, r[LIMBS];
    uint32_t nt[LIMBS] = {1}, nr[LIMBS] = {0};
    uint32_t a_copy[LIMBS];
    for (int i = 0; i < LIMBS; i++) {
        r[i] = mod[i];
        a_copy[i] = a[i];
    }
    uint32_t q[LIMBS], temp[LIMBS];
    int iterations = 0;
    uint32_t zero_arr[LIMBS] = {0};
    while (bigint_cmp(a_copy, zero_arr) != 0 && iterations < 1024) { // Limit iterations
        bigint_div(r, a_copy, q);
        bigint_mul(q, a_copy, temp);
        bigint_sub(r, temp, r);
        bigint_mul(q, nr, temp);
        bigint_sub(nt, temp, nt);
        // Swap
        for (int i = 0; i < LIMBS; i++) {
            uint32_t tmp = nt[i]; nt[i] = nr[i]; nr[i] = tmp;
            tmp = r[i]; r[i] = a_copy[i]; a_copy[i] = tmp;
        }
        iterations++;
    }
    if (bigint_cmp(t, zero_arr) < 0) bigint_add(t, mod, t);
    for (int i = 0; i < LIMBS; i++) res[i] = t[i];
}

// Device function for safe diff mod N
static __device__ void cuda_safe_diff_mod_n(const uint32_t tame[LIMBS], const uint32_t wild[LIMBS], const uint32_t n[LIMBS], uint32_t result[LIMBS]) {
    uint32_t diff[LIMBS], temp[LIMBS];
    int cmp = limb_compare(tame, wild, LIMBS);
    if (cmp >= 0) {
        limb_sub(tame, wild, diff, LIMBS);
    } else {
        limb_add(tame, n, temp, LIMBS);
        limb_sub(temp, wild, diff, LIMBS);
    }
    barrett_reduce_full(diff, n, MU_P, result);
}



// Device function: Point subtraction (p1 - p2 = p1 + (-p2))
static __device__ void point_sub(const Point* p1, const Point* p2, Point* result, const uint32_t* mod) {
    Point neg_p2;
    point_neg(p2, &neg_p2, mod);
    *result = jacobian_add(*p1, neg_p2);
}


// Scalar mul small (binary method)
static __device__ Point mul_small(const Point* p, uint32_t scalar, const uint32_t* mod) {
    Point result;
    for (int i = 0; i < LIMBS; i++) {
        result.x[i] = 0;
        result.y[i] = 0;
        result.z[i] = (i == 0) ? 1 : 0; // Identity
    }
    Point current = *p;
    while (scalar > 0) {
        if (scalar & 1) result = jacobian_add(result, current);
        current = jacobian_double(current); // From step.cu
        scalar >>= 1;
    }
    return result;
}

// GLV-optimized scalar mul


// Alias for compatibility
static __device__ void point_add_jacobian(const Point* p1, const Point* p2, Point* result, const uint32_t* mod) {
    *result = jacobian_add(*p1, *p2);
}

// Device function: Scalar multiplication by small integer
static __device__ Point point_mul_small(const Point* p, uint32_t scalar, const uint32_t* mod) {
    Point result;
    // Initialize result to infinity
    for (int i = 0; i < LIMBS; i++) {
        result.x[i] = 0;
        result.y[i] = 0;
        result.z[i] = 0;
    }

    Point current = *p;
    while (scalar > 0) {
        if (scalar & 1) {
            result = jacobian_add(result, current);
        }
        current = jacobian_add(current, current); // Double
        scalar >>= 1;
    }
    return result;
}







// Device function: Modular inverse via extended Euclid (simplified for u64 approximation)
static __device__ uint64_t mod_inverse_u64(uint64_t a, uint64_t mod) {
    int64_t m = mod, m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;

    if (m == 1) return 0;

    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    if (x1 < 0) x1 += m0;
    return x1;
}

// Device function: Factor out small primes (simplified u64)
static __device__ uint64_t factor_small_primes_u64(uint64_t val) {
    for (int i = 0; i < 32; i++) {
        uint64_t p = PRIME_MULTIPLIERS[i];
        while (val % p == 0 && val >= p) {
            val /= p;
        }
    }
    return val;
}


// Device function: Build baby steps table for BSGS
__global__ void build_baby_steps(Point* baby_table, int m, const Point* generator, const uint32_t* mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;

    Point result;
    result = point_mul_small(generator, idx, mod);

    for (int i = 0; i < LIMBS; i++) {
        baby_table[idx].x[i] = result.x[i];
        baby_table[idx].y[i] = result.y[i];
        baby_table[idx].z[i] = result.z[i];
    }
}

// Device function: BSGS solve kernel - find x such that g^x = target
__global__ void bsgs_solve_kernel(
    const Point* target,
    const Point* generator,
    const Point* baby_table,
    uint32_t* results,
    int m,
    int batch_size,
    const uint32_t* mod
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // For each batch item, we would have different targets
    // For now, assume single target per kernel launch
    Point current_target = *target;

    // Build giant step table in shared memory or use global lookup
    // Simplified: just check baby steps first
    for (int i = 0; i < m; i++) {
        if (point_equal(baby_table[i], current_target)) {
            results[idx] = i;
            return;
        }
    }

    // If not found in baby steps, would need giant step search
    // For now, mark as not found
    results[idx] = 0xFFFFFFFF;
}



// Device helper: Montgomery reduction (simplified)
static __device__ void montgomery_redc_par(const uint32_t t[WIDE_LIMBS], const uint32_t mod_[LIMBS], uint32_t n_prime, uint32_t result[LIMBS]) {
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
static __device__ void bigint_wide_mul_par(const uint32_t a[WIDE_LIMBS], const uint32_t b[LIMBS+1], uint32_t result[WIDE_LIMBS + LIMBS + 1]) {
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

// Batch BSGS collision solving for near-collisions
__global__ void batch_bsgs_collision_solve(
    const Point* deltas,        // Delta points (target - current)
    const uint32_t* alphas,     // Alpha coefficients for prime inverse
    const uint32_t* distances,  // Distance arrays for each trap
    uint32_t* solutions,        // Output solutions
    int batch_size,             // Number of collisions to solve
    uint64_t bsgs_threshold,    // Max difference for BSGS
    const uint32_t* mod,        // Modulus (secp256k1 order)
    int gold_bias_combo         // Enable GOLD factoring
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const Point* delta = &deltas[idx];
    const uint32_t* alpha = &alphas[idx * LIMBS];
    const uint32_t* dist = &distances[idx * LIMBS];

    // Diff u64 approx (sum low limbs)
    uint64_t diff = 0;
    for (int i = 0; i < LIMBS; i++) {
        diff += ((uint64_t)dist[i]) << (i * 32);
    }

    // GOLD factoring: reduce diff by factoring small primes
    if (gold_bias_combo) {
        uint64_t reduced = factor_small_primes_u64(diff);
        if (reduced < diff) {
            // Approximate factor count for threshold reduction
            int factor_count = 0;
            uint64_t temp = diff;
            while (temp > reduced) {
                temp = factor_small_primes_u64(temp);
                factor_count++;
            }
            if (factor_count > 1) {
                bsgs_threshold /= factor_count;  // Reduce threshold
            }
        }
    }

    if (diff < bsgs_threshold) {
        // Try prime inverse first if alpha is non-zero
        uint32_t zero[LIMBS] = {0};
        if (bigint_cmp_par(alpha, zero) != 0) {
            // Use simplified u64 inverse for low limb approximation
            uint64_t alpha_low = (uint64_t)alpha[0] | ((uint64_t)alpha[1] << 32);
            uint64_t mod_low = (uint64_t)mod[0] | ((uint64_t)mod[1] << 32);

            if (alpha_low > 0) {
                uint64_t inv = mod_inverse_u64(alpha_low, mod_low);
                if (inv != 0) {
                    // k = inv * (diff + 1) mod order
                    uint64_t k = (inv * (diff + 1)) % mod_low;

                    // Store result (low 64 bits)
                    solutions[idx * LIMBS] = k & 0xFFFFFFFFULL;
                    solutions[idx * LIMBS + 1] = (k >> 32) & 0xFFFFFFFFULL;
                    for (int i = 2; i < LIMBS; i++) {
                        solutions[idx * LIMBS + i] = 0;
                    }
                    return;
                }
            }
        }

        // Fallback: BSGS algorithm
        uint64_t m = (uint64_t)sqrt((double)bsgs_threshold) + 1;
        // Cap m to fit in shared memory (512 baby steps max)
        if (m > 512) m = 512;

        // Use shared memory for baby steps (limit to fit in shared memory)
        __shared__ uint32_t baby_x_shared[512 * LIMBS];  // x coordinates only
        __shared__ uint32_t baby_y_shared[512 * LIMBS];  // y coordinates only

        // Build baby steps: g^0, g^1, ..., g^{m-1}
        if (threadIdx.x < m) {
            // Generator point (secp256k1 base point)
            Point generator = {
                {0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
                 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798}, // x
                {0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
                 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8}, // y
                {1, 0, 0, 0, 0, 0, 0, 0}  // z
            };

            Point baby;
            baby = point_mul_small(&generator, threadIdx.x, mod);

            // Store in shared memory (x and y only for equality check)
            for (int i = 0; i < LIMBS; i++) {
                baby_x_shared[threadIdx.x * LIMBS + i] = baby.x[i];
                baby_y_shared[threadIdx.x * LIMBS + i] = baby.y[i];
            }
        }
        __syncthreads();

        // Giant step search
        for (uint64_t j = 0; j < m; j++) {
            // Compute giant step: g^{-(j*m)} * delta
            Point generator = {
                {0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
                 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798},
                {0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
                 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8},
                {1, 0, 0, 0, 0, 0, 0, 0}
            };

            uint64_t giant_scalar = j * m;
            Point giant_step;
            giant_step = point_mul_small(&generator, giant_scalar, mod);

            // target = delta - giant_step
            Point target_check;
            point_sub(delta, &giant_step, &target_check, mod);

            // Check if target matches any baby step
            for (uint64_t b = 0; b < m; b++) {
                bool match = true;
                for (int i = 0; i < LIMBS; i++) {
                    if (target_check.x[i] != baby_x_shared[b * LIMBS + i] ||
                        target_check.y[i] != baby_y_shared[b * LIMBS + i]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    // Found solution: x = j*m + b
                    uint64_t solution = j * m + b;

                    // Store solution (split into uint32_t array)
                    solutions[idx * LIMBS] = solution & 0xFFFFFFFFULL;
                    solutions[idx * LIMBS + 1] = (solution >> 32) & 0xFFFFFFFFULL;
                    for (int k = 2; k < LIMBS; k++) {
                        solutions[idx * LIMBS + k] = 0;
                    }
                    return;
                }
            }
        }

        // No solution found
        for (int i = 0; i < LIMBS; i++) {
            solutions[idx * LIMBS + i] = 0xFFFFFFFF;
        }
    } else {
        // Difference too large for BSGS
        for (int i = 0; i < LIMBS; i++) {
            solutions[idx * LIMBS + i] = 0xFFFFFFFF;
        }
    }
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

// Host function for launching BSGS collision solving
extern "C" void launch_batch_bsgs_collision_solve(
    Point* d_deltas,
    uint32_t* d_alphas,
    uint32_t* d_distances,
    uint32_t* d_solutions,
    int batch_size,
    uint64_t bsgs_threshold,
    cudaStream_t stream,
    int gold_bias_combo
) {
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256);

    // Get secp256k1 order
    uint32_t h_mod[LIMBS] = {
        0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    batch_bsgs_collision_solve<<<grid, block, 0, stream>>>(
        d_deltas, d_alphas, d_distances, d_solutions,
        batch_size, bsgs_threshold, d_mod, gold_bias_combo
    );

    cudaFree(d_mod);
}

// Host function for building baby steps table
extern "C" void launch_build_baby_steps(
    Point* d_baby_table,
    int m,
    Point* d_generator,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((m + 255) / 256);

    // Get secp256k1 modulus
    uint32_t h_mod[LIMBS] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    build_baby_steps<<<grid, block, 0, stream>>>(
        d_baby_table, m, d_generator, d_mod
    );

    cudaFree(d_mod);
}

// Collision check kernel with safe diff mod N
__global__ void check_collisions(uint32_t* tame_dists, uint32_t* wild_dists, uint32_t* results, int batch_size, uint32_t* n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    cuda_safe_diff_mod_n(&tame_dists[idx * 8], &wild_dists[idx * 8], n, &results[idx * 8]);
}

// Host function for launching BSGS solve
extern "C" void launch_bsgs_solve(
    Point* d_target,
    Point* d_generator,
    Point* d_baby_table,
    uint32_t* d_results,
    int m,
    int batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256);

    // Get secp256k1 modulus
    uint32_t h_mod[LIMBS] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bsgs_solve_kernel<<<grid, block, 0, stream>>>(
        d_target, d_generator, d_baby_table, d_results, m, batch_size, d_mod
    );

    cudaFree(d_mod);
}

// Phase 4/7/8 integrated collision solve kernel
__global__ void solve_collisions(DpEntry* tames, DpEntry* wilds, Point* targets, int num_collisions, int num_targets, int* solved_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_collisions) return;
    DpEntry tame = tames[idx];
    DpEntry wild = wilds[idx];
    // Phase 4: Safe diffs
    uint32_t diff_alpha[LIMBS] = {0};
    cuda_safe_diff_mod_n(tame.alpha, wild.alpha, CURVE_ORDER, diff_alpha);
    uint32_t diff_beta[LIMBS] = {0};
    cuda_safe_diff_mod_n(wild.beta, tame.beta, CURVE_ORDER, diff_beta);
    // Phase 7: Inv
    uint32_t inv_beta[LIMBS] = {0};
    mod_inverse(diff_beta, CURVE_ORDER, inv_beta);
    uint32_t k[WIDE_LIMBS] = {0};
    bigint_mul(diff_alpha, inv_beta, k);
    bigint_mod(k, CURVE_ORDER, (uint32_t*)k);
    // Phase 8: Test multi targets
    for (int t = 0; t < num_targets; ++t) {
        Point computed = mul_glv_opt(targets[t], (uint32_t*)k); // Phase 6 - use target as generator for verification
        if (point_equal(computed, targets[t])) {
            atomicExch(&solved_flags[t], 1); // Flag solved
        }
    }
}

// Batch scalar mul kernel (unified Point/uint32_t)
__global__ void batch_scalar_mul(Point *results, uint32_t *scalars, int num_points, const uint32_t* mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    uint32_t scalar_limbs[LIMBS] = {0};
    for (int i = 0; i < LIMBS; i++) scalar_limbs[i] = scalars[idx * LIMBS + i]; // Assume scalars as arrays
    Point g;
    for (int i = 0; i < LIMBS; i++) g.x[i] = GENERATOR_X[i], g.y[i] = GENERATOR_Y[i], g.z[i] = (i==0) ? 1 : 0;
    results[idx] = mul_glv_opt(g, scalar_limbs);
}


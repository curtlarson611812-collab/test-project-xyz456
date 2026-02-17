/*
 * wNAF (Windowed Non-Adjacent Form) Tables for SpeedBitCrackV3
 *
 * Precomputed tables for efficient scalar multiplication using windowed NAF,
 * reducing the number of point additions required for EC operations.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// wNAF table entry for elliptic curve points
typedef struct {
    uint32_t x[8];  // Affine x-coordinate (32 bytes)
    uint32_t y[8];  // Affine y-coordinate (32 bytes)
} wnaf_point_t;

// wNAF precomputation table for a base point
typedef struct {
    wnaf_point_t points[16];  // 2^(w-1) points for window size w=5 (32 total, but we use 16)
    uint32_t window_size;     // Window size (typically 4-6)
} wnaf_table_t;

// Device function: Compute wNAF representation of a scalar
__device__ void compute_wnaf(
    const uint32_t* scalar,  // Input scalar (8 limbs, 256 bits)
    int8_t* wnaf_digits,     // Output wNAF digits (-(2^w-1) to +(2^w-1))
    int* wnaf_length,        // Output length of wNAF representation
    int window_size = 5      // Window size (affects precomp table size)
) {
    // Copy scalar to avoid modifying original
    uint32_t s[8];
    for (int i = 0; i < 8; i++) s[i] = scalar[i];

    int length = 0;
    int mask = (1 << window_size) - 1;
    int window_mask = mask >> 1;

    while (!is_zero(s, 8)) {
        // Check if current bit is set
        if (s[0] & 1) {
            // Extract window of bits
            int window_value = s[0] & mask;

            // Apply window adjustment for wNAF
            if (window_value > window_mask) {
                window_value -= (1 << window_size);
            }

            wnaf_digits[length++] = window_value;

            // Subtract the window value
            if (window_value >= 0) {
                subtract_bits(s, window_value, 8);
            } else {
                add_bits(s, -window_value, 8);
            }
        } else {
            wnaf_digits[length++] = 0;
        }

        // Shift right by 1
        shift_right_1(s, 8);
    }

    *wnaf_length = length;
}

// Device function: wNAF scalar multiplication using precomputed table
__device__ void wnaf_scalar_mul(
    const wnaf_table_t* table,  // Precomputed wNAF table
    const int8_t* wnaf_digits,  // wNAF digits
    int wnaf_length,            // Length of wNAF
    uint32_t* result_x,         // Output point x-coordinate
    uint32_t* result_y          // Output point y-coordinate
) {
    // Initialize result to point at infinity
    bool infinity = true;
    uint32_t current_x[8] = {0};
    uint32_t current_y[8] = {0};

    // Process wNAF digits from most significant to least
    for (int i = wnaf_length - 1; i >= 0; i--) {
        // Double current point
        if (!infinity) {
            ec_double(current_x, current_y);
        }

        // Add precomputed point if digit is non-zero
        int digit = wnaf_digits[i];
        if (digit != 0) {
            int table_index;
            int sign;

            if (digit > 0) {
                table_index = digit - 1;
                sign = 1;
            } else {
                table_index = (-digit) - 1;
                sign = -1;
            }

            // Add the precomputed point
            if (infinity) {
                // First point addition
                if (sign > 0) {
                    copy_point(current_x, current_y, table->points[table_index].x, table->points[table_index].y);
                } else {
                    copy_point(current_x, current_y, table->points[table_index].x, table->points[table_index].y);
                    ec_negate(current_y);  // Negate y-coordinate
                }
                infinity = false;
            } else {
                // Subsequent point addition
                if (sign > 0) {
                    ec_add(current_x, current_y, table->points[table_index].x, table->points[table_index].y);
                } else {
                    uint32_t neg_y[8];
                    copy_point(neg_y, table->points[table_index].y, 8);
                    ec_negate(neg_y);
                    ec_add(current_x, current_y, table->points[table_index].x, neg_y);
                }
            }
        }
    }

    // Copy result
    if (!infinity) {
        copy_point(result_x, result_y, current_x, current_y);
    } else {
        // Return point at infinity (all zeros)
        for (int i = 0; i < 8; i++) {
            result_x[i] = 0;
            result_y[i] = 0;
        }
    }
}

// Kernel: Precompute wNAF table for a base point
__global__ void precompute_wnaf_table_kernel(
    const uint32_t* base_x,     // Base point x-coordinate
    const uint32_t* base_y,     // Base point y-coordinate
    wnaf_table_t* table,        // Output wNAF table
    int window_size = 5         // Window size
) {
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_index >= (1 << (window_size - 1))) return;

    // Compute odd multiple: (2*point_index + 1) * BasePoint
    // For wNAF, we precompute points for digits 1, 3, 5, ..., (2^w - 1)

    int multiplier = 2 * point_index + 1;

    // Compute multiplier * BasePoint using double-and-add
    uint32_t current_x[8], current_y[8];
    copy_point(current_x, current_y, base_x, base_y);

    for (int i = 1; i < multiplier; i++) {
        ec_double(current_x, current_y);
    }

    // Store in table
    copy_point(table->points[point_index].x, table->points[point_index].y, current_x, current_y);
    table->window_size = window_size;
}

// Kernel: Batch wNAF scalar multiplication
__global__ void batch_wnaf_scalar_mul_kernel(
    const wnaf_table_t* table,      // Precomputed wNAF table
    const uint32_t* scalars,        // Input scalars (batch_size * 8 limbs)
    uint32_t* result_xs,            // Output x-coordinates
    uint32_t* result_ys,            // Output y-coordinates
    int batch_size,                 // Number of scalars to process
    int window_size = 5             // Window size used in table
) {
    int scalar_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (scalar_idx >= batch_size) return;

    // Extract scalar for this thread
    const uint32_t* scalar = &scalars[scalar_idx * 8];

    // Compute wNAF representation
    int8_t wnaf_digits[256];  // Max length for 256-bit scalar
    int wnaf_length;
    compute_wnaf(scalar, wnaf_digits, &wnaf_length, window_size);

    // Perform scalar multiplication
    uint32_t* result_x = &result_xs[scalar_idx * 8];
    uint32_t* result_y = &result_ys[scalar_idx * 8];

    wnaf_scalar_mul(table, wnaf_digits, wnaf_length, result_x, result_y);
}

// Helper device functions
__device__ bool is_zero(const uint32_t* arr, int len) {
    for (int i = 0; i < len; i++) {
        if (arr[i] != 0) return false;
    }
    return true;
}

__device__ void subtract_bits(uint32_t* arr, int value, int len) {
    int borrow = value;
    for (int i = 0; i < len && borrow > 0; i++) {
        int temp = arr[i] - borrow;
        arr[i] = temp & 0xFFFFFFFF;
        borrow = (temp < 0) ? 1 : 0;
    }
}

__device__ void add_bits(uint32_t* arr, int value, int len) {
    int carry = value;
    for (int i = 0; i < len && carry > 0; i++) {
        int temp = arr[i] + carry;
        arr[i] = temp & 0xFFFFFFFF;
        carry = temp >> 32;
    }
}

__device__ void shift_right_1(uint32_t* arr, int len) {
    for (int i = 0; i < len - 1; i++) {
        arr[i] = (arr[i] >> 1) | ((arr[i + 1] & 1) << 31);
    }
    arr[len - 1] >>= 1;
}

__device__ void copy_point(uint32_t* dst_x, uint32_t* dst_y, const uint32_t* src_x, const uint32_t* src_y) {
    for (int i = 0; i < 8; i++) {
        dst_x[i] = src_x[i];
        dst_y[i] = src_y[i];
    }
}

__device__ void copy_point(uint32_t* dst, const uint32_t* src, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

// Placeholder elliptic curve operations (would be implemented with actual EC math)
__device__ void ec_double(uint32_t* x, uint32_t* y) {
    // Placeholder: actual secp256k1 point doubling would go here
    // For now, just increment x for testing
    x[0]++;
}

__device__ void ec_add(uint32_t* x, uint32_t* y, const uint32_t* px, const uint32_t* py) {
    // Placeholder: actual secp256k1 point addition would go here
    // For now, add px to x for testing
    for (int i = 0; i < 8; i++) {
        x[i] += px[i];
        y[i] += py[i];
    }
}

__device__ void ec_negate(uint32_t* y) {
    // Placeholder: actual secp256k1 y-coordinate negation would go here
    // For now, flip sign bit
    y[7] ^= 0x80000000;
}

// Host function: Create wNAF table
extern "C" cudaError_t create_wnaf_table(
    const uint32_t* d_base_x,      // Device base point x
    const uint32_t* d_base_y,      // Device base point y
    wnaf_table_t** d_table,        // Output device table pointer
    int window_size = 5,           // Window size
    cudaStream_t stream = 0
) {
    cudaError_t error;

    // Allocate table on device
    error = cudaMalloc(d_table, sizeof(wnaf_table_t));
    if (error != cudaSuccess) return error;

    // Launch precomputation kernel
    int num_points = 1 << (window_size - 1);  // 2^(w-1) points
    int threads_per_block = 256;
    int blocks = (num_points + threads_per_block - 1) / threads_per_block;

    precompute_wnaf_table_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_base_x, d_base_y, *d_table, window_size
    );

    return cudaGetLastError();
}

// Host function: Batch wNAF scalar multiplication
extern "C" cudaError_t batch_wnaf_scalar_mul(
    const wnaf_table_t* d_table,   // Device wNAF table
    const uint32_t* d_scalars,     // Device scalars
    uint32_t* d_result_xs,         // Device result x-coordinates
    uint32_t* d_result_ys,         // Device result y-coordinates
    int batch_size,                // Number of scalars
    int window_size = 5,           // Window size
    cudaStream_t stream = 0
) {
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    batch_wnaf_scalar_mul_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_table, d_scalars, d_result_xs, d_result_ys, batch_size, window_size
    );

    return cudaGetLastError();
}

// Host function: Destroy wNAF table
extern "C" cudaError_t destroy_wnaf_table(wnaf_table_t* d_table) {
    return cudaFree(d_table);
}
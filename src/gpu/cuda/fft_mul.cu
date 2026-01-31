// fft_mul.cu - CUDA kernel for big integer multiplication using cuFFT
// Implements Schönhage-Strassen algorithm: FFT-based polynomial multiplication
// Converts bigints to polynomials, FFT multiply, inverse FFT, carry propagation

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Convert bigint limbs to complex polynomial coefficients for FFT
__global__ void bigint_to_complex(
    cufftDoubleComplex *poly,  // Output polynomial [batch][n]
    uint32_t *bigint_limbs,    // Input bigints [batch][limbs]
    int batch_size, int limbs, int n  // n = next power of 2 >= 2*limbs
) {
    int batch_idx = blockIdx.x;
    int coeff_idx = threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= n) return;

    if (coeff_idx < limbs) {
        // Copy limb as real coefficient (little-endian)
        uint32_t limb = bigint_limbs[batch_idx * limbs + coeff_idx];
        poly[batch_idx * n + coeff_idx].x = (double)limb;
        poly[batch_idx * n + coeff_idx].y = 0.0;
    } else {
        // Pad with zeros for FFT
        poly[batch_idx * n + coeff_idx].x = 0.0;
        poly[batch_idx * n + coeff_idx].y = 0.0;
    }
}

// Pointwise complex multiplication for FFT convolution
__global__ void pointwise_complex_mul(
    cufftDoubleComplex *a, cufftDoubleComplex *b,
    cufftDoubleComplex *result, int batch_size, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n) return;

    double ar = a[idx].x, ai = a[idx].y;
    double br = b[idx].x, bi = b[idx].y;

    // Complex multiplication: (ar + ai*i) * (br + bi*i)
    result[idx].x = ar * br - ai * bi;
    result[idx].y = ar * bi + ai * br;
}

// Convert FFT result back to bigint with proper carry propagation
__global__ void complex_to_bigint_with_carry(
    uint32_t *bigint_result,   // Output bigints [batch][result_limbs]
    cufftDoubleComplex *fft_result, // FFT result [batch][n]
    double scale,              // 1/n scaling factor
    int batch_size, int n, int result_limbs
) {
    int batch_idx = blockIdx.x;
    int limb_idx = threadIdx.x;

    if (batch_idx >= batch_size || limb_idx >= result_limbs) return;

    uint32_t *result = &bigint_result[batch_idx * result_limbs];
    uint64_t sum = 0;

    // Sum all FFT coefficients that contribute to this output limb
    // For limb k, sum fft_result[k] + fft_result[k+n] + ... (circular convolution)
    for (int i = limb_idx; i < n; i += result_limbs) {
        double coeff = fft_result[batch_idx * n + i].x * scale;
        uint64_t val = (uint64_t)(coeff + 0.5); // Round to nearest integer
        sum += val;
    }

    // Apply carry propagation using warp shuffle for efficiency
    uint32_t carry_in = 0;
    if (limb_idx > 0) {
        // Get carry from previous limb via warp shuffle
        carry_in = __shfl_sync(0xFFFFFFFF, (uint32_t)(sum >> 32), limb_idx - 1);
    }

    sum += carry_in;

    // Store result and propagate carry to next limb
    result[limb_idx] = sum & 0xFFFFFFFFULL;

    if (limb_idx < result_limbs - 1) {
        uint32_t carry_out = (sum >> 32) & 0xFFFFFFFFULL;
        // Use atomicAdd for carry propagation to next limb
        atomicAdd(&result[limb_idx + 1], carry_out);
    }
}

// Host function for cuFFT-based batch big integer multiplication
extern "C" cudaError_t batch_bigint_mul_cufft(
    cufftHandle forward_plan, cufftHandle inverse_plan,
    uint32_t *d_a_limbs,      // Device: [batch][limbs_a] input a
    uint32_t *d_b_limbs,      // Device: [batch][limbs_b] input b
    uint32_t *d_result_limbs, // Device: [batch][limbs_result] outputs
    int batch_size, int limbs_a, int limbs_b,
    cudaStream_t stream
) {
    cudaError_t cuda_status;
    cufftResult cufft_status;

    // Calculate FFT size (next power of 2 >= limbs_a + limbs_b)
    int n = 1;
    while (n < limbs_a + limbs_b) n <<= 1;
    int result_limbs = limbs_a + limbs_b;

    // Allocate FFT working buffers
    cufftDoubleComplex *d_a_poly, *d_b_poly, *d_result_poly;
    size_t poly_size = batch_size * n * sizeof(cufftDoubleComplex);

    cuda_status = cudaMallocAsync(&d_a_poly, poly_size, stream);
    if (cuda_status != cudaSuccess) return cuda_status;

    cuda_status = cudaMallocAsync(&d_b_poly, poly_size, stream);
    if (cuda_status != cudaSuccess) {
        cudaFreeAsync(d_a_poly, stream);
        return cuda_status;
    }

    cuda_status = cudaMallocAsync(&d_result_poly, poly_size, stream);
    if (cuda_status != cudaSuccess) {
        cudaFreeAsync(d_a_poly, stream);
        cudaFreeAsync(d_b_poly, stream);
        return cuda_status;
    }

    // Convert bigints to complex polynomials
    dim3 conv_grid(batch_size);
    dim3 conv_block(n);

    bigint_to_complex<<<conv_grid, conv_block, 0, stream>>>(
        d_a_poly, d_a_limbs, batch_size, limbs_a, n);

    bigint_to_complex<<<conv_grid, conv_block, 0, stream>>>(
        d_b_poly, d_b_limbs, batch_size, limbs_b, n);

    // Forward FFT on both polynomials
    cufft_status = cufftExecZ2Z(forward_plan, d_a_poly, d_a_poly, CUFFT_FORWARD);
    if (cufft_status != CUFFT_SUCCESS) {
        cuda_status = cudaErrorUnknown;
        goto cleanup;
    }

    cufft_status = cufftExecZ2Z(forward_plan, d_b_poly, d_b_poly, CUFFT_FORWARD);
    if (cufft_status != CUFFT_SUCCESS) {
        cuda_status = cudaErrorUnknown;
        goto cleanup;
    }

    // Pointwise complex multiplication (convolution theorem)
    dim3 mul_grid((batch_size * n + 255) / 256);
    dim3 mul_block(256);

    pointwise_complex_mul<<<mul_grid, mul_block, 0, stream>>>(
        d_a_poly, d_b_poly, d_result_poly, batch_size, n);

    // Inverse FFT on product
    cufft_status = cufftExecZ2Z(inverse_plan, d_result_poly, d_result_poly, CUFFT_INVERSE);
    if (cufft_status != CUFFT_SUCCESS) {
        cuda_status = cudaErrorUnknown;
        goto cleanup;
    }

    // Convert back to bigint with carry propagation
    dim3 carry_grid(batch_size);
    dim3 carry_block(result_limbs);

    complex_to_bigint_with_carry<<<carry_grid, carry_block, 0, stream>>>(
        d_result_limbs, d_result_poly, 1.0 / n, batch_size, n, result_limbs);

    cuda_status = cudaGetLastError();

cleanup:
    cudaFreeAsync(d_result_poly, stream);
    cudaFreeAsync(d_b_poly, stream);
    cudaFreeAsync(d_a_poly, stream);

    return cuda_status;
}
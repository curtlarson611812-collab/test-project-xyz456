// Chunk: GPU Slice Refine (new refine_kernel.cu)
__global__ void refine_slices_gpu(PosSlice* slices, const float* biases, int count, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    PosSlice* s = &slices[idx];
    if (s->iter >= max_iter) return;
    uint256_t r = sub256(s->high, s->low);  // BigInt as uint256_t
    float b = biases[s->proxy % 81];        // Assume mod81
    s->low = add256(s->low, div256(r, 12));
    s->high = add256(s->low, mul256(r, uint256_from_float(b * 1.1f)));
    s->bias *= b;
    s->iter++;
}
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    int tx = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tx;
    __shared__ float out[256];

    if(id < N)
        out[tx] = input[id];
    else
        out[tx] = 0.0f;
    __syncthreads();

    for(int n = 128; n > 32 ; n /= 2){
        if(tx < n) 
            out[tx] += out[tx + n];
        __syncthreads();
    }

    if(tx < 32){
        float val = out[tx] + out[tx + 32];
        for(int off = 16; off > 0; off /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, off);
        
        if(tx == 0)
            atomicAdd(result, val);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}

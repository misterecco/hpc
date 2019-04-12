#include "./common/helpers.h"

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    if (tid < N) {
        int tid1 = (tid + 1) % 256;
        int tid2 = (tid + 2) % 256;
        float aSum = (a[tid] + a[tid1] + a[tid2]) / 3.0f;
        float bSum = (b[tid] + b[tid1] + b[tid2]) / 3.0f;
        c[tid] = (aSum + bSum) / 2;
    }
}

int main(void) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&(streams[i]));
    }

    int *host_a, *host_b, *host_c;
    int *dev_a[2], *dev_b[2], *dev_c[2];

    for (int i = 0; i < 2; i++) {
	HANDLE_ERROR(cudaMalloc((void**)&(dev_a[i]), N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&(dev_b[i]), N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&(dev_c[i]), N * sizeof(int)));
    }

    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    for (int i = 0, j = 0; i < FULL_DATA_SIZE; i += N, j++) {
        const cudaStream_t& stream = streams[j%2];
        HANDLE_ERROR(cudaMemcpyAsync(dev_a[j%2], host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b[j%2], host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

        kernel<<<N / 256, 256, 0, stream>>>(dev_a[j%2], dev_b[j%2], dev_c[j%2]);

        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c[j%2], N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    for (int i = 0; i < 2; i++) {
    	HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time taken: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    
    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
   

    for (int i = 0; i < 2; i++) {
        HANDLE_ERROR(cudaFree(dev_a[i]));
        HANDLE_ERROR(cudaFree(dev_b[i]));
        HANDLE_ERROR(cudaFree(dev_c[i]));
    	HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    return 0;
}

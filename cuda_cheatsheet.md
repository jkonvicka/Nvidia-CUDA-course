# CUDA cheatsheet

### Kernel declaration
``` cpp
__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}
```
### Calling kernel
```cpp
	//n_blocks, n_threads
GPUFunction<<<1, 1>>>();
```

### CUDA Synchronization
```cpp
cudaDeviceSynchronize();
```

### Mapping cuda indexes into index of data
``` cpp
int data_index = threadIdx.x + blockIdx.x * blockDim.x;
```

### Computing number of needed blocks for data
``` cpp
int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
```

### Manging shared memory
``` cpp
int N = 100;
int *a;
size_t size = N * sizeof(int);
cudaMallocManaged(&a, size);
cudaFree(a);
```

### Grid stride loop
``` cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = idx; i < N; i += stride)
{
	a[i] = a[i]; //do work
}
```

### Shared memory management error handling
``` cpp
cudaError_t err;
err = cudaMallocManaged(&a, N);                    	// Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           	// `cudaSuccess` is provided by CUDA.
{
	printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}

```

### Kernel running error handling
``` cpp
someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
	printf("Error: %s\n", cudaGetErrorString(err));
}
```

### General CUDA error handling function
``` cpp
inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}
```
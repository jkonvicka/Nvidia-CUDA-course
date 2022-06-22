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
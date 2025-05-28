
#include <stdio.h>
#define	SIZE 100


__global__ void	fire(float *test) {
	int const	i = threadIdx.x;

	if (i < SIZE) test[i] += 1.; 
}

int	main( void ) {
	float	*h_array;
	float	*d_array;

	unsigned int const	bytes = SIZE * sizeof(float);

	cudaMallocHost((void**)&h_array, bytes);
	cudaMalloc((void**)&d_array, bytes);

	for (int idx = 0; idx < SIZE; idx++) h_array[idx] = (float)idx;

	cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

	cudaEvent_t	startEvent, stopEvent;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
	fire<<<1, SIZE>>>(d_array);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	float time;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	printf("  KernelCall (ms): %f\n", time);

	cudaFree(d_array);
	cudaFreeHost(h_array);

	return 0;
}
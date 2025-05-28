
#include <iostream>
#include <assert.h>

#define	H 3
#define	W 5

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}

__global__	void	fire(float *input, float *weight, size_t pitch, float *res) {
	int	i = threadIdx.x;

	if (i < H) {
		for (int idx = 0; idx < W; idx++) {
			float	*row = (float*)((char*)weight + (i * pitch));
			res[i] += input[idx] * row[idx];
		}
	}
}

int main(void) {

	float	*h_test = NULL;
	float	*h_alpha = NULL;
	float	*h_result = NULL;

	checkCuda( cudaMallocHost((void**)&h_test,W*H*sizeof(float)) );
	float	num = 0;
	for (int i = 0; i < W * H; i++, num++) h_test[i] = num;

	checkCuda( cudaMallocHost((void**)&h_alpha, W*sizeof(float)) );
	num = 0;
	for (int i = 0; i < H; i++, num++) h_alpha[i] = num;

	checkCuda( cudaMallocHost((void**)&h_result, H*sizeof(float)) );

	float	*d_test = NULL;
	float	*d_alpha = NULL;
	float	*d_result = NULL;

	size_t	pitch_test = 0;
	checkCuda( cudaMallocPitch((void**)&d_test, &pitch_test, W*sizeof(float), H) );
	checkCuda( cudaMemcpy2D(d_test, pitch_test, h_test, W*sizeof(float), W*sizeof(float), H, cudaMemcpyHostToDevice) );

	checkCuda( cudaMalloc((void**)&d_alpha, W*sizeof(float)) );
	checkCuda( cudaMemcpy(d_alpha, h_alpha, W*sizeof(float), cudaMemcpyHostToDevice) );

	checkCuda( cudaMalloc((void**)&d_result, H*sizeof(float)) );
	checkCuda( cudaMemset(d_result, 0, H*sizeof(float)) );

	fire<<<1, H>>>(d_alpha, d_test, pitch_test, d_result);

	checkCuda( cudaMemcpy(h_result, d_result, H*sizeof(float), cudaMemcpyDeviceToHost) );

	printf("result :");
	for (int i = 0; i < H; i++) {
		printf(" %f ", h_result[i]);
	}
	printf("\n");

	cudaFree(d_alpha);
	cudaFree(d_test);
	cudaFree(d_result);
	cudaFreeHost(h_alpha);
	cudaFreeHost(h_test);
	cudaFreeHost(h_result);
	return 0;
}
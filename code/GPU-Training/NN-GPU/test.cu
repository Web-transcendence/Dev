
#include <iostream>

#define	H 3
#define	W 5

int main(void) {

	float	*h_test;

	cudaMallocHost((void**)h_test,W * H * sizeof(float));
	
	float	num = 0;
	for (int i = 0; i < W * H; i++) h_test[i] = num;

	float	**d_test;
	size_t	pitch;

	cudaMallocPitch((void**)d_test, &pitch, W * sizeof(float), H);
	
	cudaMemcpy2D();
	return 0;
}
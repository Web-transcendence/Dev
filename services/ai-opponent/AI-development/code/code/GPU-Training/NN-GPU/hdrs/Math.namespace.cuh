
namespace Math {
	// ACTIVATION FUNCTIONS
	__device__ inline float		sigmoid(float const z) {return (1.0/(1.0+std::exp(-z)));}

	__device__ void				sigmoid(float const * zs, float *act, int const size) {
		for (int idx = 0; idx < size; idx++) act[idx] = Math::sigmoid(zs[idx]);
	}

	__device__ inline float		sigmoidPrime(float const z) {return (Math::sigmoid(z)*(1.0 - Math::sigmoid(z))) ;}

	__device__ void				sigmoidPrime(float const * zs, float *act, int const size) {
		for (int idx = 0; idx < size; idx++) act[idx] = Math::sigmoidPrime(zs[idx]);
	}

	

}
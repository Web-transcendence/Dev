
#include "Network.class.cuh"

#include "Math.namespace.cuh"

#include <random>
#include <array>
#include <algorithm>

__host__	Network::Network( void ) {
	this->bitsWeights = WEIGHT_GLOBAL * sizeof(float);
	checkCuda( cudaMallocHost((void**)&this->h_weights, this->bitsWeights) );
	checkCuda( cudaMalloc((void**)&this->d_weights, this->bitsWeights) );

	this->bitsBiais = BIAI_GLOBAL * sizeof(float);
	checkCuda( cudaMallocHost((void**)&this->h_biais, this->bitsBiais));
	checkCuda( cudaMalloc((void**)&this->d_biais, this->bitsBiais));

	this->fillRandomlyArray(this->h_weights, WEIGHT_GLOBAL);
	this->fillRandomlyArray(this->h_biais, BIAI_GLOBAL);

	checkCuda( cudaMemcpy(this->d_weights, this->h_weights, this->bitsWeights, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(this->d_biais, this->h_biais, this->bitsBiais, cudaMemcpyHostToDevice) );

	auto	sizes = std::array<unsigned int, 3>{N_INPUT, N_HIDDEN, N_OUTPUT};
	this->maxSizeInput = *std::max_element(sizes.begin(), sizes.end());

	t_lArch	layers[L_GLOBAL - 1];
	layers[0].set(std::array<unsigned int, 3>{});
	for (unsigned int idx = 0; idx < L_GLOBAL - 1; idx++) {
		if (!idx) {
			
		}
	}
	return ;
}

__host__	Network::~Network( void ) {
	if (this->h_weights)
		cudaFreeHost(this->h_weights);
	if (this->h_biais)
		cudaFreeHost(this->h_biais);
	if (this->d_weights)
		cudaFreeHost(this->d_weights);
	if (this->d_biais)
		cudaFreeHost(this->d_biais);
	if (this->d_trainingData)
		cudaFreeHost(this->d_trainingData);
}

__device__ void	activationNetwork(float *activation, float *zs, float const *weights, float const *biais, t_lArch const & layer, unsigned int const sharedSize) {
	extern __shared__ float	resNeuron[];

	unsigned int const	tIdx = threadIdx.x;
	unsigned int const	bIdx = blockIdx.x;

	if (!(bIdx < layer.neurons && tIdx < layer.weights))
		return ;
	unsigned int const	i_weight = layer.s_idxWeights + (bIdx * layer.weights) + tIdx;
	resNeuron[tIdx] = activation[tIdx] * weights[i_weight];

	__syncthreads();

	if (tIdx == 0) {
		unsigned int const idxZs = layer.s_idxBiais; 
		zs[idxZs] = 0.;
		for (unsigned int sharedIdx = 0; sharedIdx < sharedSize; sharedIdx++) zs[idxZs] += resNeuron[sharedIdx];
		zs[idxZs] += biais[layer.s_idxBiais + bIdx + tIdx];
		activation[idxZs + N_INPUT] = Math::sigmoid(zs[idxZs]);
	}
	return ;
}

__host__ void	Network::SDG(std::vector<t_tuple*> &trainingData, double const eta) {
	float	*d_activation;
	float	*d_zs;

	unsigned int const	inputBytes = this->maxSizeInput * sizeof(float);
	unsigned int const	actBytes =  inputBytes * L_GLOBAL;

	checkCuda( cudaMalloc((void**)&d_activation, actBytes) );
	checkCuda( cudaMalloc((void**)&d_zs, actBytes) );

	this->trainingDataAllocation(trainingData);

	checkCuda( cudaMemcpy(d_activation, this->d_trainingData, inputBytes, cudaMemcpyDeviceToDevice) );
	for (int layer = 1; layer < L_GLOBAL; layer++) {
		unsigned int const	threadBlocks = this->sizes[layer]; // mettre en puissance de 2
		unsigned int const	threads = this->sizes[layer - 1]; // idem
		unsigned int const	sharedBytes = threads * sizeof(float);
		activationNetwork<<<threadBlocks, threads, sharedBytes>>>(d_activation, d_zs, this->d_weights, this->d_biais, ,sharedBytes);
	}

	cudaFree(d_activation);
	cudaFree(d_zs);
	cudaFree(this->d_trainingData);
	return ;
}

void	Network::trainingDataAllocation(std::vector<t_tuple*> const & trainingData) {
	float	*h_trainingData;

	size_t const	bits = (N_INPUT + N_OUTPUT) * BATCH_SIZE * sizeof(float);
	checkCuda( cudaMallocHost((void**)&h_trainingData, bits) );

	size_t const	bitsInput = N_INPUT*sizeof(float);
	size_t const	bitsOutput = N_OUTPUT*sizeof(float);

	size_t	start = 0;
	for (auto it = trainingData.begin(); it != trainingData.end(); it++, start += N_OUTPUT) {
		memcpy(&h_trainingData[start], (*it)->input, bitsInput);
		start += N_INPUT;
		memcpy(&h_trainingData[start], (*it)->expectedOutput, bitsOutput);
	}

	checkCuda( cudaMalloc(&this->d_trainingData, bits) );
	checkCuda( cudaFreeHost(h_trainingData) );
	return ;
}

void	Network::fillRandomlyArray(float *myArray, size_t const size) {
	std::random_device					rd;
	std::mt19937						gen(rd());
	double 								stddev = 1.0 / std::sqrt(WEIGHT_GLOBAL);
	std::normal_distribution<float> 	dist(0.0, stddev);

	for (size_t idx = 0; idx < size; idx++) myArray[idx] = dist(gen);
	return ;
}
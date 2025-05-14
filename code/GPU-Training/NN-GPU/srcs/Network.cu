
#include "Network.class.cuh"

#include <random>

__host__ Network::Network( void ) {
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
	return ;
}

__host__ void	Network::SDG(std::vector<t_tuple*> &trainingData, double const eta) {
	
	this->trainingDataAllocation(trainingData);
	
	checkCuda( cudaFree(this->d_trainingData) );
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
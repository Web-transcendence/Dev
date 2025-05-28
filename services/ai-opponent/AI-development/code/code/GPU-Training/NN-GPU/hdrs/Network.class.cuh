
#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP

#include "define.cuh"

#include <vector>

class	Network {
public:
	__host__ Network( void );
	__host__ ~Network( void );

	__host__ void	SDG(std::vector<t_tuple*> &trainingData, double const eta);

private:
	__host__ void	fillRandomlyArray(float	*myArray, size_t const size);
	__host__ void	trainingDataAllocation(std::vector<t_tuple*> const & trainingData);

	float	*h_weights;
	float	*h_biais;

	float	*d_weights;
	float	*d_biais;

	size_t	bitsWeights;
	size_t	bitsBiais;

	size_t	maxSizeInput;

	float	*d_trainingData;
};

#endif

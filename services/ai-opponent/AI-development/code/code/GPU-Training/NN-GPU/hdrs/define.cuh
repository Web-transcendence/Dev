
#ifndef	DEFINE_HPP
# define DEFINE_HPP

#define L_HIDDEN 1
#define	L_GLOBAL 3
#define N_INPUT 10
#define N_OUTPUT 10
#define N_HIDDEN 30
#define N_GLOBAL (N_INPUT + N_HIDDEN*(L_HIDDEN-1) + N_OUTPUT) 

#define	WEIGHT_FIRST_HIDDEN (N_INPUT * N_HIDDEN)
#define	WEIGHT_HIDDEN (N_HIDDEN * N_HIDDEN)
#define	WEIGHT_OUTPUT (N_HIDDEN * N_OUTPUT)
#define WEIGHT_GLOBAL (WEIGHT_FIRST_HIDDEN + WEIGHT_HIDDEN*(L_HIDDEN-1) + WEIGHT_OUTPUT)

#define	BIAI_FIRST_HIDDEN N_INPUT
#define	BIAI_HIDDEN N_HIDDEN
#define	BIAI_OUTPUT N_OUTPUT
#define BIAI_GLOBAL (BIAI_FIRST_HIDDEN + BIAI_HIDDEN*(L_HIDDEN-1) + BIAI_OUTPUT)

#define	BATCH_SIZE 32
#define SIZE_TRAININGDATA_BATCH ()

#include <exception>
#include <array>
#include <stdio.h>

class CudaMallocException : std::exception {
	char	*what(void) const throw() {
		return "CudaMalloc failed";
	}
};

class MallocException : std::exception {
	char	*what(void) const throw() {
		return "Malloc failed";
	}
};

typedef struct	s_lArch {
	unsigned int	n_Layer;
	unsigned int	neurons;
	unsigned int	weights;

	unsigned int	s_idxWeights;
	unsigned int	s_idxBiais;

	void	set(std::array<unsigned int, 3>	const & val) {
		n_Layer = val[0];
		neurons = val[1];
		weights = val[2];
	}; 
}	t_lArch;

typedef struct  s_tuple {
    float	*input;
    float	*expectedOutput;
}      t_tuple;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}

#endif

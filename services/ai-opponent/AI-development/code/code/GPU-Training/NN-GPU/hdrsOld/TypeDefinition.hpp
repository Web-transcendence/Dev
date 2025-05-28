/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/08 22:49:15 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define N_LAYER_HIDDEN 1
# define N_NEURON_HIDDEN 30
# define N_NEURON_INPUT 784
# define N_NEURON_OUTPUT 10
# include <exception>
# include <vector>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double*);

typedef	enum e_actFunc {SIGMOID, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

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

typedef struct  s_tuple {
    double *input;
    double *expectedOutput;

	s_tuple() {
		cudaError_t	err[2];
		err[0] = cudaMalloc(&this->input, N_NEURON_INPUT * sizeof(double));
		err[1] = cudaMalloc(&this->expectedOutput, N_NEURON_OUTPUT * sizeof(double));
		if (!err[0] || !err[1])
			throw CudaMallocException();
	}

	~s_tuple() {
		if (this->input) {cudaFree(input);}
		if (this->expectedOutput) {cudaFree(expectedOutput);}
	}
}      t_tuple;

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.namespace.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 09:50:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/01 10:09:19 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MATH_NAMESPACE_HPP
# define MATH_NAMESPACE_HPP
# include <vector>
# include "TypeDefinition.hpp"

#include <iostream>

namespace Math {
	// ACTIVATION FUNCTIONS
	__device__ inline void		sigmoid(double *z);
	__device__ void				sigmoid(double *zs, int const size);
	__device__ inline void		sigmoidPrime(double *z);
	__device__ void				sigmoidPrime(double *zs, int const size);

	__device__ inline void 		reLu(double *z);
	__device__ void 			reLu(double *zs, int const size);
	__device__ inline void 		reLuPrime(double *z);
	__device__ void				reLuPrime(double *zs, int const size);

	__device__ inline void 		leakyReLu(double *z);
	__device__ void				leakyReLu(double *zs, int const size);
	__device__ inline void 		leakyReLuPrime(double *z);
	__device__ void				leakyReLuPrime(double *zs, int const size);
	
	__device__ inline void 		tanh(double *z);
	__device__ void				tanh(double *zs, int const size);
	__device__ inline void 		tanhPrime(double *z);
	__device__ void				tanhPrime(double *zs, int const size);

	__device__ inline void 		step(double *z);
	__device__ void				step(double *zs, int const size);
	__device__ inline void 		stepPrime(double *z);
	__device__ void				stepPrime(double *zs, int const size);

	extern double(*const actFuncS[5])(double);
	extern std::vector<double>*(*const actFuncV[5])(std::vector<double> const &);
	extern double(*const primeActFuncS[5])(double);
	extern std::vector<double>*(*const primeActFuncV[5])(std::vector<double> const &);

	// COST FUNCTIONS
	double								costDerivative(double& output, double& expected);
	std::vector<double>*				costDerivative(std::vector<double> const & output, std::vector<double> const & expected);
	double								sqCostDerivative(double const & output, double const & expected);
	std::vector<double>*				sqCostDerivative(std::vector<double> const & output, std::vector<double> const & expected);
	
	// PRODUCT FUNCTIONS
	__device__ void						dotProduct(double const *v1, double const *v2, double *res, int const size);
	std::vector<double>*				hadamardProduct(std::vector<double> const & lhs, std::vector<double> const & rhs);
	std::vector<std::vector<double>>*	outerProduct(std::vector<double> const & in, std::vector<double> const & transposed);

	// TRANSPOSITION FUNCTIONS
	std::vector<std::vector<double>>*	transpose1D(std::vector<double> const & base);
	std::vector<std::vector<double>>*	transpose2D(std::vector<std::vector<double>> const & base);

	// DEBUG
	template<typename T>
	void	printdebug(T const & cont, std::string const & name) {
		std::cout<<name<<":[";
		for (auto c : cont)
			std::cout<<c<<";";
		std::cout<<"]"<<std::endl;
	}
};

#endif
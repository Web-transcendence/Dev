/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.namespace.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 09:50:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/31 12:22:16 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MATH_NAMESPACE_HPP
# define MATH_NAMESPACE_HPP
# include <vector>
# include "TypeDefinition.hpp"

#include <iostream>

namespace Math {
	// ACTIVATION FUNCTIONS
	double								sigmoid(double const z);
	std::vector<double>*				sigmoid(std::vector<double> const & zs);
	double								sigmoidPrime(double const z);
	std::vector<double>*				sigmoidPrime(std::vector<double> const & zs);

	double								reLu(double const z);
	std::vector<double>*				reLu(std::vector<double> const & zs);
	double								reLuPrime(double const z);
	std::vector<double>*				reLuPrime(std::vector<double> const & zs);

	double								leakyReLu(double const z);
	std::vector<double>*				leakyReLu(std::vector<double> const & zs);
	double								leakyReLuPrime(double const z);
	std::vector<double>*				leakyReLuPrime(std::vector<double> const & zs);
	
	double								tanh(double const z);
	std::vector<double>*				tanh(std::vector<double> const & zs);
	double								tanhPrime(double const z);
	std::vector<double>*				tanhPrime(std::vector<double> const & zs);

	double								step(double const z);
	std::vector<double>*				step(std::vector<double> const & zs);
	double								stepPrime(double const z);
	std::vector<double>*				stepPrime(std::vector<double> const & zs);

	double(*actFuncS[5])(double) = \
	{static_cast<double(*)(double const)>(&sigmoid),\
	static_cast<double(*)(double const)>(&reLu),\
	static_cast<double(*)(double const)>(&leakyReLu),\
	static_cast<double(*)(double const)>(&tanh),\
	static_cast<double(*)(double const)>(&step)};

	std::vector<double>*(*actFuncV[5])(std::vector<double> const &) = \
	{static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::sigmoid),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::reLu),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::leakyReLu),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::tanh),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::step)};

	double(*primeActFuncS[5])(double) = \
	{static_cast<double(*)(double const)>(&Math::sigmoidPrime),\
	static_cast<double(*)(double const)>(&Math::reLuPrime),\
	static_cast<double(*)(double const)>(&Math::leakyReLuPrime),\
	static_cast<double(*)(double const)>(&Math::tanhPrime),\
	static_cast<double(*)(double const)>(&Math::stepPrime)};

	std::vector<double>*(*primeActFuncV[5])(std::vector<double> const &) = \
	{static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::sigmoidPrime),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::reLuPrime),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::leakyReLuPrime),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::tanhPrime),\
	static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::stepPrime)};

	// COST FUNCTIONS
	double								costDerivative(double& output, double& expected);
	std::vector<double>*				costDerivative(std::vector<double> const & output, std::vector<double> const & expected);
	double								sqCostDerivative(double const & output, double const & expected);
	std::vector<double>*				sqCostDerivative(std::vector<double> const & output, std::vector<double> const & expected);
	
	// PRODUCT FUNCTIONS
	double								dotProduct(std::vector<double> const & v1, std::vector<double> const & v2);
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
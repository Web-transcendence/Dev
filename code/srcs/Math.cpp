/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:09:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/17 14:58:32 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Math.namespace.hpp"
#include <cmath>

double	Math::sigmoid(double const z) {return 1.0/(1.0+std::exp(-z));}
	
std::vector<double>*	Math::sigmoid(std::vector<double> const & zs) {
	auto	res = new std::vector<double>;
	
	for (auto it = zs.begin(); it != zs.end(); it++)
		res->push_back(Math::sigmoid(*it));
	return res;
}

double	Math::sigmoidPrime(double const z) {return Math::sigmoid(z)*(1 - Math::sigmoid(z));}

std::vector<double>*	Math::sigmoidPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>;
	
	for (auto it = zs.begin(); it != zs.end(); it++)
		res->push_back(Math::sigmoidPrime(*it));
	return res;
}
	
double	Math::cost_derivative(double& output, double& expected) {return output - expected;}

std::vector<double>*	Math::cost_derivative(std::vector<double> const & output, std::vector<double> const & expected) {
	auto	res = new std::vector<double>;
	
	for (auto it_o = output.begin(), it_e = expected.begin(); it_o != output.end() && it_e != expected.end(); it_o++, it_e++)
		res->push_back((*it_o) - (*it_e));
	return res;
}

double	Math::dotProduct(std::vector<double> const & v1, std::vector<double> const & v2) {
	double	res = 0.0;
	
	for (auto it_w = v2.begin(), it_i = v1.begin(); it_w != v2.end() && it_i != v1.end(); it_w++, it_i++)
		res += ((*it_i) * (*it_w));
	return res;
}
	
std::vector<double>*	Math::hadamardProduct(std::vector<double> const & lhs, std::vector<double> const & rhs) {
	auto	product = new std::vector<double>;

	for (auto it_lhs = lhs.begin(), it_rhs = rhs.begin(); it_lhs != lhs.end() && it_rhs != rhs.end(); it_lhs++, it_rhs++)
		product->push_back((*it_rhs) * (*it_lhs));
	return product;
}

std::vector<std::vector<double>*>*	Math::outerProduct(std::vector<double> const & in, std::vector<double> const & transposed) {
	auto	res = new std::vector<std::vector<double>*>;

	for (auto it_in = in.begin(); it_in != in.end(); it_in++) {
		res->push_back(new std::vector<double>);
		for (auto it_tr = transposed.begin(); it_tr != transposed.end(); it_tr++)
			res->back()->push_back((*it_in) * (*it_tr));
	}
	return res;
}

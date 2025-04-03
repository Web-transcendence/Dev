/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:09:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/01 10:55:41 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Math.namespace.hpp"
#include <cmath>

namespace Math {	
	double(*const actFuncS[5])(double) = {\
		static_cast<double(*)(double const)>(&Math::sigmoid),\
		static_cast<double(*)(double const)>(&Math::reLu),\
		static_cast<double(*)(double const)>(&Math::leakyReLu),\
		static_cast<double(*)(double const)>(&Math::tanh),\
		static_cast<double(*)(double const)>(&Math::step)};
		
	std::vector<double>*(*const actFuncV[5])(std::vector<double> const &) = {\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::sigmoid),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::reLu),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::leakyReLu),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::tanh),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::step)};
		
	double(*const primeActFuncS[5])(double) = {\
		static_cast<double(*)(double const)>(&Math::sigmoidPrime),\
		static_cast<double(*)(double const)>(&Math::reLuPrime),\
		static_cast<double(*)(double const)>(&Math::leakyReLuPrime),\
		static_cast<double(*)(double const)>(&Math::tanhPrime),\
		static_cast<double(*)(double const)>(&Math::stepPrime)};
		
	std::vector<double>*(*const primeActFuncV[5])(std::vector<double> const &) = {\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::sigmoidPrime),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::reLuPrime),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::leakyReLuPrime),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::tanhPrime),\
		static_cast<std::vector<double>*(*)(std::vector<double> const &)>(&Math::stepPrime)};
}

// ACTIVATION FONCTION
double	Math::sigmoid(double const z) {return (1.0/(1.0+std::exp(-z)));}
	
std::vector<double>*	Math::sigmoid(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::sigmoid(*it_zs);
	return res;
}

double	Math::sigmoidPrime(double const z) {return (Math::sigmoid(z)*(1.0 - Math::sigmoid(z))) ;}

std::vector<double>*	Math::sigmoidPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::sigmoidPrime(*it_zs);
	return res;
}

double	Math::reLu(double const z) {return (z > 0. ? z : 0.);}
	
std::vector<double>*	Math::reLu(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::reLu(*it_zs);
	return res;
}

double	Math::reLuPrime(double const z) {return (z > 0. ? 1. : 0.);}

std::vector<double>*	Math::reLuPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::reLuPrime(*it_zs);
	return res;
}

double	Math::leakyReLu(double const z) {return (z > 0. ? z : 0.01*z);}
	
std::vector<double>*	Math::leakyReLu(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::leakyReLu(*it_zs);
	return res;
}

double	Math::leakyReLuPrime(double const z) {return (z > 0. ? 1. : 0.01);}

std::vector<double>*	Math::leakyReLuPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::leakyReLuPrime(*it_zs);
	return res;
}

double	Math::tanh(double const z) {return ((std::exp(z)-std::exp(-z))/(std::exp(z)+std::exp(-z)));}
	
std::vector<double>*	Math::tanh(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::tanh(*it_zs);
	return res;
}

double	Math::tanhPrime(double const z) {return (1-Math::tanh(z)*Math::tanh(z));}

std::vector<double>*	Math::tanhPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::tanhPrime(*it_zs);
	return res;
}

double	Math::step(double const z) {return (z > 0 ? 1 : 0);}
	
std::vector<double>*	Math::step(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::tanh(*it_zs);
	return res;
}

double	Math::stepPrime(double const z) {(void)z;return 0;}

std::vector<double>*	Math::stepPrime(std::vector<double> const & zs) {
	auto	res = new std::vector<double>(zs.size());
	auto	it_res = res->begin();
	
	for (auto it_zs = zs.begin(); it_zs != zs.end(); it_zs++, it_res++)
		*it_res = Math::tanhPrime(*it_zs);
	return res;
}

// COST FUNCTIONS
double	Math::costDerivative(double& output, double& expected) {return output - expected;}

std::vector<double>*	Math::costDerivative(std::vector<double> const & output, std::vector<double> const & expected) {
	auto	res = new std::vector<double>(output.size());
	auto	it_res = res->begin();
	
	for (auto it_o = output.begin(), it_e = expected.begin(); it_o != output.end() && it_e != expected.end(); it_o++, it_e++, it_res++)
		*it_res = *it_o - *it_e;
	return res;
}

double	Math::sqCostDerivative(double const & output, double const & expected) {return (std::pow((output - expected), 2));}

std::vector<double>*	Math::sqCostDerivative(std::vector<double> const & output, std::vector<double> const & expected) {
	auto	res = new std::vector<double>(output.size());
	auto	it_res = res->begin();
	
	for (auto it_o = output.begin(), it_e = expected.begin(); it_o != output.end() && it_e != expected.end(); it_o++, it_e++, it_res++)
		*it_res = Math::sqCostDerivative(*it_o, *it_e);
	return res;
}

// PRODUCT FUNCTIONS
double	Math::dotProduct(std::vector<double> const & v1, std::vector<double> const & v2) {
	double	res = 0.0;
	
	for (auto it_w = v2.begin(), it_i = v1.begin(); it_w != v2.end() && it_i != v1.end(); it_w++, it_i++)
		res = res + (*it_i) * (*it_w);
	return res;
}
	
std::vector<double>*	Math::hadamardProduct(std::vector<double> const & lhs, std::vector<double> const & rhs) {
	auto	res = new std::vector<double>(lhs.size());
	auto	it_res = res->begin();

	for (auto it_lhs = lhs.begin(), it_rhs = rhs.begin(); it_lhs != lhs.end() && it_rhs != rhs.end(); it_lhs++, it_rhs++, it_res++)
		*it_res = (*it_rhs) * (*it_lhs);
	return res;
}

std::vector<std::vector<double>>*	Math::outerProduct(std::vector<double> const & in, std::vector<double> const & base) {
	auto	res = new std::vector<std::vector<double>>(in.size(), std::vector<double>(base.size()));
	auto	it_res = res->begin();
	
	for (auto it_in = in.begin(); it_in != in.end(); it_in++, it_res++) {
		auto	it_inres = (*it_res).begin();
		for (auto it_b = base.begin(); it_b != base.end(); it_b++, it_inres++)
			*it_inres = (*it_in) * (*it_b);
	}
	return res;
}

// TRANSPOSITION FUNCTIONS
std::vector<std::vector<double>>*	Math::transpose1D(std::vector<double> const & base) {
	auto	res = new std::vector<std::vector<double>>(base.size(), std::vector<double>(1));
	auto	it_res = res->begin();
	
	for (auto it = base.begin(); it != base.end(); it++, it_res++)
		(*it_res).at(0) = *it;
	return res;
}

std::vector<std::vector<double>>*	Math::transpose2D(std::vector<std::vector<double>> const & base) {
	auto	res = new std::vector<std::vector<double>>(base.back().size(), std::vector<double>(base.size()));
	auto	it = base.begin();
	auto	it_in = (*it).begin();

	for (auto it_res = res->begin(); it_res != res->end(); it_res++) {
		for (auto i = (*it_res).begin(); i != (*it_res).end(); i++) {
			if (it_in == (*it).end()) {
				++it;
				it_in = (*it).begin();
			}
			*i = *it_in;
			++it_in;
		}
	}
	return res;
}

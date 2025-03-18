/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:09:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/18 13:44:08 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Math.namespace.hpp"
#include <cmath>

double	Math::sigmoid(double const z) {return (1.0/(1.0+std::exp(-z)));}
	
std::vector<double>*	Math::sigmoid(std::vector<double> const & zs) {
	auto	res = new std::vector<double>;
	
	for (auto it = zs.begin(); it != zs.end(); it++)
		res->push_back(Math::sigmoid(*it));
	return res;
}

double	Math::sigmoidPrime(double const z) {return (Math::sigmoid(z)*(1.0 - Math::sigmoid(z))) ;}

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


std::vector<std::vector<double>*>*	transpose1D(std::vector<double> const & base) {
	auto	res = new std::vector<std::vector<double>*>(base.size());
	int		i = 0;
	for (auto b : base) {
		res->at(i) = new std::vector<double>(1);
		res->back()->back() = b;
		++i;
	}
	return res;
}

std::vector<std::vector<double>*>*	transpose2D(std::vector<std::vector<double>*> const & base) {
	int const		baseRow = base.back()->size();
	int const		baseCol = base.size();
	auto			it = base.begin();
	auto			it_in = (*it)->begin();

	auto res = new std::vector<std::vector<double>*>(baseRow);
	for (auto r : *res) {
		r = new std::vector<double>(baseCol);
		for (auto i : *r) {
			if (it_in == (*it)->end()) {
				++it;
				it_in = (*it)->begin();
			}
			*it_in = i;
			++it_in;
		}
	}
	return res;
}

void	print(std::vector<std::vector<double>*> const & toPrint) {
	for (auto tP : toPrint) {
		std::cout<<"[";
		for (auto in_tp : *tP)
			std::cout << in_tp << ";";
		std::cout<<"];";
	}
	std::cout<<std::endl;
}

int main( void ) {
	std::vector<double>	myVec;

	myVec.push_back(1);
	myVec.push_back(2);
	myVec.push_back(3);
	myVec.push_back(4);
	auto new1D = transpose1D(myVec);
	print(*new1D);
	std::vector<std::vector<double>*>	my2D;
	for (int num = 0; num < 12;) {
		auto t = new std::vector<double>;
		my2D.push_back(t);
		for (int count=0; count < 4; count++) {
			my2D.back()->push_back(num);
			++num;
		}
	}
	auto new2D = transpose2D (my2D);
	print(*new2D);
	return 0;
}
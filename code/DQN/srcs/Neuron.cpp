/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Neuron.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:14:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/04 11:06:44 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Neuron.class.hpp"
#include <algorithm>

Neuron::Neuron(unsigned int const prevLayer) {
	std::random_device					rd;
	std::mt19937						gen(rd());
	double 								stddev = 1.0 / std::sqrt(prevLayer);
	std::normal_distribution<double> 	dist(0.0, stddev);	

	this->_weight = std::vector<double>(prevLayer);
	for (auto it = this->_weight.begin(); it != this->_weight.end(); it++)
		*it = dist(gen);
	this->_bias = dist(gen);
	this->_nabla_w = std::vector<double>(prevLayer);
	this->_deltaNabla_w = std::vector<double>(prevLayer);
	this->_nabla_b = 0.0;
	this->_deltaNabla_b = 0.0;
	return ;
}

double	Neuron::feedForward(std::vector<double> const & input, ptrFuncS actFunc) const {
	return (actFunc)(Math::dotProduct(input, this->_weight) + this->_bias);	
}

double	Neuron::affineTransformation(std::vector<double> const & input) const {
	return Math::dotProduct(input, this->_weight) + this->_bias;	
}

void	Neuron::copyNeuron(Neuron const & src) {
	if (src._weight.size() != this->_weight.size())
		throw std::exception();
	std::copy(src._weight.begin(), src._weight.end(),this->_weight.begin());
	this->_bias = src._bias;
	return ;
}

void	Neuron::updateWeight(double const eta, double const miniBatchSize) {
	for (auto it_w = this->_weight.begin(), it_nw = this->_nabla_w.begin(); it_w != this->_weight.end(); it_w++, it_nw++) {
		*it_w -= eta * (*it_nw / miniBatchSize);
		*it_nw = 0.0;
	}
	return ;
}

void	Neuron::updateNabla_w( void ) {
	for (auto it_nw = this->_nabla_w.begin(), it_dnw = this->_deltaNabla_w.begin(); it_nw != this->_nabla_w.end() && it_dnw != this->_deltaNabla_w.end(); it_nw++, it_dnw++) {
		*it_nw += *it_dnw;
		*it_dnw = 0.0;
	}
	return ;
}

void	Neuron::setDeltaNabla_w(std::vector<double> const & delta) {
	auto it_w = this->_deltaNabla_w.begin();
	for (auto it = delta.begin(); it != delta.end(); it++, it_w++)
		*it_w = *it;
	return ;
}

void	Neuron::updateBias(double const eta, double const miniBatchSize) {
	this->_bias -= eta * (this->_nabla_b / miniBatchSize);
	this->_nabla_b = 0.0;
	return ;
}

void	Neuron::updateNabla_b( void ) {
	this->_nabla_b += this->_deltaNabla_b;
	this->_deltaNabla_b = 0.0;
	return ;
}

void	Neuron::setDeltaNabla_b(double delta) {
	this->_deltaNabla_b = delta;
	return ;
}

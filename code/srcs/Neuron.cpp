/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Neuron.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:14:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/18 12:53:25 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Neuron.class.hpp"
#include "Math.namespace.hpp"

Neuron::Neuron(unsigned int const prevLayer) {
	std::random_device					rd;
	std::mt19937						gen(rd());
	std::normal_distribution<double>	dist(0.0, 1.0);
	
	for (unsigned int i = 0; i < prevLayer; i++)
		this->_weight.push_back(dist(gen));
	this->_bias = dist(gen);
	for (unsigned int i = 0; i < this->_weight.size(); i++)
		this->_nabla_w.push_back(0.0);
	this->_nabla_b = 0.0;
	return ;
}

double	Neuron::feedForward(std::vector<double> const & input) const {
	return Math::sigmoid(Math::dotProduct(input, this->_weight) + this->_bias);	
}

double	Neuron::perceptron(std::vector<double> const & input) const {
	return Math::dotProduct(input, this->_weight) + this->_bias;	
}

void	Neuron::updateWeight(double const eta, double const miniBatchSize) {
	for (auto it_w = this->_weight.begin(), it_nw = this->_nabla_w.begin(); it_w != this->_weight.end(); it_w++, it_nw++) {
		*it_w = *it_w - (eta / miniBatchSize) * *it_nw;
		*it_nw = 0.0;
	}
	return ;
}

void	Neuron::updateNabla_w( void ) {
	for (auto it_nw = this->_nabla_w.begin(), it_dnw = this->_deltaNabla_w.begin(); it_nw != this->_nabla_w.end() && it_dnw != this->_deltaNabla_w.end(); it_nw++, it_dnw++)
		*it_nw += *it_dnw;
	this->_deltaNabla_w.clear();
	return ;
}

void	Neuron::setDeltaNabla_w(std::vector<double> const & delta) {
	for (auto d : delta)
		this->_deltaNabla_w.push_back(d);
	return ;
}

void	Neuron::updateBias(double const eta, double const miniBatchSize) {
	this->_bias = this->_bias - (eta / miniBatchSize) * this->_nabla_b;
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

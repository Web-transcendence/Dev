/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:04:30 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 17:13:18 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include "Math.namespace.hpp"

Layer::Layer(int const n_neurons, int const n_weights, t_actFunc actFunc) {
	this->_neurons = std::vector<Neuron*>(n_neurons);
	for (auto it = this->_neurons.begin(); it != this->_neurons.end(); it++)
		*it = new Neuron(n_weights);
	this->_actFuncSingle = Math::actFuncS[actFunc];
	this->_actFuncVector = Math::actFuncV[actFunc];
	this->_primeActFuncSingle = Math::primeActFuncS[actFunc];
	this->_primeActFuncVector = Math::primeActFuncV[actFunc];
	return ;
}

Layer::~Layer( void ) {
	for (auto it = this->_neurons.begin(); it != this->_neurons.end(); it++)
		delete *it;
	return ;
}

std::vector<double>*	Layer::feedForward(std::vector<double> const & input) {
	std::vector<double>*	res = new std::vector<double>(this->_neurons.size());
	auto					it_res = res->begin();

	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++, it_res++)
		(*it_res) = (*it_n)->feedForward(input, this->_actFuncSingle);
	return res;
}

std::vector<double>*	Layer::affineTransformation(std::vector<double> const & input) {
	std::vector<double>*	res = new std::vector<double>(this->_neurons.size());
	auto					it_res = res->begin();

	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++, it_res++)
		(*it_res) = (*it_n)->affineTransformation(input);
	return res;
}

double	Layer::callActFunc(double const input) {
	return this->_actFuncSingle(input);
}

std::vector<double>*	Layer::callActFunc(std::vector<double> const & input) {
	return this->_actFuncVector(input);
}

double	Layer::callPrimeActFunc(double const input) {
	return this->_primeActFuncSingle(input);
}

std::vector<double>*	Layer::callPrimeActFunc(std::vector<double> const & input) {
	return this->_primeActFuncVector(input);
}

void	Layer::copyLayer(Layer const & src) {
	auto	it_t = this->_neurons.begin();
	auto	it_c = src._neurons.begin();

	for (;it_t != this->_neurons.end(); it_t++, it_c++)
		(*it_t)->copyNeuron(*(*it_c));
	return ;
}

void	Layer::updateWeight(double const eta, double const miniBatchSize) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateWeight(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_w( void ) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateNabla_w();
	return ;
}

void	Layer::setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation) {
	auto	product = Math::outerProduct(delta, activation);
	auto	it_p = product->begin();
	
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++, it_p++)
		(*it_n)->setDeltaNabla_w(*it_p);
	delete product;
	return ;
}

void	Layer::updateBias(double const eta, double const miniBatchSize) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateBias(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_b( void ) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateNabla_b();
	return ;
}

void	Layer::setDeltaNabla_b(std::vector<double> const & delta) {
	auto	it_n = this->_neurons.begin();
	auto	it_d = delta.begin();
	for (;it_d != delta.end() && it_n != this->_neurons.end(); it_d++, it_n++) {
		(*it_n)->setDeltaNabla_b(*it_d);
	}
	return ;
}

std::vector<double>*	Layer::calcDelta(std::vector<double> const & delta, std::vector<double> const & sp) {
	auto	merged = std::vector<std::vector<double>>(this->_neurons.size(), std::vector<double>(this->_neurons.at(0)->_weight.size()));
	auto	it = merged.begin();

	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++) {
		auto	it_w = (*it).begin();
		for (auto it_we = (*it_n)->_weight.begin(); it_we != (*it_n)->_weight.end(); it_we++) {
			*it_w = *it_we;
			++it_w;
		}
		++it;
	}
	auto	transposed = Math::transpose2D(merged);
	auto	temp = std::vector<double>(this->_neurons.at(0)->_weight.size());
	auto	it_t = temp.begin();
	for (auto it_tr = transposed->begin(); it_tr != transposed->end(); it_tr++, it_t++) {
		(*it_t) = Math::dotProduct(*it_tr, delta);
	}
	delete transposed;
	return Math::hadamardProduct(temp, sp);	
}

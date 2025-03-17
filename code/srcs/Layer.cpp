/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:04:30 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/17 14:32:11 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include "Math.namespace.hpp"

Layer::Layer(int const n_neurons, int const n_weights) {
	for (int i = 0;i < n_neurons; i++)
		this->_neurons.push_back(new Neuron(n_weights));	
}

Layer::~Layer( void ) {
	for (auto n : this->_neurons)
		delete n;
	return ;
}

std::vector<double>*	Layer::feedForward(std::vector<double> const & input) {
	std::vector<double>*	res = new std::vector<double>;

	for (auto n : this->_neurons)
		res->push_back(n->feedForward(input));
	return res;
}

std::vector<double>*	Layer::perceptron(std::vector<double> const & input) {
	std::vector<double>*	res = new std::vector<double>;

	for (auto n : this->_neurons) {
		res->push_back(n->perceptron(input));
	}
	return res;
}

void	Layer::updateWeight(double const eta, double const miniBatchSize) {
	for (auto n : this->_neurons)
		n->updateWeight(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_w( void ) {
	for (auto n : this->_neurons)
		n->updateNabla_w();
	return ;
}

void	Layer::setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation) {
	auto	it_delta = delta.begin();
	
	for (auto n : this->_neurons) {
		std::vector<double> temp;
		for (auto a : activation)
			temp.push_back(*it_delta * a);
		n->setDeltaNabla_w(temp);
		temp.clear();
		++it_delta;
	}
	return ;
}

void	Layer::updateBias(double const eta, double const miniBatchSize) {
	for (auto n : this->_neurons)
		n->updateBias(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_b( void ) {
	for (auto n : this->_neurons)
		n->updateNabla_b();
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
	auto		transposed = new std::vector<std::vector<double>*>;
	int const	row = this->_neurons.back()->_weight.size();
	int const	col	= this->_neurons.size();

	for (int i = 0; i < row; i++)
		transposed->push_back(new std::vector<double>(col));
	auto it_t = transposed->begin();
	auto it_in = (*it_t)->begin();
	for (auto n : this->_neurons) {
		for (auto w : n->_weight) {
			if (it_in == (*it_t)->end()) {
				++it_t;
				it_in = (*it_t)->begin();
			}
			*it_in = w;
			++it_in;
		}
	}
	// Math::printdebug(delta, "delta");
	// Math::printdebug(sp, "sp");
	auto	temp = new std::vector<double>;
	for (auto t : *transposed) {
		temp->push_back(Math::dotProduct(*t, delta));
		delete t;
	}
	delete transposed;
	auto res = Math::hadamardProduct(*temp, sp);
	delete temp;
	return res;
}

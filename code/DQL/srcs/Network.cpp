/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 17:03:11 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include <algorithm>
#include <iostream>

Network::Network(std::vector<unsigned int>sizes) : _num_layers(sizes.size()), _sizes(sizes) {
	this->_layers = std::vector<Layer*>(_num_layers - 1);
	for (int i_layers = 1; i_layers < this->_num_layers; i_layers++) {
		this->_layers.at(i_layers - 1) = new Layer(sizes[i_layers], sizes[i_layers - 1]);
	}
	return ;
}

void    Network::SDG(std::vector<double> & input, std::vector<double> & expected, double const eta) {
	this->backprop_reLu(input, expected);
	this->updateNabla_b();
	this->updateNabla_w();
	this->updateBias(eta, 1.);
	this->updateWeight(eta, 1.);
}

void	Network::backprop_reLu(std::vector<double>& input, std::vector<double>& expectedOutput) {
	std::vector<std::vector<double>*>	activations(this->_layers.size() + 1);
	auto								it_a = activations.begin();
	std::vector<std::vector<double>*>	zs(this->_layers.size());
	auto								it_z = zs.begin();
	unsigned int const					lSize = this->_layers.size();

	*it_a = &input;
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++, it_z++) {
		*it_z = (*it_l)->affineTransformation(*(*it_a));
		it_a++;
		*it_a = Math::reLu(**it_z);
	}
	auto	cd = Math::cost_derivative(*activations.back(), expectedOutput);
	auto	sp = Math::reLuPrime(*zs.back());
	auto	delta = Math::hadamardProduct(*cd, *sp);
	this->_layers.back()->setDeltaNabla_b(*delta);
	this->_layers.back()->setDeltaNabla_w(*delta, *activations.at(activations.size()-2));
	delete cd;
	delete sp;
	for (unsigned int i_l = 2; i_l <= lSize; i_l++) {
		sp = Math::reLuPrime(*zs.at(lSize- i_l));
		auto nDelta = this->_layers.at(lSize-i_l+1)->calcDelta(*delta, *sp);
		delete delta;
		delta = nDelta;
		this->_layers.at(lSize-i_l)->setDeltaNabla_b(*delta);
		this->_layers.at(lSize-i_l)->setDeltaNabla_w(*delta, *activations.at(activations.size()-i_l-1));
		delete sp;
	}
	for (auto i : zs)
		delete i;
	activations.front() = NULL;
	for (auto i : activations) {
		if (i)
			delete i;
	}
}

void	Network::backprop(std::vector<double>& input, std::vector<double>& expectedOutput) {
	std::vector<std::vector<double>*>	activations(this->_layers.size() + 1);
	auto								it_a = activations.begin();
	std::vector<std::vector<double>*>	zs(this->_layers.size());
	auto								it_z = zs.begin();
	unsigned int const					lSize = this->_layers.size();

	*it_a = &input;
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++, it_z++) {
		*it_z = (*it_l)->affineTransformation(*(*it_a));
		it_a++;
		*it_a = Math::sigmoid(**it_z);
	}
	auto	cd = Math::cost_derivative(*activations.back(), expectedOutput);
	auto	sp = Math::sigmoidPrime(*zs.back());
	auto	delta = Math::hadamardProduct(*cd, *sp);
	this->_layers.back()->setDeltaNabla_b(*delta);
	this->_layers.back()->setDeltaNabla_w(*delta, *activations.at(activations.size()-2));
	delete cd;
	delete sp;
	for (unsigned int i_l = 2; i_l <= lSize; i_l++) {
		sp = Math::sigmoidPrime(*zs.at(lSize- i_l));
		auto nDelta = this->_layers.at(lSize-i_l+1)->calcDelta(*delta, *sp);
		delete delta;
		delta = nDelta;
		this->_layers.at(lSize-i_l)->setDeltaNabla_b(*delta);
		this->_layers.at(lSize-i_l)->setDeltaNabla_w(*delta, *activations.at(activations.size()-i_l-1));
		delete sp;
	}
	for (auto i : zs)
		delete i;
	activations.front() = NULL;
	for (auto i : activations) {
		if (i)
			delete i;
	}
}

std::vector<double>*	Network::feedForwardReLu(std::vector<double> const & input) {
	auto	it = this->_layers.begin();
	auto	activation = new std::vector<double>(input);
	
	for (; it != this->_layers.end(); it++) {
		auto temp = activation;
		activation = (*it)->feedForwardReLu(*activation);
		delete temp;
	}
	return activation;
}

std::vector<double>*	Network::feedForwardSigmoid(std::vector<double> const & input) {
	auto	it = this->_layers.begin();
	auto	activation = new std::vector<double>(input);
	
	for (; it != this->_layers.end(); it++) {
		auto temp = activation;
		activation = (*it)->feedForwardSigmoid(*activation);
		delete temp;
	}
	return activation;
}

int     Network::evaluate(std::vector<t_tuple*>& test_data) {
	int	correct = 0;

	for (auto it_td = test_data.begin(); it_td != test_data.end(); it_td++) {
		auto output = this->feedForwardSigmoid((*it_td)->input);
		int numOutput = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		if ((*it_td)->real == numOutput)
			++correct;
		delete output;
	}
	return correct;
}

void	Network::updateWeight(double const eta, double const miniBatchSize) {
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++)
		(*it_l)->updateWeight(eta, miniBatchSize);
	return ;
}

void	Network::updateNabla_w( void ) {
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++)
		(*it_l)->updateNabla_w();
	return ;
}

void	Network::updateBias(double const eta, double const miniBatchSize) {
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++)
		(*it_l)->updateBias(eta, miniBatchSize);
	return ;
}

void	Network::updateNabla_b( void ) {
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++)
		(*it_l)->updateNabla_b();
	return ;
}	

void    Network::myShuffle(std::vector<t_tuple*>& myVector) {
	std::random_device  rd;
	std::mt19937        g(rd());
	
	std::shuffle(myVector.begin(), myVector.end(), g);
	return ;
}


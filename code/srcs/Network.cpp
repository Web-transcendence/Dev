/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/16 14:22:05 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include "Math.namespace.hpp"
#include <algorithm>
#include <iostream>

Network::Network(std::vector<unsigned int>sizes) : _num_layers(sizes.size()), _sizes(sizes) {
	for (int i_layers = 1; i_layers < this->_num_layers; i_layers++) {
		this->_layers.push_back(new Layer(sizes[i_layers], sizes[i_layers - 1]));
	}
	return ;
}

void    Network::SDG(std::vector<t_tuple*> trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data) {
	int	n_test;
	
	if (test_data)
		n_test = test_data->size();
	for (int i = 0; i < epoch; i++) {
		myShuffle(trainingData);
		std::vector<std::vector<t_tuple*>*>  mini_batches;
		for (auto it_td = trainingData.begin(); it_td != trainingData.end();) {
			mini_batches.push_back(new std::vector<t_tuple*>(miniBatchSize));
			for (int i_mb = 0; i_mb < miniBatchSize && it_td != trainingData.end(); i_mb++, it_td++)
				mini_batches.back()->push_back(*it_td);
		}
		for (auto it_mb = mini_batches.begin(); it_mb != mini_batches.end(); it_mb++)
			this->updateMiniBatch(**it_mb, eta);	
		if (test_data) {
			std::cout<<"Epoch "<<i<<": "<<this->evaluate(*test_data)<<" / "<<n_test<<std::endl;
		}
		else
			std::cout<<"Epoch "<<i<<": complete"<<std::endl;
	}
}

void	Network::updateMiniBatch(std::vector<t_tuple*>& miniBatch, double const eta) {
	for (auto it_mb = miniBatch.begin(); it_mb != miniBatch.end(); it_mb++) {
		this->backprop((*it_mb)->input, (*it_mb)->expectedOutput);
		this->updateNabla_b();
		this->updateNabla_w();
	}
	this->updateBias(eta, static_cast<double>(miniBatch.size()));
	this->updateWeight(eta, static_cast<double>(miniBatch.size()));
	return ;
}

void	Network::backprop(std::vector<double>& input, std::vector<double>& expectedOutput) {
	std::vector<double>*				activation = &input;
	std::vector<std::vector<double>*>	activations;
	std::vector<double>*				z;
	std::vector<std::vector<double>*>	zs;
	int									os = 1;

	activations.push_back(activation);
	for (auto l : this->_layers) {
		z = l->perceptron(*activation);
		zs.push_back(z);
		activation = Math::sigmoid(*z);
		activations.push_back(activation);
	}
	auto	cd = Math::cost_derivative(*activations.back(), expectedOutput);
	auto	sp = Math::sigmoidPrime(*zs.back());			
	auto	delta = Math::hadamardProduct(*cd, *sp);
	this->_layers.back()->setDeltaNabla_b(*delta);
	this->_layers.back()->setDeltaNabla_w(*delta, **(activations.end()-2));
	delete cd;
	delete sp;
	for (auto it = this->_layers.rbegin() - os; it != this->_layers.rend(); it++, os++) {
		z = *zs.rbegin() - os;
		sp = Math::sigmoidPrime(*z);
		auto nDelta = (*(it+1))->calcDelta(*delta, *sp);
		delete delta;
		delta = nDelta;
		(*it)->setDeltaNabla_b(*delta);
		(*it)->setDeltaNabla_w(*delta, **(activations.end()-os-1));
		delete sp;
	}
}

std::vector<double>*	Network::feedForward(std::vector<double> const & input) {
	auto	it = this->_layers.begin();
	auto	activation = new std::vector<double>(input);
	
	for (; it != this->_layers.end(); it++) {
		auto temp = activation;
		activation = (*it)->feedForward(*activation);
		delete temp;
	}
	return activation;
}

int     Network::evaluate(std::vector<t_tuple*>& test_data) {
	auto	netResult = new std::vector<std::vector<double>*>;
	auto 	it_td = test_data.begin();
	int		res = 0;

	for (; it_td != test_data.begin(); it_td++)
		netResult->push_back(this->feedForward((*it_td)->input));
	it_td = test_data.begin();
	for (auto net : *netResult) {
		if (*net == (*it_td)->expectedOutput)
			++res;
		++it_td;
	}
	return res;
}

void	Network::updateWeight(double const eta, double const miniBatchSize) {
	for (auto l : this->_layers)
		l->updateWeight(eta, miniBatchSize);
	return ;
}

void	Network::updateNabla_w( void ) {
	for (auto l : this->_layers)
		l->updateNabla_w();
	return ;
}

void	Network::updateBias(double const eta, double const miniBatchSize) {
	for (auto l : this->_layers)
		l->updateBias(eta, miniBatchSize);
	return ;
}

void	Network::updateNabla_b( void ) {
	for (auto l : this->_layers)
		l->updateNabla_b();
	return ;
}	

void    Network::myShuffle(std::vector<t_tuple*>& myVector) {
	std::random_device  rd;
	std::mt19937        g(rd());
	
	std::shuffle(myVector.begin(), myVector.end(), g);
	return ;
}


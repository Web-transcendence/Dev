/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/31 12:56:14 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include <algorithm>
#include <iostream>

Network::Network(std::vector<unsigned int>sizes, t_actFunc actHiddenFunc, t_actFunc actOutputFunc) : _num_layers(sizes.size()), _sizes(sizes) {
	int	i_layers;

	this->_layers = std::vector<Layer*>(_num_layers - 1);
	for (i_layers = 1; i_layers < this->_num_layers - 1; i_layers++)
		this->_layers.at(i_layers - 1) = new Layer(sizes[i_layers], sizes[i_layers - 1], actHiddenFunc);
	this->_layers.at(i_layers - 1) = new Layer(sizes[i_layers], sizes[i_layers - 1], actOutputFunc);
	return ;
}

void    Network::SDG(t_tuple* trainingData, double const eta) {
	this->backprop(trainingData->input, trainingData->expectedOutput);
	this->updateNabla_b();
	this->updateNabla_w();
	this->updateWeight(eta, 1.);
	this->updateBias(eta, 1.);
	return ;
}

void    Network::SDG(std::vector<t_tuple*>& trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data) {
	int	n_test;
	
	if (test_data)
		n_test = test_data->size();
	for (int i = 0; i < epoch; i++) {
		std::cout<<"Starting Epoch "<<i<<std::endl;
		myShuffle(trainingData);
		auto mini_batches = std::vector<std::vector<t_tuple*>>((trainingData.size() / miniBatchSize), std::vector<t_tuple*>(miniBatchSize));
		auto it_m = mini_batches.begin();
		for (auto it_td = trainingData.begin(); it_td != trainingData.end(); it_m++) {
			for (auto it_mba = (*it_m).begin(); it_mba != (*it_m).end() && it_td != trainingData.end(); it_mba++, it_td++)
				*it_mba = *it_td;
		}
		int	display = 0;
		for (auto it_mb = mini_batches.begin(); it_mb != mini_batches.end(); it_mb++, display++)
			this->updateMiniBatch(*it_mb, eta);
		if (test_data) {
			std::cout<<std::endl<<"Epoch "<<i<<": "<<this->evaluate(*test_data)<<" / "<<n_test<<std::endl;
		}
		else
			std::cout<<std::endl<<"Epoch "<<i<<": complete"<<std::endl;
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
		*it_a = (*it_l)->callActFunc(**it_z);
	}
	auto	cd = Math::costDerivative(*activations.back(), expectedOutput);
	auto	sp = this->_layers.back()->callActFunc(*zs.back());
	auto	delta = Math::hadamardProduct(*cd, *sp);
	this->_layers.back()->setDeltaNabla_b(*delta);
	this->_layers.back()->setDeltaNabla_w(*delta, *activations.at(activations.size()-2));
	delete cd;
	delete sp;
	for (unsigned int i_l = 2; i_l <= lSize; i_l++) {
		sp = this->_layers.at(lSize-i_l+1)->callActFunc(*zs.at(lSize- i_l));
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


/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/14 16:35:18 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Math.namespace.hpp"
#include <iostream>

Network::Network(std::vector<unsigned int>sizes) : _num_layers(sizes.size()), _sizes(sizes) {
	for (int i_layers = 1; i_layers < this->_num_layers; i_layers++) {
		this->_layers.push_back(new Layer(sizes[i_layers], sizes[i_layers - 1]));
	}
	return ;
}

void    Network::SDG(std::vector<t_tuple*> trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data) {
	int const n = trainingData.size();
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
	auto	nabla_b = this->shapeBiases();
	auto	nabla_w = this->shapeWeights();

	for (auto it_mb = miniBatch.begin(); it_mb != miniBatch.end(); it_mb++) {
		auto	temp = this->backprop((*it_mb)->input, (*it_mb)->expectedOutput);
		auto	deltaNabla_b = temp[0];
		auto	deltaNabla_w = temp[1];
	}
	return ;
}

std::vector<Layer*>*	Network::backprop(std::vector<double>& input, std::vector<double>& expectedOutput) {
	
	std::vector<double>*					activation = &input;
	std::vector<std::vector<double>*>		activations;
	std::vector<double>*					z;
	std::vector<std::vector<double>*>		zs;
	
	nablaBW[0]->push_back(new std::vector<vecDouble*>);
	activations.push_back(activation);
	for (auto it = this->_layers.begin(); it != this->_layers.end(); it++) {
		for (auto it_n = (*it)->begin(); it_n != (*it)->end(); it_n++) {
			z = new std::vector<double>((*it_n)->_size);
			z->push_back(Math::sumWeighted(*activation, (*it_n)->_weight) + (*it_n)->_bias);
		}
		zs.push_back(z);
		activation = Math::sigmoid(*z);
		activations.push_back(activation);
	}
	auto delta = Math::multVec(*(Math::cost_derivative(*activations.back(), expectedOutput)), *(Math::sigmoidPrime(*zs.back())));
	nablaBW[0]->back()->push_back(delta);
	nablaBW[1]->push_back(Math::matricialMult(*delta, **(activations.end() - 2))); 
	for (int i = 2; i < this->_num_layers; i++) {
		z = zs.back() - i;
		delta =  Math::matricialMult(*delta, **(activations.end() - 2));
	}
	return ;
}

std::vector<std::vector<vecDouble*>*>*	Network::shapeBiases( void ) {
	auto	shaped = new (std::vector<std::vector<vecDouble*>*>);

	shaped->push_back(new std::vector<vecDouble*>);
	for (auto it_layers = this->_layers.begin() + 1; it_layers != this->_layers.end(); it_layers++) {
		shaped->back()->push_back(new vecDouble((*it_layers)->size()));
		for (auto it_neurons = (*it_layers)->begin(); it_neurons != (*it_layers)->end(); it_neurons++)
			shaped->back()->back()->push_back(0.0);
	}
	return shaped;
}

std::vector<std::vector<vecDouble*>*>*	Network::shapeWeights( void ) {
	auto	shaped = new (std::vector<std::vector<vecDouble*>*>);

	for (auto it_layers = this->_layers.begin() + 1; it_layers != this->_layers.end(); it_layers++) {
		shaped->push_back(new std::vector<vecDouble*>);
		for (auto it_neurons = (*it_layers)->begin(); it_neurons != (*it_layers)->end(); it_neurons++) {
			shaped->back()->push_back(new vecDouble);
			for (auto it_weights = (*it_neurons)->_weight.begin(); it_weights != (*it_neurons)->_weight.end(); it_weights++)
				shaped->back()->back()->push_back(0.0);
		}
	}
	return shaped;
}

void    Network::myShuffle(std::vector<t_tuple*>& myVector) {
	std::random_device  rd;
	std::mt19937        g(rd());
	
	std::shuffle(myVector.begin(), myVector.end(), g);
	return ;
}


/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/12 16:53:47 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "network.class.hpp"
#include <iostream>

Network::Network(std::vector<unsigned int>sizes) : _num_layers(sizes.size()), _sizes(sizes) {

	for (int i_layers = 0; i_layers < this->_num_layers; i_layers++) {
		this->_layers.push_back(new std::vector<Neuron*>);
		if (i_layers == 0) {
			for (int i_neurons = 0; i_neurons < this->_sizes[i_layers]; i_neurons++)
				this->_layers[i_layers]->push_back(new InputNeuron());
		}
		else if (i_layers == this->_num_layers - 1) {
			for (int i_neurons = 0; i_neurons < this->_sizes[i_layers]; i_neurons++)
				this->_layers[i_layers]->push_back(new OutputNeuron());
		}
		else {
			for (int i_neurons = 0; i_neurons < this->_sizes[i_layers]; i_neurons++)
				this->_layers[i_layers]->push_back(new HidenNeuron(this->_sizes[i_layers - 1]));
		}
	}
	return ;
}

void    Network::SDG(std::vector<t_tuple*>& trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data) {
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
	std::vector<t_tuple*>	nabla_b;
	std::vector<t_tuple*>	nabla_w;
	return ;
}

void    Network::myShuffle(std::vector<t_tuple*>& myVector) {
	std::random_device  rd;
	std::mt19937        g(rd());
	
	std::shuffle(myVector.begin(), myVector.end(), g);
	return ;
}
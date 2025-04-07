/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 16:17:49 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/07 13:19:07 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

Network::Network(std::vector<unsigned int>sizes, t_actFunc actHiddenFunc, t_actFunc actOutputFunc) : _num_layers(sizes.size()), _sizes(sizes) {
	int	i_layers;

	this->_layers = std::vector<Layer*>(_num_layers - 1);
	for (i_layers = 1; i_layers < this->_num_layers - 1; i_layers++)
		this->_layers.at(i_layers - 1) = new Layer(sizes[i_layers], sizes[i_layers - 1], actHiddenFunc);
	this->_layers.at(i_layers - 1) = new Layer(sizes[i_layers], sizes[i_layers - 1], actOutputFunc);
	return ;
}

Network::~Network( void ) {
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++)
		delete *it_l;
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
		for (auto it_mb = mini_batches.begin(); it_mb != mini_batches.end(); it_mb++, display++) {
			this->updateMiniBatch(*it_mb, eta);
			Network::displayProgress(display, mini_batches.size());
		}
		if (test_data) {
			std::cout<<std::endl<<"Epoch "<<i<<": "<<this->evaluate(*test_data)<<" / "<<n_test<<std::endl;
		}
		else
			std::cout<<std::endl<<"Epoch "<<i<<": complete"<<std::endl;
	}
}

void	Network::copyNetwork(Network const & src) {
	auto	it_t = this->_layers.begin();
	auto	it_c = src._layers.begin();

	for (;it_t != this->_layers.end(); it_t++, it_c++)
		(*it_t)->copyLayer(*(*it_c));
	return ;
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
	auto	sp = this->_layers.back()->callPrimeActFunc(*zs.back());
	auto	delta = Math::hadamardProduct(*cd, *sp);
	this->_layers.back()->setDeltaNabla_b(*delta);
	this->_layers.back()->setDeltaNabla_w(*delta, *activations.at(activations.size()-2));
	delete cd;
	delete sp;
	for (unsigned int i_l = 2; i_l <= lSize; i_l++) {
		sp = this->_layers.at(lSize-i_l+1)->callPrimeActFunc(*zs.at(lSize- i_l));
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
	int	correct = 0;

	for (auto it_td = test_data.begin(); it_td != test_data.end(); it_td++) {
		auto output = this->feedForward((*it_td)->input);
		int numOutput = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		int	numExpected = std::distance((*it_td)->expectedOutput.begin(), std::max_element((*it_td)->expectedOutput.begin(), (*it_td)->expectedOutput.end()));
		if (numExpected == numOutput)
			++correct;
		delete output;
	}
	return correct;
}

void	Network::printNetworkToJson(std::string const & outputFile) {
	std::ofstream		ofs(outputFile.c_str());
	std::stringstream	mySs;

	mySs<<"{"<<std::endl<<"\"Network\": {"<<std::endl;
	unsigned int	countL = 0;
	for (auto it_l = this->_layers.begin(); it_l != this->_layers.end(); it_l++) {
		mySs<<"	\"Layer "<<countL<<"\""<<": {"<<std::endl;
		unsigned int	countN = 0;
		for (auto it_n = (*it_l)->_neurons.begin(); it_n != (*it_l)->_neurons.end(); it_n++) {
			mySs<<"		\"neuron "<<countN<<"\""<<": {"<<std::endl;
			mySs<<"			\"w\": [";
			for (auto it_w = (*it_n)->_weight.begin(); it_w != (*it_n)->_weight.end(); it_w++) {
				mySs<<*it_w;
				if (it_w + 1 != (*it_n)->_weight.end())
					mySs<<",";	
			}
			mySs<<"],"<<std::endl;
			mySs<<"			\"b\":"<<(*it_n)->_bias<<std::endl;
			mySs<<"		}";
			++countN;
			if (it_n + 1 != (*it_l)->_neurons.end())
				mySs<<",";
			mySs<<std::endl;
		}
		mySs<<"	}";
		if (it_l + 1 != this->_layers.end())
			mySs<<",";
		mySs<<std::endl;
		++countL;
	}
	mySs<<"}"<<std::endl;
	mySs<<"}"<<std::endl;
	ofs << mySs.str();
	return ;
}

void	Network::displayProgress(int current, int max) {
    int width = 20;
    int progress = (current * width) / max;

    std::cout << "\r" << (current * 100) / max << "% [";
    for (int i = 0; i < width; i++) {
        std::cout << (i < progress ? '#' : '.');
    }
    std::cout << "]" << std::flush;
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


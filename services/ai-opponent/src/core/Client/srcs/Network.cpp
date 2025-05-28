 /* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:45:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/05 16:10:03 by tmouche          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "json.hpp"

#include <fstream>
#include <sstream>

#include <iostream>

Network::Network(std::string const & inFile) {
	std::string	dataStr;
	
	std::ifstream ifs(inFile.c_str());
	if (!ifs)
		throw std::exception();
	std::getline(ifs, dataStr, '\0');
	ifs.close();
	auto data = nlohmann::json::parse(dataStr)["Network"];

	unsigned int	iWeight = 0;
	unsigned int	iBias = 0;

	this->_sizes[0] = N_INPUT;
	this->_sizes[L_GLOBAL - 1] = N_OUTPUT;
	for (int i = 0; i < L_HIDDEN; i++) this->_sizes[i + 1] = N_HIDDEN;
	for (unsigned int wLayer = 1; wLayer < L_GLOBAL; wLayer++) {
		std::stringstream	ssLayer;
		ssLayer << "Layer " << wLayer - 1;
		for (unsigned int wNeuron = 0; wNeuron < this->_sizes[wLayer]; wNeuron++, iBias++) {
			std::stringstream	ssNeuron;
			ssNeuron << "neuron " << wNeuron;
			for (unsigned int wWeights = 0; wWeights < this->_sizes[wLayer - 1]; wWeights++, iWeight++) {
				std::stringstream	w;
				w << data[ssLayer.str()][ssNeuron.str()]["w"][wWeights];
				w >> this->_weights[iWeight];
			}
			std::stringstream	b;
			b << data[ssLayer.str()][ssNeuron.str()]["b"];
			b >> this->_biais[iBias];
		}
	}
}

inline	float	Network::sigmoid(float const z) {return (1.0/(1.0+std::exp(-z)));}

void	Network::sigmoid(float const * zs, float * res, unsigned int const size) {
	for (unsigned int sharedIdx = 0; sharedIdx < size; sharedIdx++) {
		res[sharedIdx] = this->sigmoid(zs[sharedIdx]);
	}
}

float	Network::dotProduct(float const * v1, float const * v2, unsigned int const size) {
	float	res = 0.0;
	
	for (unsigned int sharedIdx = 0; sharedIdx < size; sharedIdx++) {
		res += (v1[sharedIdx] * v2[sharedIdx]);
	}
	return res;
}


std::vector<float>	Network::feedForward(std::vector<float> const & input) {
	unsigned int	weightIdx = 0;
	unsigned int	biaiIdx = 0;
	unsigned int 	layerIdx = 0;
		
	for (unsigned int neuronIdx = 0; neuronIdx < this->_sizes[layerIdx + 1]; neuronIdx++, biaiIdx++, weightIdx += this->_sizes[layerIdx]) {
		_placeHidden[layerIdx % 2][neuronIdx] = this->dotProduct(input.data(), &this->_weights[weightIdx], this->_sizes[layerIdx]) + this->_biais[biaiIdx];
		_placeHidden[layerIdx % 2][neuronIdx] = this->sigmoid(_placeHidden[layerIdx % 2][neuronIdx]);
	}
	layerIdx++;
	for (; layerIdx < L_GLOBAL; layerIdx++) {
		for (unsigned int neuronIdx = 0; neuronIdx < this->_sizes[layerIdx]; neuronIdx++, biaiIdx++, weightIdx += this->_sizes[layerIdx]) {
			_placeHidden[layerIdx % 2][neuronIdx] = this->dotProduct(_placeHidden[(layerIdx - 1) % 2], &this->_weights[weightIdx], this->_sizes[layerIdx]) + this->_biais[biaiIdx];
			_placeHidden[layerIdx % 2][neuronIdx] = this->sigmoid(_placeHidden[layerIdx % 2][neuronIdx]);
		}
	}
	return std::vector<float>(this->_placeHidden[layerIdx % 2], this->_placeHidden[layerIdx % 2] + N_OUTPUT);
}
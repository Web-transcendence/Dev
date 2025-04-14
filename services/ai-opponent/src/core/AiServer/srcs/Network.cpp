/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:45:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/14 00:32:08 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "json/json.hpp"

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
	std::vector<unsigned int>	sizes(N_LAYER_HIDDEN + 2);
	sizes.at(0) = N_NEURON_INPUT;
	for (unsigned int i = 1; i < N_LAYER_HIDDEN + 1; i++)
		sizes.at(i) = N_NEURON_HIDDEN;
	sizes.at(N_LAYER_HIDDEN + 1) = N_NEURON_OUTPUT;
	this->_weights = std::vector<std::vector<std::vector<double>>>(N_LAYER_HIDDEN + 1);
	this->_biaises = std::vector<std::vector<double>>(N_LAYER_HIDDEN + 1);
	auto				it_wl = this->_weights.begin();
	auto				it_bl = this->_biaises.begin();
	unsigned int		whichLayer = 1;
	unsigned int		countLayer = 0;
	for (; it_wl != this->_weights.end(); it_wl++, it_bl++, whichLayer++, countLayer++) {
		std::stringstream	ssLayer;
		unsigned int	countNeuron = 0;
		*it_wl = std::vector<std::vector<double>>(sizes.at(whichLayer));
		*it_bl = std::vector<double>(sizes.at(whichLayer));
		ssLayer << "Layer " << countLayer;
		auto	it_wn = (*it_wl).begin();
		auto	it_bn = (*it_bl).begin();
		for (;it_wn != (*it_wl).end(); it_wn++, it_bn++, countNeuron++) {
			std::stringstream	ssNeuron;
			ssNeuron << "neuron " << countNeuron;
			*it_wn = std::vector<double>(sizes.at(whichLayer - 1));
			unsigned int	countWeight = 0;
			for (auto it_ww = (*it_wn).begin(); it_ww != (*it_wn).end(); it_ww++, countWeight++) {
				std::stringstream	w;
				w << data[ssLayer.str()][ssNeuron.str()]["w"][countWeight];
				w >> *it_ww;
			}
			std::stringstream	b;
			b << data[ssLayer.str()][ssNeuron.str()]["b"];
			b >> *it_bn;
		}
	}
	return ;
}

std::vector<double>	Network::feedForward(std::vector<double> const & input) {
	std::vector<double>	layerOutput(input);
	
	for (auto it_l = this->_weights.begin();it_l != this->_weights.end(); it_l++) {
		layerOutput.resize((*it_l).size());
		auto	it_lo = layerOutput.begin();
		for (auto it_n = (*it_l).begin(); it_n != (*it_l).end(); it_n++, it_lo++) {
			std::vector<double>	input(layerOutput);
			*it_lo = Math::leakyReLu(Math::dotProduct(input, *it_n));
		}
	}
	layerOutput.resize(N_NEURON_OUTPUT);
	return layerOutput;
}

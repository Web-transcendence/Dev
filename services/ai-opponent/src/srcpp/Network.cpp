/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:45:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/10 09:21:02 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "json.hpp"

#include <fstream>

Network::Network(std::string const & inFile) {
	std::string	dataStr;
	
	std::ifstream ifs(inFile.c_str());
	if (!ifs)
		throw std::exception();
	std::getline(ifs, dataStr, '\0');
	auto data = nlohmann::json::parse(dataStr);
	std::vector<unsigned int>	sizes(N_LAYER_HIDDEN + 2);
	sizes.at(0) = N_NEURON_INPUT;
	for (unsigned int i = 1; i < N_LAYER_HIDDEN; i++)
		sizes.at(i) = N_NEURON_HIDDEN;
	sizes.at(N_LAYER_HIDDEN + 1) = N_NEURON_OUTPUT;
	this->_weights = std::vector<std::vector<std::vector<double>>>(N_LAYER_HIDDEN + 1);
	this->_biaises = std::vector<std::vector<double>>(N_LAYER_HIDDEN + 1);
	auto			it_wl = this->_weights.begin();
	auto			it_bl = this->_biaises.begin();
	unsigned int	whichLayer = 1;
	for (; it_wl != this->_weights.end(); it_wl++, it_bl++, whichLayer++) {
		*it_wl = std::vector<std::vector<double>>(sizes.at(whichLayer));
		*it_bl = std::vector<double>(sizes.at(whichLayer));
		auto	it_wn = (*it_wl).begin();
		for (;it_wn != (*it_wl).end(); it_wn++)
			*it_wn = std::vector<double>(sizes.at(whichLayer - 1));
	}
	return ;
}


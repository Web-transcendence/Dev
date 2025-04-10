/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:33:52 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/10 09:21:12 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP
# define N_LAYER_HIDDEN 2
# define N_NEURON_INPUT 8
# define N_NEURON_OUTPUT 2
# define N_NEURON_HIDDEN 250

#include <string>
#include <vector>
#include <array>

class Network {
public:
	Network(std::string const & inFile);
	~Network( void ) {}

private:
	Network( void ) {}
	
	std::vector<std::vector<std::vector<double>>>	_weights;
	std::vector<std::vector<double>>				_biaises;
};

#endif

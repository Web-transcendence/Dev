/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:33:52 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/10 19:31:53 by tmouche          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP
# define N_LAYER_HIDDEN 1
# define N_NEURON_INPUT 16
# define N_NEURON_OUTPUT 4
# define N_NEURON_HIDDEN 25

#include <string>
#include <vector>
#include <array>

class Network {
public:
	Network(std::string const & inFile);
	~Network( void ) {}

	std::vector<double>	feedForward(std::vector<double> const & input);
	
private:
	Network( void ) {}
	
	std::vector<std::vector<std::vector<double>>>	_weights;
	std::vector<std::vector<double>>				_biaises;
};

#endif

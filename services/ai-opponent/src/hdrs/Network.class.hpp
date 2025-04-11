/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:33:52 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/11 13:40:39 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP

#include "Math.namespace.hpp"

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

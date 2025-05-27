/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 15:33:52 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/27 10:48:34 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP

#include "TypeDefinition.hpp"

#include <string>
#include <vector>
#include <array>

class Network {
public:
	Network(std::string const & inFile);
	~Network( void ) {}

	std::vector<float>	feedForward(std::vector<float> const & input);
	
private:
	Network( void ) {}


	float	dotProduct(float const * v1, float const * v2, unsigned int const size);

	float	sigmoid(float const z);
	void	sigmoid(float const * zs, float * res, unsigned int const size);

	unsigned int	_sizes[L_GLOBAL];

	float	_weights[WEIGHT_GLOBAL];
	float	_biais[BIAI_GLOBAL];
	
	//PLACEHOLDER
	float	_placeHidden[2][N_HIDDEN];
};

#endif

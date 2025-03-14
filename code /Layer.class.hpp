/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 15:56:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/14 16:20:50 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAYER_CLASS_HPP
# define LAYER_CLASS_HPP
# include <vector>
# include "Neuron.class.hpp"

class Layer {
public:
	Layer(int const n_neurons, int const n_weights) {
		for (int i = 0;i < n_neurons; i++)
			this->_neurons.push_back(new Neuron(n_weights));	
	}
private:
	std::vector<Neuron*>	_neurons;
};

#endif

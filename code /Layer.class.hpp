/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 15:56:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/14 17:13:00 by thibaud          ###   ########.fr       */
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

	void	updateWeight(double const eta, double const miniBatchSize) {
		for (auto n : this->_neurons)
			n->updateWeight(eta, miniBatchSize);
		return ;
	}

	void	updateNabla_w( void ) {
		for (auto n : this->_neurons)
			n->updateNabla_w();
		return ;
	}
	
	void	updateBias(double const eta, double const miniBatchSize) {
		for (auto n : this->_neurons)
			n->updateBias(eta, miniBatchSize);
		return ;
	}
	
	void	updateNabla_b( void ) {
		for (auto n : this->_neurons)
			n->updateNabla_b();
		return ;
	}

private:
	std::vector<Neuron*>	_neurons;
};

#endif

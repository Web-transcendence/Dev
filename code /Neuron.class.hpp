/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Neuron.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 10:25:23 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/14 17:00:46 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NEURON_HPP
# define NEURON_HPP
# include <vector>
# include <random>
# include "Math.namespace.hpp"

class Neuron {
public:
	Neuron(unsigned int const prevLayer) {
		std::random_device					rd;
		std::mt19937						gen(rd());
		std::normal_distribution<double>	dist(0.0, 1.0);
		
		for (unsigned int i = 0; i < prevLayer; i++)
			this->_weight.push_back(dist(gen));
		this->_bias = dist(gen);
		return ;
	}

	~Neuron( void ) {}

	double	feedForward(std::vector<double> const & input) const {
		return Math::sigmoid(Math::sumWeighted(input, this->_weight) + this->_bias);	
	}

	void	updateWeight(double const eta, double const miniBatchSize) {
		for (auto it_w = this->_weight.begin(), it_nw = this->_nabla_w.begin(); it_w != this->_weight.end(); it_w++)
			*it_w = *it_w - (eta / miniBatchSize) * *it_nw;
		this->_nabla_w.clear();
		return ;
	}

	void	updateNabla_w( void ) {
		for (auto it_nw = this->_nabla_w.begin(), it_dnw = this->_deltaNabla_w.begin(); it_nw != this->_nabla_w.end(); it_nw++) {
			*it_nw += *it_dnw;
		}
		this->_deltaNabla_w.clear();
	}
	
	void	updateBias(double const eta, double const miniBatchSize) {
		this->_bias -= (eta / miniBatchSize) * this->_nabla_b;
		this->_nabla_b = 0.0;
		return ;
	}
	
	void	updateNabla_b( void ) {
		this->_nabla_b += this->_deltaNabla_b;
		this->_deltaNabla_b = 0.0;
		return ;
	}

private:
	std::vector<double>	_weight;
	std::vector<double>	_nabla_w;
	std::vector<double>	_deltaNabla_w;
	double				_bias;
	double				_nabla_b;
	double				_deltaNabla_b;

friend class Network;
};

#endif

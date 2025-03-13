/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   neuron.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 10:25:23 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/13 16:23:27 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NEURON_HPP
# define NEURON_HPP
# include <vector>
# include <random>
# include <algorithm>
# include <cmath>

class Neuron {
public:
	Neuron(unsigned int const prevLayer) : _size(prevLayer) {
		std::random_device					rd;
		std::mt19937						gen(rd());
		std::normal_distribution<double>	dist(0.0, 1.0);

		for (unsigned int i = 0; i < this->_size; i++)
			this->_weight.push_back(dist(gen));
		this->_bias = dist(gen);
		return ;
	}

	~Neuron( void ) {}

	double	feedForward(std::vector<double> const & input) const {
		return sigmoid(sumWeighted(input));	
	}

private:
	double	sumWeighted(std::vector<double> const & input) const {
		double	res;

		for (auto it_w = this->_weight.begin(), it_i = input.begin(); it_w != this->_weight.end(), it_i != input.end(); it_w++, it_i++) {
			res += *it_i * *it_w;
		}
		return res + this->_bias;
	}

	static double	sigmoid(double const z) {return 1.0/(1.0+std::exp(-z));}
	
	static std::vector<double>*	sigmoid(std::vector<double> const & input) {
		std::vector<double>*	res = new std::vector<double>(input.size());
		
		for (auto it = input.begin(); it != input.end(); it++)
			res->push_back(sigmoid(*it));
		return res;
	}
	
	unsigned int const	_size;
	std::vector<double>	_weight;
	double				_bias;

friend class Network;
};

#endif

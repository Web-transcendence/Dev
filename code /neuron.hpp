/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   neuron.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 10:25:23 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/12 15:33:08 by thibaud          ###   ########.fr       */
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
	Neuron( void ) {}
	virtual ~Neuron( void ) {}

};

class InputNeuron : public Neuron {
public:
	InputNeuron( void ) {}
	~InputNeuron( void ) {}
	
private:
	int	_input;
};

class HidenNeuron : public Neuron {
public:
	HidenNeuron(unsigned int const prevLayer) : _size(prevLayer) {
		std::random_device					rd;
		std::mt19937						gen(rd());
		std::normal_distribution<double>	dist(0.0, 1.0);

		for (unsigned int i = 0; i < this->_size; i++)
			this->_weight.push_back(dist(gen));
		this->_bias = dist(gen);
		return ;
	}

	~HidenNeuron( void ) {}

	double	feedForward(double const input) const {
		return sigmoid(sumWeighted(input));	
	}

private:
	double	sumWeighted(double const input) const {
		double	res;

		for (int i = 0; i < this->_size; i++) {
			res += input * this->_weight[i];
		}
		return res + this->_bias;
	}

	double	sigmoid(double const z) const {return 1.0/(1.0+std::exp(-z));}

	unsigned int const	_size;
	std::vector<double>	_weight;
	double				_bias;
};

class OutputNeuron : public Neuron {
public:
	OutputNeuron( void ) {}
	~OutputNeuron( void ) {}
};

#endif

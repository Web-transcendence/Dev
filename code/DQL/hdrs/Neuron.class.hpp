/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Neuron.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 10:25:23 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 14:18:12 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NEURON_HPP
# define NEURON_HPP
# include <vector>
# include <random>

class Neuron {
public:
	Neuron(unsigned int const prevLayer);
	~Neuron( void ) {}
	
	double	feedForwardSigmoid(std::vector<double> const & input) const;
	double	feedForwardReLu(std::vector<double> const & input) const;
	double	affineTransformation(std::vector<double> const & input) const;
	void	updateWeight(double const eta, double const miniBatchSize);
	void	updateNabla_w( void );
	void	setDeltaNabla_w(std::vector<double> const & delta);
	void	updateBias(double const eta, double const miniBatchSize);
	void	updateNabla_b( void );
	void	setDeltaNabla_b(double delta);

private:
	std::vector<double>	_weight;
	std::vector<double>	_nabla_w;
	std::vector<double>	_deltaNabla_w;
	double				_bias;
	double				_nabla_b;
	double				_deltaNabla_b;

friend class Network;
friend class Layer;
};

#endif

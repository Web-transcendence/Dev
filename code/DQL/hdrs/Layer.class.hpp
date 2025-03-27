/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 15:56:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 14:18:33 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAYER_CLASS_HPP
# define LAYER_CLASS_HPP
# include <vector>

class Neuron;

class Layer {
public:
	Layer(int const n_neurons, int const n_weights);
	~Layer( void );

	std::vector<double>*	affineTransformation(std::vector<double> const & input);
	std::vector<double>*	feedForwardSigmoid(std::vector<double> const & input);
	std::vector<double>*	feedForwardReLu(std::vector<double> const & input);

	void	updateWeight(double const eta, double const miniBatchSize);
	void	updateNabla_w( void );
	void	setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation);
	void	updateBias(double const eta, double const miniBatchSize);
	void	updateNabla_b( void );
	void	setDeltaNabla_b(std::vector<double> const & delta);

	std::vector<double>*	calcDelta(std::vector<double> const & delta, std::vector<double> const & sp);
	
private:
	Layer( void ) {}

	std::vector<Neuron*>	_neurons;

friend class Network;
};

#endif

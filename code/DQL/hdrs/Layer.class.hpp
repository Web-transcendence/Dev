/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 15:56:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 17:13:10 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAYER_CLASS_HPP
# define LAYER_CLASS_HPP
# include "Math.namespace.hpp"
# include <vector>

class Neuron;

class Layer {
public:
	Layer(int const n_neurons, int const n_weights, t_actFunc actFunc);
	~Layer( void );

	std::vector<double>*	affineTransformation(std::vector<double> const & input);
	std::vector<double>*	feedForward(std::vector<double> const & input);

	double					callActFunc(double const input);
	std::vector<double>*	callActFunc(std::vector<double> const & input);
	double					callPrimeActFunc(double const input);
	std::vector<double>*	callPrimeActFunc(std::vector<double> const & input);

	void	copyLayer(Layer const & src);

	void	updateWeight(double const eta, double const miniBatchSize);
	void	updateNabla_w( void );
	void	setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation);
	void	updateBias(double const eta, double const miniBatchSize);
	void	updateNabla_b( void );
	void	setDeltaNabla_b(std::vector<double> const & delta);
	
	std::vector<double>*	calcDelta(std::vector<double> const & delta, std::vector<double> const & sp);
	
private:
	Layer( void ) {}

	double(*_actFuncSingle)(double const);
	std::vector<double>*(*_actFuncVector)(std::vector<double> const&);
    double(*_primeActFuncSingle)(double const);
	std::vector<double>*(*_primeActFuncVector)(std::vector<double> const&);

	std::vector<Neuron*>	_neurons;

friend class Network;
};

#endif

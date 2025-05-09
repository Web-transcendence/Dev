/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 15:56:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/08 22:46:16 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LAYER_CLASS_HPP
# define LAYER_CLASS_HPP
# include "Math.namespace.hpp"
# include <vector>

class Layer {
public:
	Layer(unsigned int const n_neurons,unsigned int const n_weights, t_actFunc actFunc);
	~Layer( void );

	double*	affineTransformation(double const *input);
	double*	feedForward(double const *input);

	double					callActFunc(double const input);
	std::vector<double>*	callActFunc(std::vector<double> const & input);
	double					callPrimeActFunc(double const input);
	std::vector<double>*	callPrimeActFunc(std::vector<double> const & input);

	void	updateWeight(double const eta, double const miniBatchSize);
	void	updateNabla_w( void );
	void	setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation);
	void	updateBias(double const eta, double const miniBatchSize);
	void	updateNabla_b( void );
	void	setDeltaNabla_b(std::vector<double> const & delta);
	
	std::vector<double>*	calcDelta(std::vector<double> const & delta, std::vector<double> const & sp);
	
private:
	Layer( void ) : sizeNeurons(0), sizeWeight(0) {}

	bool	checkErr(cudaError_t const * err, int const size);
	
	double(*_actFuncSingle)(double*);
	std::vector<double>*(*_actFuncVector)(std::vector<double> const&);
    double(*_primeActFuncSingle)(double const);
	std::vector<double>*(*_primeActFuncVector)(std::vector<double> const&);

	double		**weight;
	double		**nablaW;
	double		**deltaNablaW;
	double		*biais;
	double		*nablaB;
	double		*deltaNablaB;

	std::vector<double*>	contPtr_w;
	std::vector<double*>	contPtr_nw;
	std::vector<double*>	contPtr_dnw;

	int const	sizeNeurons;
	int const	sizeWeight;

friend class Network;
};

#endif

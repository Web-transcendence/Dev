/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 14:04:30 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/01 10:56:01 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Layer.class.hpp"
#include "Neuron.class.hpp"
#include "Math.namespace.hpp"

#include <array>;

Layer::Layer(unsigned int const n_neurons, unsigned int const n_weights, t_actFunc actFunc) :\
	sizeNeurons(n_neurons), sizeWeight(n_weights) {
	std::random_device					rd;
	std::mt19937						gen(rd());
	double 								stddev = 1.0 / std::sqrt(n_weights);
	std::normal_distribution<double> 	dist(0.0, stddev);

	cudaError_t	errN[3];
	errN[0] = cudaMalloc(&this->weight, n_neurons * sizeof(double*));
	errN[1] = cudaMalloc(&this->nablaW, n_neurons * sizeof(double*));
	errN[2] = cudaMalloc(&this->deltaNablaW, n_neurons * sizeof(double*));
	if (!this->checkErr(errN, 3)) {throw CudaMallocException();}
	this->contPtr_w = std::vector<double*>(n_neurons);
	this->contPtr_nw = std::vector<double*>(n_neurons);
	this->contPtr_dnw = std::vector<double*>(n_neurons);
	auto it_w = this->contPtr_w.begin();
	auto it_nw = this->contPtr_nw.begin();
	auto it_dnw = this->contPtr_dnw.begin();
	for (;it_w != this->contPtr_w.end();it_w++, it_nw++, it_dnw++) {
		errN[0] = cudaMalloc(&*it_w, n_weights * sizeof(double));
		errN[1] = cudaMalloc(&*it_nw, n_weights * sizeof(double));
		errN[2] = cudaMalloc(&*it_dnw, n_weights * sizeof(double));
		if (!this->checkErr(errN, 3))
			throw	CudaMallocException(); // bad handling have to free temp
		std::vector<double>	tempWeight(n_weights);
		for (double& w : tempWeight)
			w = dist(gen);
		cudaMemcpy(*it_w, tempWeight.data(), n_weights * sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(this->weight, this->contPtr_w.data(), n_neurons * sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(this->nablaW, this->contPtr_nw.data(), n_neurons * sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(this->deltaNablaW, this->contPtr_dnw.data(), n_neurons * sizeof(double*), cudaMemcpyHostToDevice);
	errN[0] = cudaMalloc(&this->biais, n_neurons * sizeof(double));
	errN[1] = cudaMalloc(&this->nablaB, n_neurons * sizeof(double));
	errN[2] = cudaMalloc(&this->deltaNablaB, n_neurons * sizeof(double));
	if (!checkErr(errN, 3))
		throw CudaMallocException();
	return ;
}

Layer::~Layer( void ) {
	auto it_w = this->contPtr_w.begin();
	auto it_nw = this->contPtr_nw.begin();
	auto it_dnw = this->contPtr_dnw.begin();
	for (;it_w != this->contPtr_w.end();it_w++, it_nw++, it_dnw++) {
		if (*it_w) {cudaFree(*it_w);}
		if (*it_nw) {cudaFree(*it_nw);}
		if (*it_nw) {cudaFree(*it_dnw);}
	}
	if (weight) {cudaFree(this->weight);}
	if (nablaW) {cudaFree(this->nablaW);}
	if (deltaNablaW) {cudaFree(this->deltaNablaW);}
	if (biais) {cudaFree(this->biais);}
	if (nablaB) {cudaFree(this->nablaB);}
	if (deltaNablaB) {cudaFree(this->deltaNablaB);}
	return ;
}

__global__ void	fireFeedForward(double const *input, double const **weight, double const *bias, double *res, ptrFuncS funcPtr, int const size) {
	int const	i = threadIdx.x;

	Math::dotProduct(input, weight[i], &res[i], size);
	res[i] += bias[i];
	(funcPtr)(&res[i]);
	return ;
}

double	*Layer::feedForward(double const *input) {
	double		*res;
	cudaError_t	err;

	err = cudaMalloc(&res, this->sizeNeurons * sizeof(double));
	if (err != cudaSuccess)
		throw CudaMallocException();
	fireFeedForward<<<1, this->sizeNeurons>>>(input, this->weight, this->biais, res, this->_actFuncSingle, this->sizeNeurons);
	return res;
}

__global__ void	fireAffineTransformation(double const *input, double const **weight, double const *bias, double *res, ptrFuncS funcPtr, int const size) {
	int const	i = threadIdx.x;

	Math::dotProduct(input, weight[i], &res[i], size);
	res[i] += bias[i];
	return ;
}

double*	Layer::affineTransformation(double const *input) {
	double		*res;
	cudaError_t	err;

	err = cudaMalloc(&res, this->sizeNeurons * sizeof(double));
	if (err != cudaSuccess)
		throw	cudaMallocException();
	fireAffineTransformation<<<1, this->sizeNeurons>>>(input, this->weight, this->biais, res, this->_actFuncSingle, this->sizeNeurons);;
	return res;
}

double	Layer::callActFunc(double const input) {
	return this->_actFuncSingle(input);
}

std::vector<double>*	Layer::callActFunc(std::vector<double> const & input) {
	return this->_actFuncVector(input);
}

double	Layer::callPrimeActFunc(double const input) {
	return this->_primeActFuncSingle(input);
}

std::vector<double>*	Layer::callPrimeActFunc(std::vector<double> const & input) {
	return this->_primeActFuncVector(input);
}

void	Layer::updateWeight(double const eta, double const miniBatchSize) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateWeight(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_w( void ) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateNabla_w();
	return ;
}

void	Layer::setDeltaNabla_w(std::vector<double> const & delta, std::vector<double> const & activation) {
	auto	product = Math::outerProduct(delta, activation);
	auto	it_p = product->begin();
	
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++, it_p++)
		(*it_n)->setDeltaNabla_w(*it_p);
	delete product;
	return ;
}

void	Layer::updateBias(double const eta, double const miniBatchSize) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateBias(eta, miniBatchSize);
	return ;
}

void	Layer::updateNabla_b( void ) {
	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++)
		(*it_n)->updateNabla_b();
	return ;
}

void	Layer::setDeltaNabla_b(std::vector<double> const & delta) {
	auto	it_n = this->_neurons.begin();
	auto	it_d = delta.begin();
	for (;it_d != delta.end() && it_n != this->_neurons.end(); it_d++, it_n++) {
		(*it_n)->setDeltaNabla_b(*it_d);
	}
	return ;
}

std::vector<double>*	Layer::calcDelta(std::vector<double> const & delta, std::vector<double> const & sp) {
	auto	merged = std::vector<std::vector<double>>(this->_neurons.size(), std::vector<double>(this->_neurons.at(0)->_weight.size()));
	auto	it = merged.begin();

	for (auto it_n = this->_neurons.begin(); it_n != this->_neurons.end(); it_n++) {
		auto	it_w = (*it).begin();
		for (auto it_we = (*it_n)->_weight.begin(); it_we != (*it_n)->_weight.end(); it_we++) {
			*it_w = *it_we;
			++it_w;
		}
		++it;
	}
	auto	transposed = Math::transpose2D(merged);
	auto	temp = std::vector<double>(this->_neurons.at(0)->_weight.size());
	auto	it_t = temp.begin();
	for (auto it_tr = transposed->begin(); it_tr != transposed->end(); it_tr++, it_t++) {
		(*it_t) = Math::dotProduct(*it_tr, delta);
	}
	delete transposed;
	return Math::hadamardProduct(temp, sp);	
}

bool	Layer::checkErr(cudaError_t const * err, int const size) {
	for (int i = 0; i < size; i++) {if (err[i] != cudaSuccess) {return false;}}
	return true;
}

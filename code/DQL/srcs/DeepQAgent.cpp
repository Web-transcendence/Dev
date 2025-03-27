/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   DeepQAgent.cpp                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 17:38:34 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "DeepQAgent.class.hpp"

#include "Environment.class.hpp"
#include "Network.class.hpp"
#include "Math.namespace.hpp"
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <array>

DeepQAgent::DeepQAgent(int const maxTraining, int const maxAct, double const learningRate, \
	double const discount, double const exploRate, double const exploDecay) : \
	_maxEpTraining(maxTraining), _maxActions(maxAct), _learningRate(learningRate), \
	_discount(discount), _explorationRate(exploRate), _explorationDecay(exploDecay) {
	this->_env = NULL;
	this->_QNet = NULL;
	return ;
}

DeepQAgent::~DeepQAgent( void ) {
	if (this->_QNet)
	    delete this->_QNet;
	return ;
}

void displayProgress(int current, int max) {
    int width = 20; // Number of total bar characters
    int progress = (current * width) / max; // Filled portion

    std::cout << "\r" << (current * 100) / max << "% [";
    for (int i = 0; i < width; i++) {
        std::cout << (i < progress ? '#' : '.');
    }
    std::cout << "]" << std::flush;
}

void	DeepQAgent::train( void ) {
	double exploRate = this->_explorationRate;
	int goal = 0;
	this->printQmatrix();
	for (int i = 0; i < this->_maxEpTraining; i++) {
		this->_env->reset();
		for (int a = 0; a < this->_maxActions; a++) {
			auto	input = this->mapPlacement(this->_env->_state);
			auto	pred_Q = this->_QNet->feedForwardReLu(*input);
			short	action = std::distance(pred_Q->begin(),\
			std::max_element(pred_Q->begin(), pred_Q->end()));
			if (1 / static_cast<double>(this->randInt()) < exploRate)
				action = randInt() % 4;
			std::array<int, 2>newState_Reward = this->_env->action(action);
			goal += newState_Reward[1];
			delete input;
			input = this->mapPlacement(newState_Reward[0]);
			auto	next_Q = this->_QNet->feedForwardReLu(*input);
			// std::cout << "state: " << this->_env->_state << std::endl;
			// Math::printdebug(*pred_Q, "predQ");
			pred_Q->at(action) = newState_Reward[1] + (this->_discount * *std::max_element(next_Q->begin(), next_Q->end()));
			// std::cout << "Nstate: " << newState_Reward[0] << std::endl;
			// Math::printdebug(*pred_Q, "upQ");
			this->_QNet->SDG(*input, *pred_Q, 0.05);
			this->_env->_state = newState_Reward[0];
			delete input;
			delete pred_Q;
			delete next_Q;
			if (this->_env->_done == true)
				break ;
		}
		if (exploRate - this->_explorationDecay > 0.001)
			exploRate -=  this->_explorationDecay;
		displayProgress(i,this->_maxEpTraining);
	}
	std::cout << std::endl;
	std::cout << "Goal hit: " << goal << std::endl;
}

bool	DeepQAgent::realisable( void ) {
	auto	testInput = this->mapPlacement(0);
	auto	output = this->_QNet->feedForwardReLu(*testInput);
	double	res = 0.0;
	
	for (auto o : *output)
		res += o;
	delete output;
	delete testInput;
	if (res == 0.)
		return false;
	return true;
}

void	DeepQAgent::printQmatrix(void) {
	for (unsigned int i = 0; i < this->_env->_myMap.size(); i++) {
		auto	input = this->mapPlacement(i);
		auto	output = this->_QNet->feedForwardReLu(*input);
		std::cout<<"Stage "<<i;
		Math::printdebug(*output, "");
		delete input;
		delete output;
	}
	return ;
}

void	DeepQAgent::test( void ) {
	this->_env->reset();
	this->printQmatrix();
	if (!this->realisable()) {
		std::cout << "NOT REALISABLE" << std::endl;
		this->_env->render();
		return ;
	}
	for (int a = 0; a < this->_maxActions; a++) {
		std::cout << "===========";
		this->_env->render();
		auto	input = this->mapPlacement(this->_env->_state);
		auto	pred_Q = this->_QNet->feedForwardReLu(*input);
		short	action = std::distance(pred_Q->begin(),\
		std::max_element(pred_Q->begin(), pred_Q->end()));
		std::array<int, 2>newState_Reward = this->_env->action(action);
		this->_env->_state = newState_Reward[0];
		delete input;
		delete pred_Q;
		if (this->_env->_done == true) {
			std::cout << "=== END ===";
			this->_env->render();
			break ;
		}
	}
}

void	DeepQAgent::genQMatrix( void ) {
	if (!this->_env)
		throw std::exception();
	this->_QMatrix = std::vector<std::vector<double>>(this->_env->_myMap.size(), std::vector<double>(4));
}

void	DeepQAgent::genQNet( void ) {
	if (!this->_env)
		throw std::exception();
	std::vector<unsigned int>	sizes(3);
	sizes[0] = this->_env->getObservationSpaceSize();
	sizes[1] = 10; // Arbitral value;
	sizes[2] = this->_env->getActionsSpaceSize();
	this->_QNet = new Network(sizes);
	return ;
}

std::vector<double>*	DeepQAgent::mapPlacement(int const state) {
	auto	placement = new std::vector<double>(this->_env->getObservationSpaceSize());
	placement->at(state) = 1.0;
	return placement;
}

int	DeepQAgent::randInt( void ) {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_int_distribution<int>	dist(0, 100);
	
	int	res = 0;
	while (!res)
		res = dist(gen);
	return res;
}
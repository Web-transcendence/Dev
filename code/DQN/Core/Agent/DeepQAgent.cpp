/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   DeepQAgent.cpp                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/31 13:02:28 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "DeepQAgent.class.hpp"

#include "../Network/Network.class.hpp"
#include "../Utils/Math.namespace.hpp"
#include "../Environment/Environment.class.hpp"
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

void	DeepQAgent::trainQMatrix( void ) {
	double exploRate = this->_explorationRate;
	this->_goalTraining = 0;
	for (int i = 0; i < this->_maxEpTraining; i++) {
		this->_env->reset();
		for (int a = 0; a < this->_maxActions; a++) {
			int action = this->policyQMatrix(TRAIN);
			std::array<int, 2>newState_Reward = this->_env->action(action);
			this->_QMatrix[this->_env->_state][action] = this->_learningRate\
				*(newState_Reward[1]\
				+this->_discount\
				*(*std::max_element(this->_QMatrix[newState_Reward[0]].begin(), \
					this->_QMatrix[newState_Reward[0]].end())));
			this->_env->_state = newState_Reward[0];
			this->_goalTraining += newState_Reward[1];
			if (this->_env->_done == true)
				break ;
		}
		if (exploRate - this->_explorationDecay > 0.001)
			exploRate -=  this->_explorationDecay;
		displayProgress(i, this->_maxEpTraining);
	}
	std::cout << std::endl << "QMatrix trained" << std::endl;
}

void	DeepQAgent::trainQNet( void )  {
	double exploRate = this->_explorationRate;

	if (!this->_goalTraining)
		return ;
	for (int iEp = 0; iEp < this->_maxEpTraining; iEp++) {
		this->_env->reset();
		for (int iAct = 0; iAct < this->_maxActions; iAct++) {
			auto	inputPrev = this->mapPlacement(this->_env->_state);
			auto	prevO = this->_QNet->feedForward(*inputPrev);
			int		action = std::distance(prevO->begin(),\
					std::max_element(prevO->begin(), prevO->end()));
			if (this->randDouble() < this->_explorationRate)
				action = this->randInt() % 4;
			std::array<int, 2>nextNreward = this->_env->action(action);	
			auto	inputNext = this->mapPlacement(nextNreward[0]);
			auto	nextO = this->_QNet->feedForward(*inputNext);
			auto	update0 = std::vector<double>(*prevO);
			update0[action] = (nextNreward[1] + (this->_discount * *std::max_element(nextO->begin(), nextO->end())));
			this->_QNet->SDG(*inputPrev, update0, 0.05);
			this->_env->_state = nextNreward[0];
			delete inputPrev;
			delete prevO;
			delete inputNext;
			delete nextO;
			if (this->_env->_done == true)
				break ;
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay;
		displayProgress(iEp, this->_maxEpTraining);
	}
	this->printQNet();
	this->printQMatrix();
	return ;
}

void	DeepQAgent::trainQNetFromQMatrix( void ) {
	if (!this->_goalTraining)
		return ;
	for (int i = 0; i < this->_maxEpTraining; i++) {
		for (int state = 0; state < this->_env->getObservationSpaceSize(); state++) {
			auto	input = this->mapPlacement(state);
			auto&	expected = this->_QMatrix.at(state);
			this->_QNet->SDG(*input, expected, 0.05);
 			delete input;
		}
		displayProgress(i, this->_maxEpTraining);
	}
	std::cout << std::endl << "QNet trained" << std::endl;
	return ;
}

void	DeepQAgent::testQMatrix( void ) {
	this->_env->reset();
	std::cout << "== QMatrix TEST ==" << std::endl;
	if (!this->_goalTraining) {
		std::cout << "== GOAL NOT ACCESSIBLE ==" << std::endl;
		return ;
	}
	for (int a = 0; a < this->_maxActions; a++) {
		std::cout << "===========";
		this->_env->render();
		int action = this->policyQMatrix(TEST);
		std::array<int, 2>newState_Reward = this->_env->action(action);
		this->_env->_state = newState_Reward[0];
		if (this->_env->_done == true) {
			std::cout << "=== END ===";
			this->_env->render();
			break ;
		}
	}
}

void	DeepQAgent::testQNet( void ) {
	this->_env->reset();
	std::cout << "== QNET TEST ==" << std::endl;
	if (!this->_goalTraining) {
		std::cout << "== GOAL NOT ACCESSIBLE ==" << std::endl;
		return ;
	}
	for (int a = 0; a < this->_maxActions; a++) {
		std::cout << "===========";
		this->_env->render();
		auto	input = this->mapPlacement(this->_env->_state);
		auto	output = this->_QNet->feedForward(*input);
		int		action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		delete input;
		delete output;
		std::array<int, 2>newState_Reward = this->_env->action(action);
		this->_env->_state = newState_Reward[0];
		if (this->_env->_done == true) {
			std::cout << "=== END ===";
			this->_env->render();
			break ;
		}
	}
	return ;
}

int	DeepQAgent::policyQMatrix(t_mode const mode) {
	int	act = 0;
	
	if (mode == TRAIN) {
		if (1.0 / static_cast<double>(this->randInt()) > this->_explorationRate) {
			act = std::distance(this->_QMatrix[this->_env->_state].begin(),\
			std::max_element(this->_QMatrix[this->_env->_state].begin(),\
			this->_QMatrix[this->_env->_state].end()));
		}
		else
			act = randInt() % 4;
	}
	else
		act = std::distance(this->_QMatrix[this->_env->_state].begin(),\
			std::max_element(this->_QMatrix[this->_env->_state].begin(),\
			this->_QMatrix[this->_env->_state].end()));
	return act;
}

void	DeepQAgent::printQMatrix(void) {
	std::cout<<std::endl;
	std::cout<<"==QMATRIX=="<<std::endl;
	for (int i = 0; i < this->_env->getObservationSpaceSize(); i++) {
		std::cout<<"Stage "<<i;
		Math::printdebug(this->_QMatrix.at(i), "");
	}
	return ;
}

void	DeepQAgent::printQNet(void) {
	std::cout<<std::endl;
	std::cout<<"==QNET=="<<std::endl;
	for (int i = 0; i < this->_env->getObservationSpaceSize(); i++) {
		auto	input = this->mapPlacement(i);
		auto	output = this->_QNet->feedForward(*input);
		std::cout<<"Stage "<<i;
		Math::printdebug(*output, "");
		delete input;
		delete output;
	}
	return ;
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
	sizes[0] = 16;
	sizes[1] = 25;
	sizes[2] = 4;
	this->_QNet = new Network(sizes, RELU, RELU);
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
	return dist(gen);
}

double	DeepQAgent::randDouble( void ) {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_real_distribution<double>	dist(0., 1.);
	return dist(gen);
}

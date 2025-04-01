/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/01 14:18:59 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Agent.class.hpp"

#include "../Network/Network.class.hpp"
#include "../Utils/Math.namespace.hpp"
#include "../Environment/Environment.class.hpp"
#include "ExpReplay.class.hpp"
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <array>

Agent::Agent(int const maxTraining, int const maxAct, double const learningRate, \
	double const discount, double const exploRate, double const exploDecay) : \
	_maxEpTraining(maxTraining), _maxActions(maxAct), _learningRate(learningRate), \
	_discount(discount), _explorationRate(exploRate), _explorationDecay(exploDecay) {
	this->_env = NULL;
	this->_QNet = NULL;
	return ;
}

Agent::~Agent( void ) {
	if (this->_QNet)
	    delete this->_QNet;
	return ;
}

void	Agent::batchTrain(unsigned int const batchSize) {
	unsigned int	action = 0;

	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	auto	TNetQ = std::vector<double>(batchSize);
	auto	it_tnet = TNetQ.begin();
	for (auto it_b = batches.begin(); it_b != batches.end(); it_b++,it_tnet++) {
		auto	oQNet = this->_QNet->feedForward((*it_b)->nextState->allState);
		delete oQNet;
		action = std::distance(oQNet->begin(), std::max_element(oQNet->begin(), oQNet->end()));
		auto	oTNet = this->_QNet->feedForward((*it_b)->nextState->allState);
		*it_tnet = oTNet->at(action);
		delete oTNet;
	}
		
}

void	Agent::trainQNet( void )  {
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


void	Agent::testQNet( void ) {
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

void	Agent::printQNet(void) {
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

void	Agent::genQNet(std::vector<unsigned int> const & sizes, t_actFunc hidden, t_actFunc output) {
	if (!this->_env || this->_QNet)
		throw std::exception();
	this->_QNet = new Network(sizes, hidden, output);
	return ;
}

void	Agent::genTNet(std::vector<unsigned int> const & sizes, t_actFunc hidden, t_actFunc output) {
	if (!this->_env || this->_TNet)
		throw std::exception();
	this->_TNet = new Network(sizes, hidden, output);
	return ;
}

int	Agent::randInt( void ) {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_int_distribution<int>	dist(0, 100);
	return dist(gen);
}

double	Agent::randDouble( void ) {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_real_distribution<double>	dist(0., 1.);
	return dist(gen);
}

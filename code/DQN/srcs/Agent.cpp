/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/04 13:44:32 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Agent.class.hpp"

#include "Network.class.hpp"
#include "Math.namespace.hpp"
#include "Environment.class.hpp"
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
	this->_TNet = NULL;
	this->_xp = NULL;
	return ;
}

Agent::~Agent( void ) {
	if (this->_QNet)
	    delete this->_QNet;
	if (this->_TNet)
		delete this->_TNet;
	if (this->_xp)
		delete this->_xp;
	return ;
}

void	Agent::train( void ) {
	double 				exploRate = this->_explorationRate;
	std::vector<double>	recordReward;
	
	if (!this->_goalTraining)
		return ;
	for (int iEp = 0; iEp < this->_maxEpTraining; iEp++) {
		double	totalReward = 0.0;
		this->_env->reset();
		for (int iAct = 0; iAct < this->_maxActions; iAct++) {
			auto	experience = new t_exp;
			experience->state = this->_env->_state;
			this->getAction(experience, exploRate);
			this->_env->action(experience);
			totalReward += experience->reward;
			this->_xp->add(experience);
			this->batchTrain(15);
			this->_env->_done = experience->done;
			this->_env->_state = experience->nextState;
			if (this->_env->_done)
				break ;
			if (!(iAct % 25) && iAct)
				this->TNetUpdate();
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay * 10;
		recordReward.push_back(totalReward);
		this->_QNet->displayProgress(iEp % 100, 100);
		if (!(iEp % 100) && iEp) {
			double averageReward = std::accumulate(recordReward.begin(),recordReward.end(),0.0) / recordReward.size();
			std::fill(recordReward.begin(), recordReward.end(), 0.0);
			std::cout<<"episodes: "<<iEp-100<<" to "<<this->_maxEpTraining<<", average reward: "<<averageReward<<", exploration: "<<exploRate<<std::endl;  
		}
	}
	return ;
}

void	Agent::test( void ) {
	
	if (!this->_goalTraining) {
		std::cout << "=== IMPOSSIBLE ===" << std::endl;
		return ;
	}
	for (int iAct = 0; iAct < this->_maxActions; iAct++) {
		t_exp	experience;
		experience.state = this->_env->_state;
		auto	output = this->_QNet->feedForward(this->_env->_state);
		experience.action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		this->_env->action(&experience);
		this->_env->_state = experience.nextState;
		if (experience.done)
			break ;
		delete output;
	}
	if (this->_env->_myMap[this->_env->getUIntState(this->_env->_state)] == 'G')
		std::cout << "=== SUCESS ===" << std::endl;
	else
		std::cout << "=== FAIL ===" << std::endl;
	return ;
}

void	Agent::trainQMatrix( void ) {
	double exploRate = this->_explorationRate;
	this->_goalTraining = 0;
	for (int i = 0; i < 10000; i++) {
		this->_env->reset();
		for (int a = 0; a < this->_maxActions; a++) {
			t_exp	xp;
			xp.action = this->policyQMatrix(TRAIN);
			xp.state = this->_env->_state;
			this->_env->action(&xp);
			this->_QMatrix[this->_env->getUIntState(xp.state)][xp.action] = this->_learningRate\
				*(xp.reward\
				+this->_discount\
				*(*std::max_element(this->_QMatrix[this->_env->getUIntState(xp.nextState)].begin(), \
					this->_QMatrix[this->_env->getUIntState(xp.nextState)].end())));
			this->_env->_state = xp.nextState;
			this->_goalTraining += xp.reward;
			this->_env->_done = xp.done;
			if (this->_env->_done == true)
				break ;
		}
		if (exploRate - this->_explorationDecay > 0.001)
			exploRate -=  this->_explorationDecay;
	}
	std::cout << "Training finished ";
	if (this->_goalTraining)
		std::cout << "map ok" << std::endl;
	else
		std::cout << "map nok" << std::endl;
	this->_env->render();
}

int	Agent::policyQMatrix(t_mode const mode) {
	int	act = -1;
	
	if (mode == TRAIN) {	
		if (this->randDouble() < this->_explorationRate) 
			act = randInt() % 4;
	}
	if (act == -1) {
		unsigned int const state = this->_env->getUIntState(this->_env->_state);
		act = std::distance(this->_QMatrix[state].begin(),\
		std::max_element(this->_QMatrix[state].begin(),\
		this->_QMatrix[state].end()));
	}
	return act;
}

void	Agent::batchTrain(unsigned int const batchSize) {
	unsigned int	action;

 	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	auto	expected = std::vector<double>(OUTPUT_SIZE);
	t_tuple	training;
	for (auto it_b = batches->begin(); it_b != batches->end(); it_b++) {
		auto	oQNet = this->_QNet->feedForward((*it_b)->nextState);
		action = std::distance(oQNet->begin(), std::max_element(oQNet->begin(), oQNet->end()));
		delete oQNet;
		auto	oTNet = this->_TNet->feedForward((*it_b)->nextState);
		expected.at(action) = (*it_b)->reward;
		if (!(*it_b)->done)
			expected.at(action) += this->_discount * oTNet->at(action);
		delete oTNet;
		training.input = (*it_b)->state;
		training.expectedOutput = expected;
		this->_QNet->SDG(&training, 0.05);
	}
	delete batches;
	return ;
}

void	Agent::getAction(t_exp * exp, double exploRate) const {
	if (this->randDouble() < exploRate)
		exp->action = this->randInt() % NUM_ACTION;
	else {
		auto	output = this->_QNet->feedForward(exp->state);
		exp->action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		delete output;
	}
	return ;
}

void	Agent::TNetUpdate( void ) {
	this->_TNet->copyNetwork(*this->_QNet);
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

void	Agent::genExpReplay(unsigned int const min, unsigned int const max) {
	this->_xp = new ExpReplay(max, min);
	return ;
}

void	Agent::genQMatrix( void ) {
	if (!this->_env)
		throw std::exception();
	this->_QMatrix = std::vector<std::vector<double>>(this->_env->_myMap.size(), std::vector<double>(NUM_ACTION));
}

int	Agent::randInt( void ) const {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_int_distribution<int>	dist(0, 100);
	return dist(gen);
}

double	Agent::randDouble( void ) const {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_real_distribution<double>	dist(0., 1.);
	return dist(gen);
}

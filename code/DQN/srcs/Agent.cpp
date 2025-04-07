/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/07 11:30:47 by thibaud          ###   ########.fr       */
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
			this->batchTrain(16);
			if (experience->done) {
				if (experience->reward) {this->TNetUpdate();}
				break ;
			}
			this->_env->_done = experience->done;
			this->_env->_state = experience->nextState;
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay;
		recordReward.push_back(totalReward);
		this->_QNet->displayProgress(iEp % 100, 100);
		if (!(iEp % 100) && iEp) {
			double averageReward = std::accumulate(recordReward.begin(),recordReward.end(),0.0) / recordReward.size();
			recordReward.clear();
			std::cout<<"epoch: "<<iEp-100<<" to "<<this->_maxEpTraining<<", average reward: "<<averageReward<<", exploration: "<<exploRate<<std::endl;  
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
 	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	t_tuple	training;
	for (auto it_b = batches->begin(); it_b != batches->end(); it_b++) {
		auto			oQNetState = this->_QNet->feedForward((*it_b)->state);
		auto			oTNet = this->_TNet->feedForward((*it_b)->nextState);
		auto			adjusted = std::vector<double>(*oQNetState);
		unsigned int	act = (*it_b)->action;
		unsigned int	actNext = std::distance(oTNet->begin(), std::max_element(oTNet->begin(), oTNet->end()));
		adjusted.at(act) = (*it_b)->reward; 
		if ((*it_b)->done == false) 
		adjusted.at(act) += this->_discount*(oTNet->at(actNext));
		training.input = (*it_b)->state;
		training.expectedOutput = adjusted;
		this->_QNet->SDG(&training, 0.05);
		delete oTNet;
		delete oQNetState;
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

void	Agent::printQMatrix(void) {
	std::cout<<std::endl;
	std::cout<<"==QMATRIX=="<<std::endl;
	for (int i = 0; i < 16; i++) {
		std::cout<<"Stage "<<i;
		Math::printdebug(this->_QMatrix.at(i), "");
	}
	return ;
}

void	Agent::printQNet( void ) {
	std::vector<double>	state(16);
	std::cout<<std::endl;
	std::cout<<"==QNET=="<<std::endl;
	for (int i = 0; i < 16; i++) {
		state.at(i) = 1.;
		auto	output = this->_QNet->feedForward(state);
		std::cout<<"Stage "<<i<<": ";
		Math::printdebug(*output, "");
		delete output;
		state.at(i) = 0.;
	}
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/28 22:07:45 by thibaud          ###   ########.fr       */
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
	
	for (int iEp = 0; iEp < this->_maxEpTraining; iEp++) {
		double	totalReward = 0.0;
		this->_env->reset();
		std::array<double,6>	prevState = this->_env->getState();
		for (int iAct = 0; iAct < this->_maxActions; iAct++) {
			auto	experience = new t_exp;
			experience->prevState = prevState;
			experience->state = this->_env->getState();
			prevState = this->_env->getState();
			this->getAction(experience, exploRate);
			this->_env->action(experience);
			totalReward += experience->reward;
			this->_xp->add(experience);
			this->batchTrain(16);
			if (experience->done) {
				if (experience->reward) {this->TNetUpdate();}
				break ;
			}
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay;
		recordReward.push_back(totalReward);
		this->_QNet->displayProgress(iEp % 10, 10);
		if (!(iEp % 10) && iEp) {
			double averageReward = std::accumulate(recordReward.begin(),recordReward.end(),0.0) / static_cast<double>(recordReward.size());
			recordReward.clear();
			std::cout<<"epoch: "<<iEp-100<<" to "<<this->_maxEpTraining<<", average reward: "<<averageReward<<", exploration: "<<exploRate<<std::endl;  
		}
	}
	return ;
}

// void	Agent::test( void ) {
	
// 	for (int iAct = 0; iAct < this->_maxActions; iAct++) {
// 		t_exp	experience;
// 		experience.state = this->_env->_state;
// 		auto	output = this->_QNet->feedForward(this->_env->_state);
// 		experience.action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
// 		this->_env->action(&experience);
// 		this->_env->_state = experience.nextState;
// 		if (experience.done)
// 			break ;
// 		delete output;
// 	}
// 	if (this->_env->_myMap[this->_env->getUIntState(this->_env->_state)] == 'G')
// 		std::cout << "=== SUCESS ===" << std::endl;
// 	else
// 		std::cout << "=== FAIL ===" << std::endl;
// 	return ;
// }


void	Agent::batchTrain(unsigned int const batchSize) {
 	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	t_tuple	training;
	for (auto it_b = batches->begin(); it_b != batches->end(); it_b++) {
		auto			input = this->_env->getStateVector((*it_b)->state, (*it_b)->prevState);
		auto			nextInput = this->_env->getStateVector((*it_b)->nextState, (*it_b)->state);
		auto			oQNet = this->_QNet->feedForward(*input);
		auto			oTNet = this->_TNet->feedForward(*nextInput);
		auto			adjusted = std::vector<double>(*oQNet);
		unsigned int	act = (*it_b)->action;
		unsigned int	actNext = std::distance(oTNet->begin(), std::max_element(oTNet->begin(), oTNet->end()));
		adjusted.at(act) = (*it_b)->reward; 
		if ((*it_b)->done == false) 
		adjusted.at(act) += this->_discount*(oTNet->at(actNext));
		training.input = input;
		training.expectedOutput = &adjusted;
		this->_QNet->SDG(&training, 0.05);
		delete oTNet;
		delete oQNet;
		delete input;
		delete nextInput;
	}
	delete batches;
	return ;
}

void	Agent::getAction(t_exp * exp, double exploRate) const {
	if (this->randDouble() < exploRate)
		exp->action = this->randInt() % OUTPUT_SIZE;
	else {
		auto	input = this->_env->getStateVector(exp->state, exp->prevState);
		auto	output = this->_QNet->feedForward(*input);
		exp->action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
		delete input;
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

void	Agent::saveNetwork( void ) {
	this->_QNet->printNetworkToJson("weights.json");
	return ;	
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

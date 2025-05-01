/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/01 21:09:58 by tmouche          ###   ########.fr       */
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
#include <thread>
#include <chrono>
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
		std::vector<double>	recordStep;
		double	totalReward = 0.0;
		this->_env->reset();
		for (int iAct = 0; ; iAct++) {
			auto	experience = new t_exp;
			experience->state = this->_env->getState();
			this->getAction(experience, exploRate);
			this->_env->action(experience);
			totalReward += experience->reward;
			this->_xp->add(experience);
			this->batchTrain(16);
			if (experience->reward >= 1.) {this->TNetUpdate();}
			if (experience->done) {
				recordStep.push_back(iAct);
				break ;
			}
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay;
		recordReward.push_back(totalReward);
		this->_QNet->displayProgress(iEp % 100, 100);
		if (!(iEp % 100) && iEp) {
			double averageReward = std::accumulate(recordReward.begin(),recordReward.end(),0.0) / static_cast<double>(recordReward.size());
			double averageStep = std::accumulate(recordStep.begin(),recordStep.end(),0.0) / static_cast<double>(recordStep.size());
			recordReward.clear();
			std::cout<<"epoch: "<<iEp-100<<" to "<<this->_maxEpTraining<<", average reward: "<<averageReward<<", average steps: "<<averageStep<<", exploration: "<<exploRate<<std::endl;
			this->_QNet->printNetworkToJson("weights.json");
		}
	}
	return ;
}

void	Agent::test( void ) {
	this->_env->reset();
	for (int iAct = 0; iAct < this->_maxActions; iAct++) {
		t_exp	experience;
		experience.state = this->_env->getState();
		this->getAction(&experience, 0.0);
		this->_env->action(&experience);
    	this->_env->displayState(*experience.state);
		if (experience.done)
			break ;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	return ;
}

void	Agent::test(Network & QnetTest) {
	this->_env->reset();
	for (int iAct = 0; iAct < 1000; iAct++) {
		t_exp	experience;
		experience.state = this->_env->getState();
		auto	output = QnetTest.feedForwardTest(*experience.state);
		experience.action = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
		this->_env->action(&experience);
    	this->_env->displayState(*experience.state);
		if (experience.done) {
			this->_env->reset();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	return ;
}


void	Agent::batchTrain(unsigned int const batchSize) {
 	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	t_tuple	training;
	for (auto it_b = batches->begin(); it_b != batches->end(); it_b++) {
		auto			oQNet = this->_QNet->feedForward(*(*it_b)->state);
		auto			oTNet = this->_TNet->feedForward(*(*it_b)->nextState);
		auto			adjusted = std::vector<double>(*oQNet);
		unsigned int	act = (*it_b)->action;
		unsigned int	actNext = std::distance(oTNet->begin(), std::max_element(oTNet->begin(), oTNet->end()));
		adjusted.at(act) = (*it_b)->reward; 
		if ((*it_b)->done == false) {adjusted.at(act) += this->_discount*(oTNet->at(actNext));}
		training.input = (*it_b)->state.get();
		training.expectedOutput = &adjusted;
		this->_QNet->SDG(&training, 0.05);
		// std::cout << " ------------------ " << std::endl;
		// Math::printdebug(*training.input, "input");
		// Math::printdebug(*training.expectedOutput, "output");
		delete oQNet;
		delete oTNet;
	}
	delete batches;
	return ;
}

void	Agent::getAction(t_exp * exp, double exploRate) const {
	if (this->randDouble() < exploRate)
		exp->action = this->randInt() % OUTPUT_SIZE;
	else {
		auto	output = this->_QNet->feedForward(*exp->state);
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

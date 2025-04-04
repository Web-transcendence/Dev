/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 21:48:28 by thibaud          ###   ########.fr       */
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
	return ;
}

Agent::~Agent( void ) {
	if (this->_QNet)
	    delete this->_QNet;
	if (this->_TNet)
		delete this->_TNet;
	return ;
}

void	Agent::train( void ) {
	double 				exploRate = this->_explorationRate;
	std::vector<double>	recordReward;
	
	for (int iEp = 0; iEp < this->_maxEpTraining; iEp++) {
		double	totalReward = 0.0;
		for (int iAct = 0; iAct < this->_maxActions; iAct++) {
			auto	experience = new t_exp;
			experience->state = this->_env->_state;
			this->getAction(experience, exploRate);
			this->_env->action(experience);
			totalReward += experience->reward;
			this->_xp->add(experience);
			this->batchTrain(64);
			if (experience->done)
				break ;
			if (!(iAct % 25))
				this->TNetUpdate();
			this->_env->_state = experience->nextState;
		}
		if (exploRate - this->_explorationDecay > 0.0001)
			exploRate -= this->_explorationDecay;
		recordReward.push_back(totalReward);
		if (!(iEp % 100)) {
			double averageReward = std::accumulate(recordReward.begin(),recordReward.end(),0.0) / recordReward.size();
			recordReward.clear();
			std::cout<<"episodes: "<<iEp-100<<" to "<<iEp<<", average reward: "<<averageReward<<", exploration: "<<exploRate<<std::endl;  
		}
	}
	return ;
}

void	Agent::batchTrain(unsigned int const batchSize) {
	unsigned int	action = 0;

	if (this->_xp->getNum() < this->_xp->getMin())
		return ;
	auto	batches = this->_xp->getBatch(batchSize);
	auto	QNetQ = std::vector<double>(batchSize);
	auto	it_qnet = QNetQ.begin();
	auto	expected = std::vector<double>(OUTPUT_SIZE);
	t_tuple	training;
	for (auto it_b = batches.begin(); it_b != batches.end(); it_b++,it_qnet++) {
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
		this->_QNet->SDG(&training, this->_learningRate);
		expected.clear();
	}
	return ;
}

void	Agent::getAction(t_exp * exp, double exploRate) const {
	auto	output = this->_QNet->feedForward(exp->state);
	if (this->randDouble() < exploRate)
		exp->action = this->randInt() % 2;
	else
		exp->action = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
	delete output;
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

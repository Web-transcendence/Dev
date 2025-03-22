/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   QAgent.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:47:33 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/22 15:35:52 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "QAgent.class.hpp"

#include "Environment.class.hpp"
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <array>

QAgent::QAgent(int const maxTraining, int const maxAct, double const learningRate, \
	double const discount, double const exploRate, double const exploDecay) : \
	_maxEpTraining(maxTraining), _maxActions(maxAct), _learningRate(learningRate), \
	_discount(discount), _explorationRate(exploRate), _explorationDecay(exploDecay) {
		this->_env = NULL;
	return ;
}

QAgent::~QAgent( void ) {}

void	QAgent::train( void ) {
	double exploRate = this->_explorationRate;

	for (int i = 0; i < this->_maxEpTraining; i++) {
		this->_env->reset();
		for (int a = 0; a < this->_maxActions; a++) {
			int action = this->policy(TRAIN);
			std::array<int, 2>newState_Reward = this->_env->action(action);
			this->_QMatrix[this->_env->_state][action] = this->_QMatrix[this->_env->_state][action]\
				+this->_learningRate\
				*(\
					newState_Reward[1]\
					+this->_discount\
					*(*std::max_element(this->_QMatrix[newState_Reward[0]].begin(), this->_QMatrix[newState_Reward[0]].end()))\
				)\
				-this->_QMatrix[this->_env->_state][action];
			this->_env->_state = newState_Reward[0];
			if (this->_env->_done == true)
				break ;
		}
		if (exploRate - this->_explorationDecay > 0.001)
			exploRate -=  this->_explorationDecay; 
	}
}

void	QAgent::test( void ) {
	this->_env->reset();
	for (int a = 0; a < this->_maxActions; a++) {
		this->_env->render();
		int action = this->policy(TEST);
		std::array<int, 2>newState_Reward = this->_env->action(action);
		this->_env->_state = newState_Reward[0];
		if (this->_env->_done == true) {
			std::cout << "END ";
			this->_env->render();
			break ;
		}
	}
}

int	QAgent::policy(t_mode const mode) {
	int	act[4] = {UP, DOWN, RIGHT, LEFT};
	int	idx;
	
	if (mode == TRAIN) {
		if (this->randInt() > this->_explorationRate)
			idx = std::distance(this->_QMatrix[this->_env->_state].begin(),\
				std::max_element(this->_QMatrix[this->_env->_state].begin(),\
				this->_QMatrix[this->_env->_state].end()));
		else
			idx = randInt() % 4;
	}
	else
		idx = std::distance(this->_QMatrix[this->_env->_state].begin(),\
			std::max_element(this->_QMatrix[this->_env->_state].begin(),\
			this->_QMatrix[this->_env->_state].end()));
	return act[idx];
}



void	QAgent::genQMatrix( void ) {
	if (!this->_env)
		throw std::exception();
	this->_QMatrix = std::vector<std::vector<double>>(this->_env->_myMap.size(), std::vector<double>(4));
}

int	QAgent::randInt( void ) {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_int_distribution<int>	dist(0, 100);
	return dist(gen);
}
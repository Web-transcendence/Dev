/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 21:57:09 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"

#include "TypeDefinition.hpp"
#include <random>
#include <iostream>

Environment::Environment(int const col, int const row, double const rewardTo, unsigned int const maxStep) \
	: _row(row), _col(col), _rewardThreshold(rewardTo), _maxEpStep(maxStep) {
	char	corpus[2] = {'F','H'};

	this->_myMap = std::vector<char>(row * col);
	this->_myMap.front() = 'S';
	this->_myMap.back() = 'G';
	for (auto it = this->_myMap.begin(); it != this->_myMap.end(); it++) {
		if (!*it)
			*it = corpus[randInt()];
	}
	this->_state= std::vector<double>(IN_STATE);
	return ;
}

Environment::~Environment( void ) {}

void	Environment::action(t_exp * exp) {
	unsigned int const	size = this->_myMap.size();
	int					diff[4] = {-1 * int(this->_col), int(this->_col), 1 , -1 };

	exp->state = this->_state;
	exp->nextState = this->_state;
	if ((exp->action == RIGHT && static_cast<unsigned int>(exp->state[0]) % this->_col != this->_col - 1) \
	|| (exp->action == LEFT && static_cast<unsigned int>(exp->state[0]) % this->_col != 0) \
	|| (exp->action == UP && static_cast<unsigned int>(exp->state[0]) - this->_col < size) \
	|| (exp->action == DOWN && static_cast<unsigned int>(exp->state[0]) + this->_col < size))
		exp->nextState[0] += diff[exp->action];
	char const	place = this->_myMap[exp->nextState[0]];
	exp->reward = 0;
	if (place == 'G' || place == 'H') {
		exp->done = true;
		if (place == 'G')
			++exp->reward;
	}
	return ;
}

void	Environment::render( void ) {
	for (unsigned int i = 0; i < this->_myMap.size(); i++) {
		if (i % 4 == 0)
			std::cout << std::endl;
		if (static_cast<int>(i) == this->_state[0])
			std::cout << "'" << this->_myMap[i] << "'";
		else
			std::cout << this->_myMap[i];
	}
	std::cout << std::endl;
	return ;
}

int	Environment::randInt( void ) {
	static std::random_device				rd;
    static std::mt19937						gen(rd());
	static std::discrete_distribution<int>	dist({2, 1});
	return dist(gen);	
}
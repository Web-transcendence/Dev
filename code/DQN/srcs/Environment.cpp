/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/30 17:44:24 by tmouche          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"

#include "TypeDefinition.hpp"
#include "Math.namespace.hpp"
#include <random>
#include <iostream>
#include <algorithm>

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
	this->_state = std::vector<double>(IN_STATE);
	return ;
}

Environment::~Environment( void ) {}

void	Environment::action(t_exp * exp) {
	unsigned int const	size = this->_myMap.size();
	unsigned int		state = this->getUIntState(exp->state);
	int					diff[4] = {-1 * int(this->_col), int(this->_col), 1 , -1 };
	if ((exp->action == RIGHT && state % this->_col != this->_col - 1) \
	|| (exp->action == LEFT && state % this->_col != 0) \
	|| (exp->action == UP && state - this->_col < size) \
	|| (exp->action == DOWN && state + this->_col < size))
		state += diff[exp->action];
	char const	place = this->_myMap[state];
	if (place == 'G' || place == 'H') {
		exp->done = true;
		if (place == 'G')
			exp->reward += 1;
		else
			exp->reward -= 0.1;
	}
	exp->nextState.at(state) = 1.0;
	return ;
}

void	Environment::reset( void ) {
	this->_done = false;
	std::fill(this->_state.begin(), this->_state.end(), 0.0);
	this->_state.front() = 1.0;
	return ;
}

void	Environment::render( void ) {
	for (unsigned int i = 0; i < this->_myMap.size(); i++) {
		if (i % 4 == 0)
			std::cout << std::endl;
		std::cout << this->_myMap[i];
	}
	std::cout << std::endl;
	return ;
}

unsigned int	Environment::getUIntState(std::vector<double> const & src) {
	return std::distance(src.begin(), std::max_element(src.begin(), src.end()));
}

int	Environment::randInt( void ) {
	static std::random_device				rd;
    static std::mt19937						gen(rd());
	static std::discrete_distribution<int>	dist({2, 1});
	return dist(gen);	
}
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 11:02:44 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
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
	this->_state = 0;
	this->_done = false;
	return ;
}

Environment::~Environment( void ) {}



std::array<int, 2>	Environment::action(int const act) {
	unsigned int const	size = this->_myMap.size();
	int					nextState = this->_state;
	int					diff[4] = {-1 * int(this->_col), 1* int(this->_col), 1 , -1 };
	if ((act == RIGHT && this->_state % this->_col != this->_col - 1) \
	|| (act == LEFT && this->_state % this->_col != 0) \
	|| (act == UP && this->_state - this->_col < size) \
	|| (act == DOWN && this->_state + this->_col < size))
		nextState += diff[act];
	char const	place = this->_myMap[nextState];
	int			reward = 0;
	if (place == 'G' || place == 'H') {
		this->_done = true;
		if (place == 'G')
			++reward;
	}
	return std::array<int, 2>{nextState, reward};
}

void	Environment::render( void ) {
	for (unsigned int i = 0; i < this->_myMap.size(); i++) {
		if (i % 4 == 0)
			std::cout << std::endl;
		if (static_cast<int>(i) == this->_state)
			std::cout << "'" << this->_myMap[i] << "'";
		else
			std::cout << this->_myMap[i];
	}
	std::cout << std::endl;
	return ;
}

void	Environment::reset( void ) {
	this->_state = 0;
	this->_done = false;
	return ;
}

int	Environment::randInt( void ) {
	static std::random_device				rd;
    static std::mt19937						gen(rd());
	static std::discrete_distribution<int>	dist({2, 1});
	return dist(gen);	
}
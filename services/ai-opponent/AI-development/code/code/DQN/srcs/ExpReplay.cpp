/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ExpReplay.cpp                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:39:14 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/07 11:04:50 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ExpReplay.class.hpp"
#include <set>
#include <random>

#include <iostream>

ExpReplay::ExpReplay( void ) : _max(0), _min(0) {};

ExpReplay::ExpReplay(unsigned int const max, unsigned int const min) : _max(max), _min(min) {
	this->_size = 0;
	this->_experiences = std::vector<t_exp*>(this->_max, NULL);
	return ;
}

ExpReplay::~ExpReplay( void ) {
	for (auto it = this->_experiences.begin(); it != this->_experiences.end(); it++) {
		if (*it)
			delete *it;
	}
	return ;
}

void	ExpReplay::add(t_exp * experience) {
	if (this->_size != this->_max) {
		this->_experiences.at(this->_size) = experience;
		++this->_size;
	}
	else {
		delete this->_experiences.back();
		for (auto it_a = this->_experiences.end() - 1, it_b = it_a - 1; it_a != this->_experiences.begin(); it_a--, it_b--)
			*it_a = *it_b;
		this->_experiences.front() = experience;
	}
	return ;
}

std::vector<t_exp*>*	ExpReplay::getBatch(unsigned int const size) const {
	std::set<int>						resRandomIdx;
	std::random_device 					rd;
	std::mt19937 						gen(rd());  
	std::uniform_int_distribution<int>	dist(0, this->_size - 1);
	auto								res = new std::vector<t_exp*>(size);
	
	if (size > this->_size)
		throw std::exception();
	for (;resRandomIdx.size() < size;) {resRandomIdx.emplace(dist(gen));}
	auto it_res = res->begin();
	for (auto idx : resRandomIdx)
		*it_res++ = this->_experiences.at(idx);
	return res;
}

unsigned int	ExpReplay::getMax( void ) const {return this->_max;}
unsigned int	ExpReplay::getMin( void ) const {return this->_min;}
unsigned int	ExpReplay::getNum( void ) const {return this->_size;}

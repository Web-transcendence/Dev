/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ExpReplay.cpp                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/25 11:40:45 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/25 12:35:57 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "ExpReplay.class.hpp"
#include <numeric>
#include <random>
#include <algorithm>

ExpReplay::ExpReplay(unsigned int const maxExp, unsigned int const minExp) {
	this->maxExp = maxExp;
	this->minExp = minExp;
	this->experiences = std::vector<t_exp*>(maxExp);
	this->size = 0;
	return ;	
}

std::vector<t_exp*>*	ExpReplay::getBatch(unsigned int const batchSize) {
	std::random_device 	rd;
    std::mt19937 		g(rd());
	auto				batch = new std::vector<t_exp*>(batchSize);
	auto				rand = std::vector<unsigned int>(batchSize);
	auto				it_rand = rand.begin();

	std::iota(it_rand, rand.end(), 0);
	std::shuffle(it_rand, rand.end(), g);
	for (auto it = batch->begin(); it != batch->end(); it++, it_rand++)
		*it = this->experiences.at(*it_rand);
	return batch;
}

void	ExpReplay::add(t_exp * experience) {
	if (this->size == this->maxExp)
		delete this->experiences.back();
	else
		++this->size;
	this->experiences.pop_back();
	this->experiences.insert(this->experiences.begin(), experience);
	return ;
}


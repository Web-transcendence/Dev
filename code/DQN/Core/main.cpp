/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/30 01:23:38 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "DeepQAgent.class.hpp"
#include <iostream>

int main( void ) {
	Environment	myEnv(4,4, 0.82, 100);
	DeepQAgent	myAgent(40000, 99, 0.83, 0.93, 1.0, 1.0/40000);

	myAgent.setMap(myEnv);
	myAgent.genQMatrix();
	myAgent.genQNet();
	std::cout << std::endl << "=== TRAINING ===" << std::endl;
	myAgent.trainQMatrix();
	myAgent.trainQNet();
	std::cout << std::endl << "=== TESTING ===" << std::endl;
	myAgent.testQMatrix();
	myAgent.testQNet();
	return 0;
}

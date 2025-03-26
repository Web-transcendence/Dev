/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/26 15:37:31 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "DeepQAgent.class.hpp"
#include <iostream>

int main( void ) {
	Environment	myEnv(4,4, 0.82, 100);
	DeepQAgent	myAgent(20000, 99, 0.83, 0.93, 1.0, 1.0/20000);

	myAgent.setMap(myEnv);
	myAgent.genQNet();
	std::cout << std::endl << "=== TRAINING ===" << std::endl;
	myAgent.train();
	std::cout << std::endl << "=== TESTING ===" << std::endl;
	myAgent.test();
	return 0;
}


// myNetwork.SDG(dataset.training, 30, 10, 3.0, &dataset.testing);
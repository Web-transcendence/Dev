/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 21:57:17 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "Agent.class.hpp"
#include "Network.class.hpp"
#include <iostream>

int main( void ) {
	Environment	myEnv(4,4, 0.82, 100);
	Agent	myAgent(40000, 99, 0.83, 0.93, 1.0, 1.0/40000);
	std::vector<unsigned int>	net = {6,250,250,2};
	myAgent.setMap(myEnv);
	myAgent.genTNet(net, LEAKYRELU, LEAKYRELU);
	myAgent.genQNet(net, LEAKYRELU, LEAKYRELU);
	std::cout << std::endl << "=== TRAINING ===" << std::endl;
	myAgent.train();
	std::cout << std::endl << "=== TESTING ===" << std::endl;
	// myAgent.test();
	return 0;
}

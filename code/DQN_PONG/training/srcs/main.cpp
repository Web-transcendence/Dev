/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/29 02:13:39 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "Agent.class.hpp"
#include "Network.class.hpp"
#include <iostream>

int main( void ) {
	Environment	myEnv(1000);
	Agent	myAgent(1000, 99, 0.95, 0.93, 1.0, 1./1000.);
	std::vector<unsigned int>	net = {2400,25,25,3};
	myAgent.setMap(myEnv);
	myAgent.genTNet(net, LEAKYRELU, LEAKYRELU);
	myAgent.genQNet(net, LEAKYRELU, LEAKYRELU);
	myAgent.genExpReplay(1000, 5000);
	std::cout << std::endl << "=== TRAINING DQN ===" << std::endl;
	myAgent.train();
	std::cout << std::endl << "=== TESTING ===" << std::endl;
	// myAgent.test();
	return 0;
}

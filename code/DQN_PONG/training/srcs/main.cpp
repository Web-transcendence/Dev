/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/04 18:55:53 by tmouche          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "Agent.class.hpp"
#include "Network.class.hpp"
#include <iostream>

int main(int ac, char** av) {
	Environment	myEnv;
	Agent	myAgent(50000, 250, 0.95, 0.93, 1.0, 1./50000.);
	std::vector<unsigned int>	net = {INPUT_SIZE,25,OUTPUT_SIZE};
	myAgent.setMap(myEnv);
	myAgent.genTNet(net, SIGMOIDs, SIGMOIDs);
	myAgent.genQNet(net, SIGMOIDs, SIGMOIDs);
	myAgent.genExpReplay(2500, 25000);
	std::cout << std::endl << "=== TRAINING DQN ===" << std::endl;
	myAgent.train();
	std::cout << std::endl << "=== TESTING ===" << std::endl;
	Network	tester(av[1]);
	myAgent.test(tester);
	(void)ac;
	return 0;
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 15:36:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/22 15:45:46 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"
#include "QAgent.class.hpp"
#include <iostream>

int main( void ) {
	Environment	myEnv(4,4, 0.82, 100);
	QAgent		myAgent(20000, 99, 0.83, 0.93, 1.0, 1.0/20000);

	myAgent.setMap(myEnv);
	myAgent.train();
	std::cout << "=== TESTING ===" << std::endl;
	myAgent.test();
	return ;
}
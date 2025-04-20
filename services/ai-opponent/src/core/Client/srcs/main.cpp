/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/20 14:37:11 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/20 14:53:08 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"
#include "Factory.class.hpp"

#include <iostream>

int	main(int ac, char** av) {
	if (ac != 2) {
		std::cout << "Error: Parameters: ./Facory <Game Server Ws>" << std::endl;
		return 1;
	}
	try {
		Factory	myFactory(av[1]);
		myFactory.run();
	} catch (std::exception const & e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
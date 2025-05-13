/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/20 14:37:11 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/07 14:27:51 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"
#include "Factory.class.hpp"

#include <iostream>

int	main(int ac, char** av) {
	if (ac != 3) {
		std::cout << "Error: Parameters: ./Factory <Game Server Ws> <port http factory>" << std::endl;
		return 1;
	}
	try {
		std::stringstream	ss(av[2]);
		int					port;
		ss >> port;
		Factory	myFactory(av[1]);
		myFactory.run(port);
	} catch (std::exception const & e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
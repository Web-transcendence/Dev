/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/20 14:37:11 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/21 13:24:51 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"
#include "Factory.class.hpp"

#include <iostream>

int	main( void ) {
	try {
		Factory	myFactory;
		myFactory.run();
	} catch (std::exception const & e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
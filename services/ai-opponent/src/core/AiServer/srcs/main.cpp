/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/10 15:41:00 by tmouche           #+#    #+#             */
/*   Updated: 2025/05/21 13:58:20 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Math.namespace.hpp"
#include "AiServer.class.hpp"

#include <iostream>
#include <string>

int	main( void ) {
	try {
		AiServer	server("weights.json");
		server.start();
	} catch (websocketpp::exception const & e) {
		std::cout << e.what() << std::endl;
	} catch (std::exception const & e ) {
		std::cout << e.what() << std::endl;
	}
	return 0;
}

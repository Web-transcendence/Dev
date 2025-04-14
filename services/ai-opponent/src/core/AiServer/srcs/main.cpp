/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/10 15:41:00 by tmouche           #+#    #+#             */
/*   Updated: 2025/04/11 13:00:30 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Math.namespace.hpp"
#include "AiServer.class.hpp"

#include <iostream>
#include <string>
#include <chrono>



int	main( void ) {
	try {
		AiServer	server("weight.json");
		server.start();
	} catch (websocketpp::exception const & e) {
		std::cout << e.what() << std::endl;
	} catch ( ... ) {
		std::cout << "An unexpected exception" << std::endl;
	}
	return 0;
}

// auto debut = std::chrono::high_resolution_clock::now();
// auto res = QNet.feedForward(input);
// auto fin = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double, std::milli> duree = fin - debut;
// std::cout << "Temps d'exécution : " << duree.count() << " ms" << std::endl;
// Math::printdebug(res, "output");
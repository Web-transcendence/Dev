/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/22 15:12:37 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/23 14:32:38 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Game.hpp"
#include "Server.hpp"

#include <thread>
#include <chrono>

int	main( void ) {
	Server	myServer(8080, "Game Center");
	Game	game1(16019, "Game1");
	Game	game2(16020, "Game2");

	std::thread	gc([&myServer]{myServer.run();});
	std::thread	g1([&game1]{game1.run();});
	std::thread	g2([&game2]{game2.run();});
	
	std::cout << "Game Center waiting for connection.." << std::endl;
	while (!myServer.isReady()) {std::this_thread::sleep_for(std::chrono::milliseconds(100));}
	std::cout << "Game Center ready" << std::endl;
	myServer.request(game1.getWs());
	myServer.request(game2.getWs());
	
	while (!game1.isReady()) {std::this_thread::sleep_for(std::chrono::milliseconds(100));}
	std::cout << "Game Server 1 ready" << std::endl;
	while (!game2.isReady()) {std::this_thread::sleep_for(std::chrono::milliseconds(100));}
	std::cout << "Game Server 2 ready" << std::endl;
	std::thread	g12([&game1]{game1.stateSendLoop(15);});
	std::thread	g22([&game2]{game2.stateSendLoop(15);});
	g12.join();
	g22.join();
	std::terminate();
	return (0);
}
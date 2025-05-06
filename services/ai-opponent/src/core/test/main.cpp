/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/06 08:52:55 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/06 13:37:42 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../third-party/crow.h"
#include <thread>
#include <chrono>
#include <string>
#include <csignal>
#include <iostream>
#include <atomic>
#include <unistd.h> 

std::atomic<bool>	shouldExit{false};
crow::SimpleApp*	myApp;

void	handling_sigterm(int) {
	shouldExit.store(true);
}

void	loop(std::thread::id id, int port) {
	while (shouldExit.load() == false) {
		std::cout << "Pid: " << getpid() << " :ShouldExit: " << shouldExit.load() << std::endl;
		std::cout << "parent: " << id << ": child: " << std::this_thread::get_id() << ": Hello World" << std::endl;
		std::cout << "port: " << port << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	myApp->stop();
	return ;
}

void	test(crow::SimpleApp & app) {
	CROW_ROUTE(app, "/callAI/<int>")([](int port){
		std::thread::id	parentId = std::this_thread::get_id();
		signal(SIGTERM, handling_sigterm);
		std::thread t([parentId, port](){loop(parentId, port);});
		t.detach();
		return "OK";
	});
}

int main( void ) {
	crow::SimpleApp	app;

	myApp = &app;
	std::cout << "Pid: " << getpid() << std::endl;
	test(app);
    app.port(16016).multithreaded().run();
    return 0;
}
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mainClientWs.cpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/07 09:30:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/07 10:15:29 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <iostream>
#include <thread>
#include <future>


typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

int	main( void ) {
	client	c;
	std::promise<bool>	promise;
	std::future<bool> future = promise.get_future();
	
	c.init_asio();
	c.set_fail_handler([&promise](websocketpp::connection_hdl hdl){
		std::cout << "connection Failed" << std::endl;
		promise.set_value(false);
	});
	c.set_open_handler([&promise](websocketpp::connection_hdl hdl){
		std::cout << "connected" << std::endl;
		promise.set_value(true);
	});
	std::cout << "set handler ok" << std::endl;
	websocketpp::lib::error_code	ec;
	auto server = c.get_connection("ws://0.0.0.0:4040", ec);
	if (ec) {
		std::cout << "pas de co" << std::endl;
		return 1;
	}
	std::cout << "got connection" << std::endl;
	c.connect(server);
	std::thread	t([&c](){c.run();});
	bool success = future.get();
	if (success)
		std::cout << "connection set and ready" << std::endl;
	else
		std::cout << "connection failed" << std::endl;
	t.join();
	return 0;
}
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ia_server.cpp                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 13:17:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/09 14:54:53 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <websocketpp/config/asio_no_tls.hpp>

#include <websocketpp/server.hpp>

#include <iostream>
#include <string>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;
typedef server::connection_ptr connection_ptr;

void	on_open(websocketpp::connection_hdl hdl) {
	std::cout << "Connected " << std::endl;
	return ;
}

void	on_close(websocketpp::connection_hdl hdl) {
	std::cout << "Disconnected " << std::endl;
	return ;
}



int main ( void ) {
	server	myServer;

	try {
		
		myServer.set_access_channels(websocketpp::log::alevel::all);
		myServer.clear_access_channels(websocketpp::log::alevel::frame_payload);
		myServer.set_open_handler(&on_open);
		myServer.set_close_handler(&on_close);


		myServer.init_asio();
		
		myServer.listen(9002);

		myServer.start_accept();
		myServer.run();
	} catch (websocketpp::exception const & e) {
		std::cout << e.what() << std::endl;
	} catch ( ... ) {
		std::cout << "An unexpected exception" << std::endl;
	}
	return 0;
}
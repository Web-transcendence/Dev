/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mainServerWs.cpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/07 09:55:10 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/07 09:58:07 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

# include <websocketpp/config/asio_no_tls.hpp>
# include <websocketpp/server.hpp>
#include <iostream>
#include <thread>


typedef websocketpp::server<websocketpp::config::asio> server;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;
typedef server::connection_ptr connection_ptr;

int	main( void ) {
	server	serv;
	
	serv.set_fail_handler([](websocketpp::connection_hdl hdl){std::cout << "connection Failed" << std::endl;});
	serv.set_open_handler([](websocketpp::connection_hdl hdl){std::cout << "connected" << std::endl;});
	serv.init_asio();
	
	serv.listen(4040);

	serv.start_accept();
	std::cout << "Server ready on 4040" << std::endl;
	serv.run();
}
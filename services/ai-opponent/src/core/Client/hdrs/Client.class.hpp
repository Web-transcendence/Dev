/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/14 09:46:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/20 16:27:14 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CLIENT_CLASS_HPP
# define CLIENT_CLASS_HPP

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include "../../third-party/json/json.hpp"

#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

class Client {
public:
	Client(std::string const & urlGame);
	~Client( void );

	void	stop( void );
	void	run( void );

	bool	getActive( void );

private:
	Client( void );
	
	void	on_message(websocketpp::connection_hdl hdl, client::message_ptr msg);
	void	on_message_aiServer(nlohmann::json const & data);
	void	on_message_gameServer(nlohmann::json const & data);
	
	void	loop( void );
	
	bool	checkTime( void );

	client	c;

	server_ptr	aiServer;
	server_ptr	gameServer;

	std::mutex			cgMutex;
	std::vector<double>	currentGameState;

	std::atomic<bool>	active;

	std::atomic<std::chrono::steady_clock::time_point>	t1;
};


#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/14 09:46:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/23 10:00:41 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CLIENT_CLASS_HPP
# define CLIENT_CLASS_HPP

#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_STL_
#define _WEBSOCKETPP_CPP11_FUNCTIONAL_
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include "Environment.class.hpp"
#include "httplib.h"
#include "json.hpp"

#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <array>
#include <future>

typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

class Client {
public:
	Client(int const gameId);
	~Client( void );

	void	stop( void );
	void	run( void );

	t_state	getActive( void );

private:
	Client( void );


	
	void	on_message(websocketpp::connection_hdl hdl, client::message_ptr msg);
	void	on_message_aiServer(nlohmann::json const & data);
	void	on_message_gameServer(nlohmann::json const & data);
	void	resetEnv(nlohmann::json const & data);
	
	void	loop( void );
	
	bool	giveArrow(std::string const & key, nlohmann::json & j);

	bool	checkTime( void );

	int const	gameId;

	client	c;

	server_ptr	aiServer;
	server_ptr	gameServer;
	
	Environment	localPong;
	
	httplib::Client	factoryServer;

	std::string	lastKey;
	
	std::mutex	stateMutex;

	std::atomic<t_state>	active;

	std::promise<bool>	promiseGS;
	std::promise<bool>	promiseAI;
	std::promise<bool>	promiseGame;

	std::atomic<std::chrono::steady_clock::time_point>	t1;

	std::array<std::string, 3> const	allInput;

	std::mutex	logMutex;
};


#endif

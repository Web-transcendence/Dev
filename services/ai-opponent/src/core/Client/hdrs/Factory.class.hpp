/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Factory.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/14 09:47:28 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/21 13:24:02 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FACTORY_CLASS_HPP
# define FACTORY_CLASS_HPP

#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_STL_
#define _WEBSOCKETPP_CPP11_FUNCTIONAL_
#define MAX_CLIENTS 50

#include "TypeDefinition.hpp"

#include "json.hpp"
#include "crow.h"

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <queue>
#include <map>


typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

class Client;

class Factory {
public:
	Factory( void );
	~Factory( void );

	void	run();

private:
	void	createGame(int const gameId);
	void	deleteGame(int const gameId);

	void	settlingMessage(unsigned int const sizePool);

	crow::SimpleApp	app;

	std::string const	_gameServerWs;
	
	std::mutex				_mMutex;
	std::queue<std::string>	_messages;

	std::map<int, std::shared_ptr<Client>>	_connectedClients;

	std::mutex	ccMutex;
	std::mutex	sendMutex;
}; 

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Factory.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/14 09:47:28 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/05 11:16:46 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FACTORY_CLASS_HPP
# define FACTORY_CLASS_HPP

#define MAX_CLIENTS 50

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include "json.hpp"

#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <queue>
#include <map>


typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

class Client;

class Factory {
public:
	Factory(std::string const & serverWs);
	~Factory( void );

	void	run( void );

private:
	Factory( void ) {}
	
	void	on_message(websocketpp::connection_hdl hdl, client::message_ptr msg);

	void	createGame(std::string const & ws);
	void	deleteGame(std::string const & ws);

	void	settlingMessage(unsigned int const sizePool);

	client		myFactory;
	server_ptr	gameServer;
	
	std::mutex						_mMutex;
	std::queue<client::message_ptr>	_messages;

	std::map<std::string, std::shared_ptr<Client>>	_connectedClients;
	
	std::mutex	ccMutex;
	std::mutex	sendMutex;
}; 

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Factory.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/18 16:55:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/23 15:58:13 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Factory.class.hpp"

#include "Client.class.hpp"

Factory::Factory(std::string const & serverWs) {
	this->myFactory.init_asio();
	this->myFactory.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg) {this->on_message(hdl, msg);});
	websocketpp::lib::error_code	ec;
	this->gameServer = this->myFactory.get_connection(serverWs, ec);
	if (ec) {
		std::cout << "Error: " << ec.message() << std::endl;
		throw std::exception();
	}
	this->myFactory.connect(this->gameServer);
	return ;
}

Factory::~Factory( void ) {
	return ;
}

void	Factory::run( void ) {
	std::thread	t([this](){this->myFactory.run();});
	while (true) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		this->_mMutex.lock();
		unsigned int const	sizeMessagesPool = this->_messages.size();
		this->_mMutex.unlock();
		if (!sizeMessagesPool)
			continue;
		try {
			this->settlingMessage(sizeMessagesPool);
		} catch(std::exception & e) {
			std::cout << "Error: " << e.what() << std::endl;
		}
	}
	this->myFactory.stop();
	return ;
}

void	Factory::settlingMessage(unsigned int const sizePool) {
	for (unsigned int settled = 0; settled < sizePool; settled++) {
		this->_mMutex.lock();
		auto	data = nlohmann::json::parse(this->_messages.front()->get_payload());
		this->_messages.pop();
		this->_mMutex.unlock();
		if (data["ID"] != "Game")
			throw std::exception(); // unrecognized token
		if (data["state"] == "start")
			this->createGame(data["ws"]);
		else if (data["state"] == "stop")
			this->deleteGame(data["ws"]);
		else
			throw std::exception();
	}
}

void	Factory::createGame(std::string const & ws) {
	try {
		if (this->_connectedClients.size() == MAX_CLIENTS)
			throw std::exception();
		auto	c = std::make_shared<Client>(ws);
		if (this->_connectedClients[ws])
			throw std::exception(); // duplicate ws
		this->_connectedClients[ws] = c;
		std::thread	t([c]() {c->run();});
		t.detach();
	} catch (std::exception const & e) {
		std::cout << e.what() << std::endl; // instantiation failed
	}
	return ;
}

void	Factory::deleteGame(std::string const & ws) {
	auto	currentClient = this->_connectedClients.find(ws);
	if (currentClient == this->_connectedClients.end())
		throw std::exception(); // client not found
	if (!currentClient->second->getActive())
		this->_connectedClients.erase(currentClient); // voir avec ben le handle de la connection/ deco peut etre different
	return ;
}

void	Factory::on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
	// auto	data = nlohmann::json::parse(msg->get_payload());
	
	(void)hdl;
	this->_mMutex.lock();
	this->_messages.push(msg);
	this->_mMutex.unlock();
}


/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Factory.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/18 16:55:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/06 13:57:59 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Factory.class.hpp"

#include "Client.class.hpp"

#include <csignal>

Factory::Factory(std::string const & serverWs) : _gameServerWs(serverWs) {
	CROW_ROUTE(this->app, "/createAI/<int>")([this](int gameId){
		nlohmann::json	j;

		j["state"] = "create";
		j["id"] = gameId;
		this->_mMutex.lock();
		this->_messages.push(j.dump());
		this->_mMutex.unlock();
		return "OK";
	});

	CROW_ROUTE(this->app, "/deleteAI/<int>")([this](int gameId){
		nlohmann::json	j;
		
		j["state"] = "delete";
		j["id"] = gameId;
		this->_mMutex.lock();
		this->_messages.push(j.dump());
		this->_mMutex.unlock();
		return "OK";
	});
	
}

// Factory::Factory(std::string const & serverWs) {
// 	this->myFactory.init_asio();
// 	this->myFactory.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg) {this->on_message(hdl, msg);});
// 	websocketpp::lib::error_code	ec;
// 	this->gameServer = this->myFactory.get_connection(serverWs, ec);
// 	if (ec) {
// 		std::cout << "Error: " << ec.message() << std::endl;
// 		throw std::exception();
// 	}
// 	this->myFactory.connect(this->gameServer);
// 	return ;
// }

Factory::~Factory( void ) {
	return ;
}

void	Factory::run(int const port) {
	std::thread	t([this](){
		while (shouldStop.load() == false) {
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
		this->app.stop();
	});
	t.detach();
	this->app.port(port).multithreaded().run();
	return ;
}

void	Factory::settlingMessage(unsigned int const sizePool) {
	for (unsigned int settled = 0; settled < sizePool;) {
		if (this->_connectedClients.size() == MAX_CLIENTS) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			continue;
		}
		this->_mMutex.lock();
		auto	message = this->_messages.front();
		this->_messages.pop();
		this->_mMutex.unlock();
		auto	data = nlohmann::json::parse(message);
		if (data["state"] == "create")	
			this->createGame(data["id"]);
		else if (data["state"] == "delete")
			this->deleteGame(data["id"]);
		else
			throw std::exception(); // unknow token
		++settled;
	}
}

void	Factory::createGame(int const gameId) {
	try {
		auto	c = std::make_shared<Client>(gameId);
		if (this->_connectedClients[gameId])
			throw std::exception(); // duplicate game
		this->_connectedClients[gameId] = c;
		std::thread	t([c]() {c->run();});
		t.detach();
	} catch (std::exception const & e) {
		std::cout << e.what() << std::endl; // instantiation failed
	}
	return ;
}

void	Factory::deleteGame(int const gameId) {
	auto	currentClient = this->_connectedClients.find(gameId);
	if (currentClient == this->_connectedClients.end())
		throw std::exception(); // client not found
	if (!currentClient->second->getActive())
		this->_connectedClients.erase(currentClient);
	return ;
}

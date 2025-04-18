/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Factory.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/18 16:55:07 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/18 19:28:46 by thibaud          ###   ########.fr       */
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

void	Factory::on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
	auto	data = nlohmann::json::parse(msg->get_payload());
	
	if (data["ID"] != "Game")
		throw std::exception(); // unrecognized token
	if (data["state"] == "start")
		this->createGame(data["ws"]);
	else if (data["state"] == "stop")
		this->deleteGame(data["ws"]);
	else
		throw std::exception(); // unrecognized token

}

void	Factory::createGame(std::string const & ws) {
	nlohmann::json	j;

	j["ID"] = "Factory";
	try {
		this->ccMutex.lock();
		if (this->_connectedClients.size() == MAX_CLIENTS)
			throw std::exception();
		this->ccMutex.unlock();
		auto	c = std::make_shared<Client>(ws);
		this->ccMutex.lock();
		if (this->_connectedClients[ws])
			throw std::exception(); // duplicate ws
		this->_connectedClients[ws] = c;
		this->ccMutex.unlock();
		std::thread	t([c]() {c->run();});
		t.detach();
		j["state"] = "OK";
	} catch (std::exception const & e) {
		std::cout << e.what() << std::endl; // instantiation failed
		j["state"] = "NOK";
	}
	this->sendMutex.lock();
	this->gameServer->send(j.dump());
	this->sendMutex.unlock();
}


void	Factory::run( void ) {
	while (true) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return ;
}

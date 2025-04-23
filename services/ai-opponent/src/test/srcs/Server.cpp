/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Server.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/22 16:34:19 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/23 15:07:02 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Server.hpp"

#include <iostream>
#include <sstream>
#include <chrono>

Server::Server(uint16_t port, std::string const & name) {
	this->nameServer = name;
	this->myServer.set_access_channels(websocketpp::log::alevel::all);
	this->myServer.clear_access_channels(websocketpp::log::alevel::frame_payload);

	this->myServer.set_open_handler([this](websocketpp::connection_hdl hdl) {this->on_open(hdl);});
	this->myServer.set_close_handler([this](websocketpp::connection_hdl hdl) {this->on_close(hdl);});
	this->myServer.set_message_handler([this](websocketpp::connection_hdl hdl, message_ptr msg) {this->on_message(hdl, msg); }) ;
	this->myServer.init_asio();
	this->myServer.listen(port);
	this->myServer.start_accept();
	std::stringstream	ss;
	ss << "ws://127.0.0.1:" << port;
	this->ws = ss.str();
	this->hdlFactory = NULL;
	this->connected.store(false);
	std::cout << this->nameServer << " open on: " << this->ws << std::endl;
}

void Server::on_open(websocketpp::connection_hdl hdl) {
	std::cout << "Connection: ";
	this->hdlFactory = this->myServer.get_con_from_hdl(hdl);
	this->connected.store(true);
	std::cout << "Factory connected" << std::endl;
	return ;
}

void	Server::on_close(websocketpp::connection_hdl hdl) {
	std::cout << "Disconnection";
}

void	Server::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
	hdl.reset();
	std::cout << "The game center should not received message, connection closed" << std::endl;
}

void	Server::request(std::string ws) {
	nlohmann::json	j;
	j["ID"] = "Game";
	j["state"] = "start";
	j["ws"] = ws;
	this->myServer.send(this->hdlFactory, j.dump(), websocketpp::frame::opcode::text);
	return ;
}

void	Server::run(void) {this->myServer.run();}
void	Server::stop(void) {this->myServer.stop();}

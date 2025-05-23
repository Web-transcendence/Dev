/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AiServer.cpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 13:17:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/23 00:12:50 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "AiServer.class.hpp"

#include "Network.class.hpp"
#include "json.hpp"

#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>

AiServer::AiServer(std::string const & QNetConfigFile) : _QNet(Network(QNetConfigFile)) {
	return ;
}

AiServer::~AiServer( void ) {
	return ;
}

void	AiServer::start( void ) {
	this->_myServer.set_access_channels(websocketpp::log::alevel::all);
	this->_myServer.clear_access_channels(websocketpp::log::alevel::all);
	this->_myServer.clear_error_channels(websocketpp::log::elevel::all);

	this->_myServer.set_message_handler([this](websocketpp::connection_hdl hdl, message_ptr msg) {this->on_message(hdl, msg);});
	
	this->_myServer.init_asio();
	
	this->_myServer.listen(AI_SERVER_PORT);

	this->_myServer.start_accept();

	std::cout << "AI server running on ws://0.0.0.0:9090" << std::endl;

	this->_myServer.run();
	return ;
}

void	AiServer::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
	double	_1[sizeof(double)*N_NEURON_INPUT];
	memcpy(_1, msg->get_payload().c_str(), sizeof(double)*N_NEURON_INPUT);
	auto	input = std::vector<double>(_1, _1+N_NEURON_INPUT);
	auto	oQNet = this->_QNet.feedForward(input);
	nlohmann::json	j;
	j["type"] = "ai";
	j["data"] = oQNet;
	this->_myServer.send(hdl, j.dump(), websocketpp::frame::opcode::text);
	return ;
}

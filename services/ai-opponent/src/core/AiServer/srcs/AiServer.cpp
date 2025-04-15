/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AiServer.cpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 13:17:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/15 16:08:17 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "AiServer.class.hpp"

#include "Network.class.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

double	_1[sizeof(double)*16]; //place holder input
char	_2[sizeof(double)*4]; // place holder output

AiServer::AiServer(std::string const & QNetConfigFile) : _QNet(Network(QNetConfigFile)) {
		return ;
}

AiServer::~AiServer( void ) {
	return ;
}

void	AiServer::start( void ) {
	this->_myServer.set_access_channels(websocketpp::log::alevel::all);
	this->_myServer.clear_access_channels(websocketpp::log::alevel::frame_payload);
	
	this->_myServer.set_open_handler([this](websocketpp::connection_hdl hdl) {this->on_open(hdl);});
	this->_myServer.set_close_handler([this](websocketpp::connection_hdl hdl) {this->on_close(hdl);});
	this->_myServer.set_message_handler([this](websocketpp::connection_hdl hdl, message_ptr msg) {this->on_message(hdl, msg);});
	
	this->_myServer.init_asio();
	
	this->_myServer.listen(9002);

	this->_myServer.start_accept();
	this->_myServer.run();
	return ;
}

void	AiServer::on_open(websocketpp::connection_hdl hdl) {
	std::cout << "Connected " << std::endl;
	return ;
}

void	AiServer::on_close(websocketpp::connection_hdl hdl) {
	std::cout << "Disconnected " << std::endl;
	return ;
}

void	AiServer::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
	std::cout << "Header: " << msg->get_header() << std::endl;
	memcpy(_1, msg->get_payload().c_str(), sizeof(double)*16);
	auto	input = std::vector<double>(_1, _1+16);
	auto	oQNet = this->_QNet.feedForward(input);
	memcpy(_2, oQNet.data(), sizeof(double)*4);
	this->_myServer.send(hdl, _2, websocketpp::frame::opcode::binary);
	return ;
}

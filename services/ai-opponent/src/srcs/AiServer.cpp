/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AiServer.cpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/09 13:17:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/11 14:03:25 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "AiServer.class.hpp"

#include "Network.class.hpp"

#include <iostream>
#include <string>

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
	if (msg->get_payload().size() != N_NEURON_INPUT * sizeof(double))	{
		std::cout << "error: " << hdl.lock().get() << " send " << msg->get_payload() <<\
		": Not a correct message" << std::endl;
		return ;
	}
	std::cout << "input raw: " << msg->get_payload() << std::endl;
	char*		dup = strdup(msg->get_payload().c_str());
	double*		myDouble = reinterpret_cast<double*>(dup);
	auto		input = std::vector<double>(N_NEURON_INPUT);
	std::cout << "input: ";
	int	i = 0;
	for (auto it = input.begin(); it != input.end(); it++) {
		*it = myDouble[i];
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	auto		oQNet = this->_QNet.feedForward(input);
	std::string	out(reinterpret_cast<char const *>(oQNet.data()), oQNet.size() * sizeof(double));
	this->_myServer.send(hdl, out, websocketpp::frame::opcode::binary);
	return ;
}

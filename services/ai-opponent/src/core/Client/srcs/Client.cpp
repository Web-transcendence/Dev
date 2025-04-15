/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/15 14:55:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/15 22:23:28 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"

#include "TypeDefinition.hpp"

#include <chrono>
#include <sstream>
#include <exception>

char	_1[sizeof(double)*N_NEURON_INPUT]; //place holder input
double	_2[sizeof(double)*N_NEURON_OUTPUT]; //place holder output

Client::Client(std::string const & urlGame) {
	this->currentGameState = std::vector<double>(N_NEURON_INPUT);
	
	this->c.init_asio();
	this->c.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg){this->on_message(hdl, msg);});
	websocketpp::lib::error_code	ec;
	this->aiServer = c.get_connection("ws://localhost:9002", ec);
	this->gameServer = c.get_connection(urlGame, ec);
	if (ec) {
		std::cout << "Error: " << ec.message() << std::endl;
		throw std::exception();
	}
	c.connect(this->aiServer);
	c.connect(this->gameServer);
	return ;
}

Client::~Client( void ) {
	return ;
}

void	Client::on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
	auto	data = nlohmann::json::parse(msg->get_payload());
	auto	data = nlohmann::json::parse(msg->get_payload());
	if (data["ID"] == "Game")
		this->aiServer->send(msg);
	else if (data["ID"] == "AI")
		this->on_message_aiServer(data);
	return ;
}

void	Client::on_message_aiServer(nlohmann::json_abi_v3_12_0::json data) {
	unsigned int		idx = 0;
	std::vector<double>	output(N_NEURON_OUTPUT);

	for (auto it = output.begin(); it != output.end(); it++) {
		std::stringstream	ss;
		ss << data["Data"][idx];
		ss >> *it;
	}
	std::stringstream	ssKey;
	// definir l input grace a la plus grande valeur contenue dans output
	this->gameServer->send(ssKey.str());
	return ;
}

void	Client::on_message_gameServer(nlohmann::json_abi_v3_12_0::json data) {
	unsigned int	idx = 0;
	cgMutex.lock();
	for (auto it = this->currentGameState.begin(); it != this->currentGameState.end(); it++, idx++) {
		std::stringstream	ss;
		ss << data["Data"][idx];
		ss >> *it;
	}
	cgMutex.unlock();
	return ;
}

void	Client::loop( void ) {
	// resortir un nouvel input tous les 100ms
	while (1) {
		cgMutex.lock();
		memcpy(_1, this->currentGameState.data(), sizeof(double)*N_NEURON_INPUT);
		cgMutex.unlock();
		this->aiServer->send(_1, sizeof(double)*N_NEURON_INPUT);
		std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
	}
}

void	Client::run( void ) {this->c.run();}
void	Client::stop( void ) {this->c.stop();}
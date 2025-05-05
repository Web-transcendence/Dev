/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/15 14:55:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/05 13:15:14 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"

#include "TypeDefinition.hpp"

#include <sstream>
#include <exception>
#include <ratio>
#include <ctime>

char	_1[sizeof(double)*N_NEURON_INPUT]; //place holder input
double	_2[sizeof(double)*N_NEURON_OUTPUT]; //place holder output

Client::Client(std::string const & urlGame) : allInput(std::array<std::string, 3>{UP, DOWN, NOTHING}) {
	this->currentGameState = std::vector<double>(N_NEURON_INPUT);
	
	this->c.init_asio();
	this->c.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg){this->on_message(hdl, msg);});
	websocketpp::lib::error_code	ec;
	this->aiServer = this->c.get_connection("ws://localhost:9002", ec);
	this->gameServer = this->c.get_connection(urlGame, ec);
	if (ec) {
		std::cout << "Error: " << ec.message() << std::endl;
		throw std::exception();
	}
	this->c.connect(this->aiServer);
	this->c.connect(this->gameServer);
	lastKey = NOTHING;
	return ;
}

Client::~Client( void ) {
	return ;
}

void	Client::on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
	auto	data = nlohmann::json::parse(msg->get_payload());
	if (data["ID"] == "Game")
		this->on_message_gameServer(data); // pb ? 
	else if (data["ID"] == "AI")
		this->on_message_aiServer(data);
	(void)hdl;
	return ;
}

void	Client::on_message_aiServer(nlohmann::json const & data) {
	unsigned int		idx = 0;
	std::vector<double>	o(N_NEURON_OUTPUT);

	for (auto it = o.begin(); it != o.end(); it++) {
		std::stringstream	ss;
		ss << data["Data"][idx];
		ss >> *it;
	}
	nlohmann::json	j;
	int const	key = std::distance(o.begin(), std::max_element(o.begin(), o.end()));
	j["type"] = "input";
	bool const	send = this->giveArrow(this->allInput.at(key), j);
	if (send == true)
		this->gameServer->send(j.dump());
	return ;
}

void	Client::on_message_gameServer(nlohmann::json const & data) {
	unsigned int		idx = 0;
	std::vector<double>	temp(N_NEURON_INPUT);

	for (auto it = temp.begin(); it != temp.end(); it++, idx++)
		*it = data["Data"][idx];
	cgMutex.lock();
	memcpy(this->currentGameState.data(), temp.data(), sizeof(double)*N_NEURON_INPUT);
	cgMutex.unlock();
	this->t1 = std::chrono::steady_clock::now();
	return ;
}

void	Client::loop( void ) {
	// resortir un nouvel input tous les 100ms
	while (1) {
		cgMutex.lock();
		memcpy(_1, this->currentGameState.data(), sizeof(double)*N_NEURON_INPUT);
		cgMutex.unlock();
		this->aiServer->send(_1, sizeof(double)*N_NEURON_INPUT);
		std::this_thread::sleep_for(std::chrono::milliseconds(CLIENT_INPUT_TIME_SPAN));
		if (!checkTime()) {
			this->c.stop();
			this->active.store(false);
			break ;
		}
	}
	return ;
}

bool	Client::giveArrow(std::string const & key, nlohmann::json & j) {
	bool	send = false;

	if (key != this->lastKey) {
		j["key"] = lastKey;
		j["state"] = RELEASE;
		if (key != RELEASE) {	
			this->gameServer->send(j.dump());
			j["key"] = key;
			j["state"] = PRESS;
		}
		send = true;
	}
	this->lastKey = key;
	return send;
}

bool	Client::checkTime( void ) {
	auto	t2 = std::chrono::steady_clock::now();
	auto	timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - this->t1.load());
	if (timeSpan.count() >= CLIENT_MAX_SPAN_STATE)
		return false;
	return true;
}

bool	Client::getActive( void ) {return this->active.load();}

void	Client::run( void ) {
	this->active.store(true);
	std::thread	t([this](){this->c.run();});
	t.detach();
	this->loop();
	return ;
}

void	Client::stop( void ) {this->c.stop();}
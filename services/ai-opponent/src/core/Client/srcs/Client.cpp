/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/15 14:55:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/19 22:16:31 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"

#include "TypeDefinition.hpp"

#include <exception>
#include <sstream>
#include <ratio>
#include <ctime>

Client::Client(std::string const & wsGameServer, int const gameId) :\
		gameId(gameId),\
		factoryServer("http://0.0.0.0:16016"),\
		allInput(std::array<std::string, 3>{UP, DOWN, NOTHING}) {
	auto res = this->factoryServer.Get("/ping");
	if (!res)
		throw DisconnectedFactoryException();
	this->c.init_asio();
	this->c.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg){this->on_message(hdl, msg);});
	this->c.set_fail_handler([this]([[maybe_unused]]websocketpp::connection_hdl hdl){if (!this->promiseSet.load())this->promise.set_value(false);});
	this->c.set_open_handler([this]([[maybe_unused]]websocketpp::connection_hdl hdl){if (!this->promiseSet.load())this->promise.set_value(true);});
	websocketpp::lib::error_code	ec;
	this->aiServer = this->c.get_connection("ws://0.0.0.0:9090", ec);
	this->gameServer = this->c.get_connection(wsGameServer, ec);
	if (ec) {
		std::cout << "Error: " << ec.message() << std::endl;
		throw WsConnectionException();
	}
	this->c.connect(this->aiServer);
	this->c.connect(this->gameServer);
	lastKey = NOTHING;
	this->promiseSet.store(false);
	return ;
}

Client::~Client( void ) {
	return ;
}

void	Client::on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
	auto	data = nlohmann::json::parse(msg->get_payload());
	if (data["source"] == "ai") {
		std::cout << "ai" << std::endl;
		this->on_message_aiServer(data);
	}
	else if (data["source"] == "game") {
		std::cout << "game" << std::endl;
		this->on_message_gameServer(data); 
	}
	(void)hdl;
	return ;
}

void	Client::on_message_aiServer(nlohmann::json const & data) {
	unsigned int		idx = 0;
	std::vector<double>	o(N_NEURON_OUTPUT);

	for (auto it = o.begin(); it != o.end(); it++) {
		std::stringstream	ss;
		ss << data["data"][idx];
		ss >> *it;
	}
	nlohmann::json	j;
	int const	key = std::distance(o.begin(), std::max_element(o.begin(), o.end()));
	this->stateMutex.lock();
	this->localPong.action(key);
	this->stateMutex.unlock();
	j["type"] = "input";
	bool const	send = this->giveArrow(this->allInput.at(key), j);
	if (send == true)
		this->gameServer->send(j.dump());
	return ;
}

void	Client::on_message_gameServer(nlohmann::json const & data) {
	std::vector<double>	temp(N_RAW_STATE);

	t_ball	ball(std::array<double, 6>{\
		data["ball"]["x"],\
		data["ball"]["y"],\
		data["ball"]["angle"],\
		data["ball"]["speed"],\
		data["ball"]["ispeed"],\
		data["ball"]["radius"]
	});
	t_paddle rPaddle(std::array<double, 5>{\
		data["paddle1"]["x"],\
		data["paddle1"]["x"],\
		data["paddle1"]["width"],\
		data["paddle1"]["height"],\
		data["paddle1"]["speed"]
	});
	t_paddle lPaddle(std::array<double, 5>{\
		data["paddle2"]["x"],\
		data["paddle2"]["x"],\
		data["paddle2"]["width"],\
		data["paddle2"]["height"],\
		data["paddle2"]["speed"]
	});
	this->stateMutex.lock();
	this->localPong.reset(ball, lPaddle, rPaddle);
	this->stateMutex.unlock();
	this->t1.store(std::chrono::steady_clock::now());
	return ;
}

void	Client::loop( void ) {
	this->t1.store(std::chrono::steady_clock::now());
	while (this->active.load() == true) {
		this->stateMutex.lock();
		auto	input = this->localPong.getState();
		this->stateMutex.unlock();
		if (!input)
			return ;
		this->aiServer->send(input->data(), sizeof(double)*N_NEURON_INPUT);
		if (!checkTime())
			this->active.store(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
	std::cout << timeSpan.count() << std::endl;
	if (timeSpan.count() >= CLIENT_MAX_SPAN_STATE)
		return false;
	return true;
}

bool	Client::getActive( void ) {return this->active.load();}

void	Client::run( void ) {
	std::future<bool>	future = this->promise.get_future();
	this->active.store(true);
	std::thread	t([this](){this->c.run();});
	t.detach();
	bool	success = future.get();
	this->promiseSet.store(true);
	if (success)
		this->loop();
	this->stop();
	this->factoryServer.Get("/deleteAI/"+this->gameId);
	return ;
}

void	Client::stop( void ) {this->c.stop();}

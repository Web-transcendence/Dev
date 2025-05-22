/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/15 14:55:53 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/23 00:22:54 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Client.class.hpp"

#include "TypeDefinition.hpp"
#include "Debug.namespace.hpp"

#include <exception>
#include <sstream>
#include <ratio>
#include <ctime>

Client::Client(int const gameId) :\
		gameId(gameId),\
		factoryServer(FACTORY_SERVER_ADDRESS),\
		allInput(std::array<std::string, 3>{UP, DOWN, NOTHING}) {
	auto res = this->factoryServer.Get("/ping");
	if (!res)
		throw DisconnectedFactoryException();
	this->c.clear_access_channels(websocketpp::log::alevel::all);
	this->c.clear_error_channels(websocketpp::log::elevel::all);		
	this->c.init_asio();
	this->c.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg){this->on_message(hdl, msg);});
	this->c.set_fail_handler([this]([[maybe_unused]]websocketpp::connection_hdl hdl){
		auto	con = this->c.get_con_from_hdl(hdl);
		Debug::consoleLog("failed connection to: "+con.get()->get_uri()->str(), this->gameId, this->logMutex);
		if (con.get()->get_uri()->str() == GAME_SERVER_ADDRESS) {
			Debug::consoleLog("Game server connection failed", this->gameId, this->logMutex);
			this->promiseGS.set_value(false);
		}
		else if (con.get()->get_uri()->str() == AI_SERVER_ADDRESS) {
			Debug::consoleLog("AI server connection failed", this->gameId, this->logMutex);
			this->promiseAI.set_value(false);
		}
	});
	this->c.set_open_handler([this]([[maybe_unused]]websocketpp::connection_hdl hdl){
		auto	con = this->c.get_con_from_hdl(hdl);
		Debug::consoleLog("new connection from: "+con.get()->get_uri()->str(), this->gameId, this->logMutex);
		if (con.get()->get_uri()->str() == GAME_SERVER_ADDRESS) {
			Debug::consoleLog("Game server connection etablished", this->gameId, this->logMutex);
			this->promiseGS.set_value(true);
		}
		else if (con.get()->get_uri()->str() == AI_SERVER_ADDRESS) {
			Debug::consoleLog("AI server connection etablished", this->gameId, this->logMutex);
			this->promiseAI.set_value(true);
		}
	});
	websocketpp::lib::error_code	ec;
	this->aiServer = this->c.get_connection(AI_SERVER_ADDRESS, ec);
	this->gameServer = this->c.get_connection(GAME_SERVER_ADDRESS, ec);
	if (ec) {
		std::cout << "Error: " << ec.message() << std::endl;
		throw WsConnectionException();
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
	if (this->c.get_con_from_hdl(hdl).get()->get_uri()->str() == AI_SERVER_ADDRESS) {
		auto	data = nlohmann::json::parse(msg->get_payload());
		this->on_message_aiServer(data);
	}
	else if (this->c.get_con_from_hdl(hdl).get()->get_uri()->str() == GAME_SERVER_ADDRESS) {
		auto	data = nlohmann::json::parse(msg->get_payload());
		this->on_message_gameServer(data);
	}
	else
		return ;
	if (this->active.load() == WAITING) {
		this->active.store(ON_GOING);
		this->promiseGame.set_value(true);
	}
	return ;
}

void	Client::on_message_aiServer(nlohmann::json const & data) {
	unsigned int		idx = 0;
	std::vector<double>	o(N_NEURON_OUTPUT);

	for (auto it = o.begin(); it != o.end(); it++, idx++) {
		std::stringstream	ss;
		ss << data["data"][idx];
		ss >> *it;
	}
	int const	key = std::distance(o.begin(), std::max_element(o.begin(), o.end()));
	this->stateMutex.lock();
	this->localPong.action(key);
	this->stateMutex.unlock();
	nlohmann::json	j;
	j["type"] = "input";
	bool const	send = this->giveArrow(this->allInput.at(key), j);
	if (send == true) {
		this->gameServer->send(j.dump());
	}
	return ;
}

void	Client::on_message_gameServer(nlohmann::json const & extData) {
	nlohmann::json	data = extData;
	t_ball	ball(std::array<double, 6>{\
		data["ball"]["x"],\
		data["ball"]["y"],\
		data["ball"]["angle"],\
		data["ball"]["speed"],\
		data["ball"]["ispeed"],\
		data["ball"]["radius"],
	});
	t_paddle rPaddle(std::array<double, 5>{\
		data["paddle2"]["x"],\
		data["paddle2"]["y"],\
		data["paddle2"]["width"],\
		data["paddle2"]["height"],\
		data["paddle2"]["speed"]
	});
	t_paddle lPaddle(std::array<double, 5>{\
		data["paddle1"]["x"],\
		data["paddle1"]["y"],\
		data["paddle1"]["width"],\
		data["paddle1"]["height"],\
		data["paddle1"]["speed"]
	});
	this->stateMutex.lock();
	this->localPong.reset(ball, lPaddle, rPaddle);
	this->stateMutex.unlock();
	this->t1.store(std::chrono::steady_clock::now());
	return ;
}

void	Client::loop( void ) {
	auto	futureStart = this->promiseGame.get_future();
	
	futureStart.get();
	this->t1.store(std::chrono::steady_clock::now());
	while (this->active.load() == ON_GOING) {
		this->stateMutex.lock();
		auto	input = this->localPong.getState();
		this->stateMutex.unlock();
		this->aiServer->send(input->data(), sizeof(double)*N_NEURON_INPUT);
		if (!checkTime())
			this->active.store(FINISHED);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	return ;
}

bool	Client::giveArrow(std::string const & key, nlohmann::json & j) {
	bool	send = false;

	if (key != this->lastKey) {
		j["key"] = lastKey;
		j["state"] = RELEASE;
		if (key != NOTHING) {	
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

t_state	Client::getActive( void ) {return this->active.load();}

void	Client::run( void ) {
	std::future<bool>	futureGS = this->promiseGS.get_future();
	std::future<bool>	futureAI = this->promiseAI.get_future();

	this->active.store(WAITING);
	std::thread	t([this](){this->c.run();});
	t.detach();
	bool	successAI = futureAI.get();
	bool	successGS = futureGS.get();
	if (successGS && successAI) {
		nlohmann::json	init;
		init["type"] = "socketInit";
		init["nick"] = "AI";
		init["room"] = this->gameId;
		this->gameServer->send(init.dump(), websocketpp::frame::opcode::text);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		nlohmann::json	ready;
		ready["type"] = "ready";
		ready["mode"] = "remote";
		this->gameServer->send(ready.dump(), websocketpp::frame::opcode::text);
		this->loop();
	}
	this->stop();
	this->factoryServer.Get("/deleteAI/"+this->gameId);
	return ;
}

void	Client::stop( void ) {this->c.stop();}

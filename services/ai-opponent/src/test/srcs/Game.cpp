/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Game.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/22 17:55:39 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/22 19:25:19 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Game.hpp"
#include <random>

Game::Game(uint16_t port, std::string const & name) {
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
	ss << "htpp://localhost:" << port;
	this->ws = ss.str();
	this->connected.store(false);
	std::cout << this->nameServer << " open on: " << this->ws << std::endl;
}

void Game::on_open(websocketpp::connection_hdl hdl) {
	this->hdlClient = hdl;
	this->connected.store(true);
	std::cout << "Client connected" << std::endl;
	return ;
}

void	Game::on_close(websocketpp::connection_hdl hdl) {
	std::cout << "Disconnection";
}

void	Game::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
	auto	data = nlohmann::json::parse(msg->get_payload());

	std::cout << this->nameServer << " receveived: " << data["arrow"] << std::endl;
}

void	Game::stateSendLoop(int const runtime) {
	std::random_device					rd;
	std::mt19937						gen(rd());
	std::uniform_int_distribution<int>	dist(0, 15);
	
	for (int count = 0; count < runtime; count++) {
		std::vector<double>	input(16);
		input.at(dist(gen)) = 1.0;
		nlohmann::json	j;
		j["ID"] = "Game";
		j["Data"] = input;
		this->myServer.send(this->hdlClient, j.dump(), websocketpp::frame::opcode::text);
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

void	Game::run(void) {this->myServer.run();}
void	Game::stop(void) {this->myServer.stop();}
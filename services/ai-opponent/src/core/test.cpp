/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/19 16:09:55 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/20 10:01:59 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

# include "third-party/websocketpp/config/asio_no_tls.hpp"
# include "third-party/websocketpp/server.hpp"
# include "third-party/json.hpp"
# include <httplib.h>
# include <iostream>
# include <thread>
# include <chrono>
# include <sstream>

#define	PORT 6363 

#define	HEIGHT 800
#define WIDTH 1200

typedef websocketpp::server<websocketpp::config::asio> server;

typedef server::message_ptr message_ptr;
typedef server::connection_ptr connection_ptr;

std::vector<websocketpp::connection_hdl>	connected;

void	sendToConnected(websocketpp::connection_hdl & hdl, server & myServ) {
	nlohmann::json	j;
			
	j["source"] = "game";
	j["type"] = "gameUpdate";
	j["paddle1"]["x"] = 30.;
	j["paddle1"]["y"] = HEIGHT / 2.;
	j["paddle1"]["width"] = 20.;
	j["paddle1"]["height"] = 200.;
	j["paddle2"]["speed"] = 0.;
	j["paddle2"]["x"] = WIDTH - 30.;
	j["paddle2"]["y"] = HEIGHT / 2.;
	j["paddle2"]["width"] = 20.;
	j["paddle2"]["height"] = 200.;
	j["paddle2"]["speed"] = 0.;
	j["ball"]["x"] = HEIGHT / 2;
	j["ball"]["y"] = WIDTH / 2;
	j["ball"]["radius"] = 12.;
	j["ball"]["speed"] = 12.;
	j["ball"]["ispeed"] = 12.;
	j["ball"]["angle"] = M_PI;
	myServ.send(hdl, j.dump(), websocketpp::frame::opcode::TEXT);
	std::cout << "log: new message send for " << hdl.lock().get() << std::endl;
	return ;
}


void	initServer(server & myServ) {
	myServ.set_access_channels(websocketpp::log::alevel::all);
	myServ.clear_access_channels(websocketpp::log::alevel::frame_payload);
	
	myServ.set_open_handler([&myServ](websocketpp::connection_hdl hdl){
		std::cout << "log: new client connected" << std::endl;

		connected.push_back(hdl);
	});

	myServ.set_message_handler([](websocketpp::connection_hdl hdl, message_ptr msg) {
		std::cout << "log: new message received: " << msg->get_payload() << std::endl;
	});
	
	myServ.init_asio();
	
	myServ.listen(PORT);

	myServ.start_accept();

	std::thread	t([&myServ]() {myServ.run();});
	t.detach();

	std::cout << "log: server is running on ws://0.0.0.0:" << PORT << std::endl;
}

int	main( void ) {
	server	myServ;

	initServer(myServ);
	
	httplib::Client cli("http://0.0.0.0:16016");

	std::string	myReq[5] = {"/createAI/130", "/createAI/131", "/createAI/132", "/createAI/133", "/createAI/134"};

	for (int i = 0; i < 5; i++) {
		
		auto res = cli.Get(myReq[i]);
		
		if (res && res->status == 200) {
			std::cout << res->body << std::endl;
		}
	}
	
	while (true) {
		for (auto it = connected.begin(); it != connected.end(); it++) {
			sendToConnected(*it, myServ);
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
	};

	return 0;
}
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Server.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/22 16:29:19 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/22 19:14:27 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef	SERVER_HPP
# define SERVER_HPP

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include "../../core/third-party/json/json.hpp"

#include <string>
#include <atomic>

typedef websocketpp::server<websocketpp::config::asio> server;
typedef websocketpp::connection<websocketpp::config::asio>	connection;

typedef server::message_ptr message_ptr;
typedef server::connection_ptr connection_ptr;

class Server {
public:
	Server(uint16_t port, std::string const & name);

	void	run(void);
	void	stop(void);

	void	request(std::string ws);

	bool	isReady(void) {return this->connected.load();}

private:
	void	on_open(websocketpp::connection_hdl hdl);
	void	on_close(websocketpp::connection_hdl hdl);
	void	on_message(websocketpp::connection_hdl hdl, message_ptr msg);

	server						myServer;
	std::string					nameServer;
	std::string					ws;
	std::shared_ptr<connection>	hdlFactory;
	std::atomic<bool>			connected;
};

#endif

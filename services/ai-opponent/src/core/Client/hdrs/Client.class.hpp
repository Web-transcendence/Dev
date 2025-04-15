/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Client.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/14 09:46:47 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/15 20:30:50 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CLIENT_CLASS_HPP
# define CLIENT_CLASS_HPP

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include "../../third-party/json/json.hpp"

#include <vector>

typedef websocketpp::client<websocketpp::config::asio_client>	client;
typedef std::shared_ptr<websocketpp::connection<websocketpp::config::asio_client>>	server_ptr;

class Client {
public:
	Client(std::string const & urlGame);
	~Client( void );

private:
	Client( void );
	
	void	on_message(websocketpp::connection_hdl hdl, client::message_ptr msg);
	void	on_message_aiServer(nlohmann::json_abi_v3_12_0::json data);
	void	on_message_gameServer(nlohmann::json_abi_v3_12_0::json data);

	void	loop( void );
	
	void	run( void );
	void	stop( void );
	
	client	c;

	server_ptr	aiServer;
	server_ptr	gameServer;

	std::mutex			cgMutex;
	std::vector<double>	currentGameState;
};


#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AiServer.class.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/11 10:33:19 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/14 00:38:47 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef	AISERVER_CLASS_HPP
# define AISERVER_CLASS_HPP
# include <map>
# include <string>

#include "Network.class.hpp"

# include <websocketpp/config/asio_no_tls.hpp>
# include <websocketpp/server.hpp>

typedef websocketpp::server<websocketpp::config::asio> server;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;
typedef server::connection_ptr connection_ptr;

// typedef struct	s_client {
	
// }	t_client;

class	AiServer {
public:
	AiServer(std::string const & QNetConfigFile);
	~AiServer( void);

	void	start( void );
private:
	AiServer( void );
	
	server	_myServer;
	Network	_QNet;
	
	void	on_open(websocketpp::connection_hdl hdl);
	void	on_close(websocketpp::connection_hdl hdl);
	void	on_message(websocketpp::connection_hdl hdl, message_ptr msg);
};

#endif

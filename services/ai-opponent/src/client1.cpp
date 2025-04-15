#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
 

typedef websocketpp::client<websocketpp::config::asio_client> client;

char	_1[sizeof(double)*16]; //place holder input
double	_2[sizeof(double)*4]; //place holder output


void	printDouble(double const * d, unsigned int const size) {
	unsigned int	idx = 0;

	std::cout << "My double: ";
	while (idx < size) {
		std::cout << d[idx++] << "; "; 
	}
	std::cout << std::endl;
	return ;
}	

int main( void ) {
	client	c;

	try {
		c.init_asio();
		c.set_message_handler([](websocketpp::connection_hdl, client::message_ptr msg) {
			std::cout << "Header: " << msg->get_header() << std::endl;
			memcpy(_2, msg->get_payload().c_str(), sizeof(double)*4);
			printDouble(_2, 4);
		});
		
		websocketpp::lib::error_code ec;
		auto con = c.get_connection("ws://localhost:9002", ec);
		if (ec) {
			std::cout << "Error: " << ec.message() << std::endl;
			return 1;
		}

		c.connect(con);
		
		std::thread t([&c]() {
			c.run();
		});
		
		std::this_thread::sleep_for(std::chrono::seconds(1));
		double	test[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0};
		memcpy(_1, test, sizeof(double)*16);
		con->send(_1, sizeof(double)*16);
		std::this_thread::sleep_for(std::chrono::seconds(1));
		c.stop();
		t.join();
	} catch (std::exception const & e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}
	return 0;
}
 
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

typedef websocketpp::client<websocketpp::config::asio_client> client;


int main( void ) {
	client	c;

	try {
		c.init_asio();
		c.set_message_handler([](websocketpp::connection_hdl, client::message_ptr msg) {
			char	*dup = strdup(msg->get_payload().c_str());
			double	*rec = reinterpret_cast<double*>(dup);
			std::cout << "Received: " << rec[0] << " " << rec[1] << " " << rec[2] << " " << std::endl;
			free(dup);
		});
		
		websocketpp::lib::error_code ec;
		auto con = c.get_connection("ws://localhost:8080", ec);
		if (ec) {
			std::cout << "Error: " << ec.message() << std::endl;
			return 1;
		}
		
		c.connect(con);
		
		std::thread t([&c]() {
			c.run();
		});
		
		std::this_thread::sleep_for(std::chrono::seconds(1));
		while (1) {
			double test[3] = {0.3,0.3,0.3};
			char	*test_r = reinterpret_cast<char*>(test);
			con->send(test_r, sizeof(test));
			std::this_thread::sleep_for(std::chrono::seconds(3));
		}
		c.stop();
		t.join();
	} catch (std::exception const & e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}
	return 0;
}
 
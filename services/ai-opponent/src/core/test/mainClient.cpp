/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mainClient.cpp                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/07 08:37:59 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/07 08:56:07 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../third-party/httplib.h"
#include <iostream>


int	main( void ) {
	httplib::Client	c("http://0.0.0.0:16016");
		
	
	auto	res = c.Get("/callAI/123");
	if (!res)
		return 1;
	std::cout << "message: " << res->body << std::endl;
	return 0; 
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Debug.namespace.hpp                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/21 11:19:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/21 13:16:31 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef DEBUG_NAMESPACE_HPP
# define DEBUG_NAMESPACE_HPP

#include <string>
#include <chrono>
#include <ctime>
#include <iostream>

namespace Debug {
	
	void	consoleLog(std::string const & log, int const id) {
		auto now = std::chrono::system_clock::now();
		std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		
		std::string time_str = std::ctime(&now_c);
		time_str.pop_back();
	
		std::cout<<"["<< time_str << "][" << id << "] - " << log << " " << std::endl;
		return ;
	}

};

#endif
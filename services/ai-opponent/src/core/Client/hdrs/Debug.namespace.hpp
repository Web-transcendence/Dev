/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Debug.namespace.hpp                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/21 11:19:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/27 10:37:01 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef DEBUG_NAMESPACE_HPP
# define DEBUG_NAMESPACE_HPP

#include <string>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

namespace Debug {
	
	void	consoleLog(std::string const & log, int const id, std::mutex & atomic) {
		auto now = std::chrono::system_clock::now();
		std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		
		std::string time_str = std::ctime(&now_c);
		time_str.pop_back();
	
		atomic.lock();
		std::cout<<"["<< time_str << "][" << id << "] - " << log << " " << std::endl;
		atomic.unlock();
		return ;
	}

	void	displayState(std::vector<float> const & vec) {
		std::cout << "\033[H"; // remet le curseur en haut à gauche
		for (size_t i = 0; i < vec.size(); ++i) {
			if (i % 30 == 0 && i != 0)
				std::cout << '\n';
			std::cout << std::setw(3) << std::fixed << std::setprecision(1) << 0.0;
		}
		std::cout << "\033[H"; // remet le curseur en haut à gauche
		for (auto it = vec.begin(); it != vec.end();) {
			for (int lap = 0; lap < 30; lap++, it++) {
				if (*it != 0.)
					std::cout << "\033[31m"; // rouge
				else
					std::cout << "\033[0m";  // reset couleur
				std::cout << std::setw(3) << std::fixed << std::setprecision(1) << *it;
			}
			std::cout << '\n';
		}
		std::cout << "\033[0m" << std::flush; // reset couleur à la fin
	}

};

#endif
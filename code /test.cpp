/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   test.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/12 10:43:29 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/12 10:43:35 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <iostream>
#include <vector>
#include <random>

int main() {
    const int size = 10; // Number of elements
    std::vector<double> values(size);

    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::normal_distribution<double> dist(0.0, 1.0); // Mean=0, Std=1

    for (double& v : values) {
        v = dist(gen);
    }

    // Print results
    for (double v : values) {
        std::cout << v << " ";
    }
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 13:39:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/01 09:36:30 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Mnist.class.hpp"

int main(void) {
	Mnist	dataset("data/train-images.idx3-ubyte",\
					"data/train-labels.idx1-ubyte",\
					"data/t10k-images.idx3-ubyte",\
					"data/t10k-labels.idx1-ubyte");
	std::vector<unsigned int>	sizes(3);
	sizes[0] = 784;
	sizes[1] = 30;
	sizes[2] = 10; 
	Network	myNetwork(sizes, SIGMOID, SIGMOID);
	myNetwork.SDG(dataset.training, 30, 10, 3.0, &dataset.testing);
	return 0;
}

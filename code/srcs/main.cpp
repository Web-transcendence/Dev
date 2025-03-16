/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 13:39:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/16 14:24:59 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Network.class.hpp"
#include "Mnist.class.hpp"

int main(void) {
	Mnist	dataset("../data/train-images.idx3-ubyte",\
					"../data/train-labels.idx3-ubyte",\
					"../data/t10k-images.idx3-ubyte",\
					"../data/t10k-labels.idx3-ubyte");

	dataset.printFirst();
	return 0;
}
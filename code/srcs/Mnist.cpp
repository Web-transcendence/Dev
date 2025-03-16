/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Mnist.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 13:00:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/16 14:20:02 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Mnist.class.hpp"
#include <iostream>
#include <fstream>

Mnist::Mnist(std::string const & trI, \
            std::string const & trL, \
            std::string const & teI, \
            std::string const & teL) {
    std::cout << "Loading training images..." << std::endl;
    this->trainImages = this->loadImages(trI);
    this->trainLabels = this->loadLabels(trL);
    std::cout << "Loaded " << trainImages->size() << " training images." << std::endl;
    std::cout << "Loading testing images..." << std::endl;
    this->testImages = this->loadImages(teI);
    this->testLabels = this->loadLabels(teL);
    std::cout << "Loaded " << testImages->size() << " testing images." << std::endl;
	return ;
}

Mnist::~Mnist( void ) {
	if (this->trainImages)
		delete this->trainImages;
	if (this->trainLabels)
		delete this->trainLabels;
	if (this->testImages)
		delete this->testImages;
	if (this->testLabels)
		delete this->testLabels;
	return ;
}

std::vector<std::vector<uint8_t>>* Mnist::loadImages(std::string const & filename) const {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    // int32_t magic = readInt(file);
    int32_t numImages = readInt(file);
    int32_t rows = readInt(file);
    int32_t cols = readInt(file);
    int32_t imageSize = rows * cols;
    
    auto images = new std::vector<std::vector<uint8_t>>(numImages, std::vector<uint8_t>(imageSize));
    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), imageSize);
    }
    return images;
}

std::vector<uint8_t>* Mnist::loadLabels(std::string const & filename) const {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    // int32_t magic = readInt(file);
    int32_t numLabels = readInt(file);

    auto labels = new std::vector<uint8_t>(numLabels);
    file.read(reinterpret_cast<char*>(labels->data()), numLabels);

    return labels;
}

void    Mnist::printFirst( void ) {
    std::cout << "First image pixel values: " << std::endl;

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (static_cast<int>(this->trainImages->front()[i * 28 + j]) > 128 ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "First image label: " << static_cast<int>(this->trainLabels->front()) << std::endl;
    return ;
}

int32_t Mnist::readInt(std::ifstream & file) const {
    int32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), sizeof(num));
    return __builtin_bswap32(num);  // Convert from big-endian to little-endian
}
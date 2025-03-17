/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Mnist.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 13:00:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/16 20:07:07 by thibaud          ###   ########.fr       */
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
    this->convert();
    std::cout << "Images and Labels converted, ready to train" << std::endl;
	return ;
}

Mnist::~Mnist( void ) {
	if (this->trainImages) {
        for (auto i : *this->trainImages)
            delete i;
		delete this->trainImages;
    }
	if (this->trainLabels)
		delete this->trainLabels;
	if (this->testImages) {
        for (auto i : *this->testImages)
            delete i;
		delete this->testImages;
    }
	if (this->testLabels)
		delete this->testLabels;
    for (auto t : this->training) {
        delete t;
    }
    for (auto t : this->testing) {
        delete t;
    }
	return ;
}

std::vector<std::vector<uint8_t>*>* Mnist::loadImages(std::string const & filename) const {
    std::ifstream file(filename.c_str(), std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    int32_t magic = readInt(file);
    int32_t numImages = readInt(file);
    int32_t rows = readInt(file);
    int32_t cols = readInt(file);
    int32_t imageSize = rows * cols;
    auto images = new std::vector<std::vector<uint8_t>*>(numImages);
    for (auto it = images->begin(); it != images->end(); it++)
        *it = new std::vector<uint8_t>(imageSize);
    for (auto i : *images)
        file.read(reinterpret_cast<char*>(i->data()), imageSize);
    (void)magic;
    file.close();
    return images;
}

std::vector<uint8_t>* Mnist::loadLabels(std::string const & filename) const {
    std::ifstream file(filename.c_str(), std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    int32_t magic = readInt(file);
    int32_t numLabels = readInt(file);
    auto labels = new std::vector<uint8_t>(numLabels);
    file.read(reinterpret_cast<char*>(labels->data()), numLabels);
    (void)magic;
    file.close();
    return labels;
}


int32_t Mnist::readInt(std::ifstream & file) const {
    int32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), sizeof(num));
    return __builtin_bswap32(num);  // Convert from big-endian to little-endian
}

void    Mnist::printFirst( void ) {
    std::cout << "First image pixel values: " << std::endl;

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (static_cast<int>((*this->trainImages->front())[i * 28 + j]) > 128 ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "First image label: " << static_cast<int>(this->trainLabels->front()) << std::endl;
    return ;
}

void    Mnist::convert( void ) {
    auto    it_i = this->trainImages->begin();
    auto    it_l = this->trainLabels->begin();
    for (;it_i != this->trainImages->end(); it_i++, it_l++) {
        auto temp = new t_tuple;
        for (auto ti : **it_i)
            temp->input.push_back(static_cast<double>(ti));
        temp->expectedOutput[static_cast<int>(*it_l)] = 1.0;
        temp->real = static_cast<int>(*it_l);
        this->training.push_back(temp);
    }
    it_i = this->testImages->begin();
    it_l = this->testLabels->begin();
    for (;it_i != this->testImages->end(); it_i++, it_l++) {
        auto temp = new t_tuple;
        for (auto ti : **it_i)
            temp->input.push_back(static_cast<double>(ti));
        temp->expectedOutput[static_cast<int>(*it_l)] = 1.0;
        temp->real = static_cast<int>(*it_l);
        this->testing.push_back(temp);
    }
    return ;
}
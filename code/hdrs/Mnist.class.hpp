/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Mnist.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/16 12:55:43 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/16 14:19:53 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MNIST_CLASS_HPP
# define MNIST_CLASS_HPP
# include <vector>
# include <string>

class Mnist {
public:
	Mnist(std::string const & ti, std::string const & tl, std::string const & tei, std::string const & tel);
	~Mnist( void );
	
	void    							printFirst( void );

	std::vector<std::vector<uint8_t>>*	trainImages;
	std::vector<uint8_t>*				trainLabels;
	std::vector<std::vector<uint8_t>>*	testImages;
	std::vector<uint8_t>*				testLabels;

private:
	Mnist(void) {}

	std::vector<std::vector<uint8_t>>*	loadImages(std::string const & filename) const;
	std::vector<uint8_t>* 				loadLabels(std::string const & filename) const;
	
	int32_t 							readInt(std::ifstream & file) const;
};



#endif

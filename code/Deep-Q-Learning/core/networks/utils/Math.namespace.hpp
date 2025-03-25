/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Math.namespace.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/14 09:50:32 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/19 11:42:14 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MATH_NAMESPACE_HPP
# define MATH_NAMESPACE_HPP
# include <vector>

#include <iostream>

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;
	int					real;

	s_tuple() : expectedOutput(10, 0.0) {}
}      t_tuple;


namespace Math {
	double								sigmoid(double const z);
	std::vector<double>*				sigmoid(std::vector<double> const & zs);

	double								sigmoidPrime(double const z);
	std::vector<double>*				sigmoidPrime(std::vector<double> const & zs);

	double								cost_derivative(double& output, double& expected);
	std::vector<double>*				cost_derivative(std::vector<double> const & output, std::vector<double> const & expected);
	
	double								dotProduct(std::vector<double> const & v1, std::vector<double> const & v2);
	std::vector<double>*				hadamardProduct(std::vector<double> const & lhs, std::vector<double> const & rhs);
	std::vector<std::vector<double>>*	outerProduct(std::vector<double> const & in, std::vector<double> const & transposed);

	std::vector<std::vector<double>>*	transpose1D(std::vector<double> const & base);
	std::vector<std::vector<double>>*	transpose2D(std::vector<std::vector<double>> const & base);

	template<typename T>
	void	printdebug(T const & truc, std::string const & name) {
		std::cout<<name<<":[";
		for (auto t : truc)
			std::cout<<t<<";";
		std::cout<<"]"<<std::endl;
	}
};

#endif
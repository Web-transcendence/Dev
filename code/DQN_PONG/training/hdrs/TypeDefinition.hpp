/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/28 01:36:48 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define OUTPUT_SIZE 3
# define INPUT_SIZE 2400
# define UP 0
# define DOWN 1
# define NOTHING 2
# include <vector>
# include <array>
# include <cstring>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOIDs, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

typedef struct s_exp {
	std::array<double, 6>	state;
	int						action;	
	double					reward;
	std::array<double, 6>	nextState;
	std::array<double, 6>	prevState;
	bool					done;
	s_exp() : reward(0.), done(false) {}
}	t_exp;

typedef struct  s_tuple {
    std::vector<double>* input;
    std::vector<double>* expectedOutput;
}      t_tuple;

#endif

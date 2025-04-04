/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 21:52:15 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define OUTPUT_SIZE 4
# define INPUT_SIZE 16
# define UP 0
# define DOWN 1
# define RIGHT 2
# define LEFT 3
# define NUM_ACTION 4
# define IN_STATE 1
# include <vector>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOIDs, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

typedef struct s_exp {
	std::vector<double>	state;
	int					action;	
	double				reward;
	std::vector<double>	nextState;
	bool				done;
	s_exp() : state(IN_STATE, 0.0), nextState(IN_STATE, 0.0), done(false) {}
}	t_exp;

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;

	s_tuple() : input(INPUT_SIZE, 0.0),  expectedOutput(OUTPUT_SIZE, 0.0) {}
}      t_tuple;

#endif

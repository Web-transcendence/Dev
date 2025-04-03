/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 18:18:36 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define OUTPUT_SIZE 4
# define INPUT_SIZE 16
# include <vector>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOID, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;
// typedef enum e_action {UP, DOWN} t_action;

typedef struct s_state {
	// double	ballXY[2];
	// double	ballAngle;
	// double	pad;
	// double	OppPad;
	std::vector<double>	allState;
	s_state() : allState(5, 0.0) {}
}	t_state;

// typedef struct s_exp {
// 	t_state*	state;
// 	t_action	action;	
// 	double		reward;
// 	t_state*	nextState;
// 	bool		done;
// }	t_exp;

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;

	s_tuple() : input(INPUT_SIZE, 0.0),  expectedOutput(OUTPUT_SIZE, 0.0) {}
}      t_tuple;

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/11 09:37:14 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define UP 0
# define DOWN 1
# define IN_STATE 16
# define N_LAYER_HIDDEN 1
# define N_NEURON_INPUT 16
# define N_NEURON_OUTPUT 4
# define N_NEURON_HIDDEN 25
# include <vector>
# include <cstring>

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
	s_exp() : state(IN_STATE, 0.0), reward(0.), nextState(IN_STATE, 0.0), done(false) {}
}	t_exp;

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;

	s_tuple() : input(N_NEURON_INPUT, 0.0),  expectedOutput(N_NEURON_OUTPUT, 0.0) {}
}      t_tuple;

#endif

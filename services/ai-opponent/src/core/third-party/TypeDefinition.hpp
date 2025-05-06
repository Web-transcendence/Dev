/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/06 13:36:21 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define UP "w"
# define DOWN "s"
# define NOTHING "n"
# define PRESS "down"
# define RELEASE "release"
# define N_LAYER_HIDDEN 1
# define N_NEURON_INPUT 600
# define N_NEURON_OUTPUT 3
# define N_NEURON_HIDDEN 25
# define AI_SERVER_PORT 9090
# define CLIENT_INPUT_TIME_SPAN 100
# define CLIENT_MAX_SPAN_STATE 3.0
# include <vector>
# include <cstring>
# include <atomic>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOIDs, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

extern std::atomic<bool>	shouldStop{false};

typedef struct s_exp {
	std::vector<double>	state;
	int					action;	
	double				reward;
	std::vector<double>	nextState;
	bool				done;
	s_exp() : state(N_NEURON_INPUT, 0.0), reward(0.), nextState(N_NEURON_INPUT, 0.0), done(false) {}
}	t_exp;

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;

	s_tuple() : input(N_NEURON_INPUT, 0.0),  expectedOutput(N_NEURON_OUTPUT, 0.0) {}
}      t_tuple;

void	handling_SIGTERM(int) {
	shouldStop.store(true);
	return ;
}

#endif

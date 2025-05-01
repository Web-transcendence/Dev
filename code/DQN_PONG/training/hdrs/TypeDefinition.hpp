/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tmouche <tmouche@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/01 20:59:16 by tmouche          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP
# define OUTPUT_SIZE 3
# define INPUT_SIZE 600
# define HIDDEN_SIZE 25
# define N_LAYER_HIDDEN 1
# define UP 0
# define DOWN 1
# define NOTHING 2

#include <cstring>
#include <memory>
#include <vector>
#include <array>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOIDs, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

typedef struct s_exp {
	std::shared_ptr<std::vector<double>>	state;
	std::shared_ptr<std::vector<double>>	nextState;
	int										action;	
	double									reward;
	bool									done;
	s_exp() : reward(0.), done(false) {}
}	t_exp;

typedef struct  s_tuple {
    std::vector<double>* input;
    std::vector<double>* expectedOutput;
}      t_tuple;

#endif

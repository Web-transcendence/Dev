/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   TypeDefinition.hpp                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:15:27 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/27 14:06:26 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TYPEDEFINITION_HPP
# define TYPEDEFINITION_HPP

#define UP "w"
#define DOWN "s"
#define NOTHING "n"
#define PRESS "down"
#define RELEASE "release"

#define L_HIDDEN 1
#define	L_GLOBAL 3
#define N_INPUT 600
#define N_OUTPUT 3
#define N_HIDDEN 25
#define N_GLOBAL (N_INPUT + N_HIDDEN*(L_HIDDEN-1) + N_OUTPUT) 

#define	WEIGHT_FIRST_HIDDEN (N_INPUT * N_HIDDEN)
#define	WEIGHT_HIDDEN (N_HIDDEN * N_HIDDEN)
#define	WEIGHT_OUTPUT (N_HIDDEN * N_OUTPUT)
#define WEIGHT_GLOBAL (WEIGHT_FIRST_HIDDEN + WEIGHT_HIDDEN*(L_HIDDEN-1) + WEIGHT_OUTPUT)

#define	BIAI_FIRST_HIDDEN N_INPUT
#define	BIAI_HIDDEN N_HIDDEN
#define	BIAI_OUTPUT N_OUTPUT
#define BIAI_GLOBAL (BIAI_FIRST_HIDDEN + BIAI_HIDDEN*(L_HIDDEN-1) + BIAI_OUTPUT)

#define N_RAW_STATE 6
#define FACTORY_SERVER_PORT 16016
#define FACTORY_SERVER_ADDRESS "http://0.0.0.0:16016"
#define GAME_SERVER_ADDRESS "ws://pong:4443/ws"
#define CLIENT_INPUT_TIME_SPAN 100
#define CLIENT_MAX_SPAN_STATE 3.0
#define INPUT_TIMESTAMP 60

#include <atomic>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <exception>

using ptrFuncV = std::vector<double>*(*)(std::vector<double> const &);
using ptrFuncS = double(*)(double const);

typedef	enum e_actFunc {SIGMOIDs, RELU, LEAKYRELU, TANH, STEP} t_actFunc;
typedef enum e_mode {TRAIN, TEST} t_mode;

typedef enum	e_state {
	WAITING,
	ON_GOING,
	FINISHED
}	t_state;

class	DisconnectedFactoryException : std::exception {
	const char*	what() const noexcept {
		return "can not find factory";
	}
};

class	WsConnectionException : std::exception {
	const char*	what() const noexcept {
		return "connection to websocket failed";
	}
};

class	UnknownMessageTokenException : std::exception {
	const char*	what() const noexcept {
		return "Unknown token in message";
	}
};

class	DuplicateGameException : std::exception {
	const char*	what() const noexcept {
		return "Game Id on use";
	}
};

class	UnknownGameException : std::exception {
	const char*	what() const noexcept {
		return "Unknown Game Id";
	}
};


#endif

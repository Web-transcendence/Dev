/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/28 02:13:24 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"

#include "TypeDefinition.hpp"
#include "Math.namespace.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>

Environment::Environment(unsigned int const maxStep) : \
	ball(std::array<double, 6>{WIDTH/2,HEIGHT,2,0,10,10}), \
	rPaddle(std::array<double, 5>{WIDTH-30,HEIGHT/2,20,200,10}), \
	lPaddle(std::array<double, 5>{30,HEIGHT/2,20,200,10}), \
    _maxEpStep(maxStep) {
	return ;
}

Environment::~Environment( void ) {}

void	Environment::action(t_exp * exp) {
	this->moovePaddle(exp->action);
	this->mooveBall(exp);
	exp->nextState = this->getState();
	return ;
}

void	Environment::moovePaddle(int const action) {
	if (action)
		this->rPaddle.y -= this->rPaddle.s;
	else if (action)
		this->rPaddle.y += this->rPaddle.s;
	
	this->lPaddle.y += (this->randDouble() > 0.5 ? 1 : -1) * this->lPaddle.s * 0.5;
	
	if (this->rPaddle.y < 0.5 * this->rPaddle.h)
		this->rPaddle.y = 0.5 * this->rPaddle.h;
	else if (rPaddle.y > HEIGHT - this->rPaddle.h * 0.5)
		this->rPaddle.y = HEIGHT - 0.5 * this->rPaddle.h;
	
	if (this->lPaddle.y < 0.5 * this->lPaddle.h)
		this->lPaddle.y = 0.5 * this->lPaddle.h;
	else if (this->lPaddle.y > HEIGHT - this->lPaddle.h * 0.5)
		this->lPaddle.y = HEIGHT - 0.5 * this->lPaddle.h;
	return ;
}

void	Environment::mooveBall(t_exp * exp) {
	double	oldX = ball.x;
    double	oldY = ball.y;
    int		collision = 0;
    ball.x += cos(ball.a) * ball.s;
    ball.y += sin(ball.a) * ball.s;
    if ((ball.x > rPaddle.x - 0.5 * rPaddle.w && (ball.a < 0.5 * M_PI || ball.a > 1.5 * M_PI)) || (ball.x < lPaddle.x + 0.5 * lPaddle.w && (ball.a > 0.5 * M_PI && ball.a < 1.5 * M_PI)))
        collision = checkCollision(oldX, oldY); // 0 = nothing || 1 = right || 2 = left
    if (collision == 1) {
        oldY = oldY + tan(ball.a) * (rPaddle.x - (0.5 * rPaddle.w) - oldX);
        oldX = rPaddle.x - (0.5 * rPaddle.w);
        bounceAngle(rPaddle, "right");
        ball.x = oldX + cos(ball.a) * (sqrt(pow(ball.y - oldY, 2) + pow(ball.x - oldX, 2)));
        ball.y = oldY + sin(ball.a) * (sqrt(pow(ball.y - oldY, 2) + pow(ball.x - oldX, 2)));
    } else if (collision == 2) {
        oldY =  oldY - tan(ball.a) * (lPaddle.x + (0.5 * lPaddle.w) - oldX);
        oldX = lPaddle.x + (0.5 * lPaddle.w);
        bounceAngle(lPaddle, "left");
        ball.x = oldX + cos(ball.a) * (sqrt(pow(ball.y - oldY, 2) + pow(ball.x - oldX, 2)));
        ball.y = oldY + sin(ball.a) * (sqrt(pow(ball.y - oldY, 2) + pow(ball.x - oldX, 2)));
    }
    if (ball.x > WIDTH) {
        exp->done = true;
        exp->reward = -1.;
		return ;
    }
    else if (ball.x < 0) {
        exp->done = true;
        exp->reward = 1.;
		return ;
    }
    // if (ball.x > WIDTH) {
    //     ball.x = WIDTH - (ball.x - WIDTH);
    //     ball.a = M_PI - ball.a;
    // } else if (ball.x < 0) {
    //     ball.x = -ball.x;
    //     ball.a = M_PI - ball.a;
    // }
    if (ball.y > HEIGHT) {
        ball.y = HEIGHT - (ball.y - HEIGHT);
        ball.a = 2 * M_PI - ball.a;
    } else if (ball.y < 0) {
        ball.y = -ball.y;
        ball.a = 2 * M_PI - ball.a;
    }
    norAngle();
	return ;
}

void	Environment::reset( void ) {
	this->_done = false;
    if (this->ball.x < 0.)
        this->ball.a = M_PI;
    else
        this->ball.a = 0.;
    this->ball.x = 0.5 * WIDTH;
    this->ball.y = 0.5 * HEIGHT;
    this->ball.s = this->ball.is;
    this->lPaddle.y = 0.5 * HEIGHT;
    this->rPaddle.y = 0.5 * HEIGHT;
	return ;
}

int		Environment::checkCollision(double oldX, double oldY) {
    int sign = 1;
    int posy = 0;
    if (ball.a > 0.5 * M_PI && ball.a < 1.5 * M_PI)
        sign = -1;
    if (sign == 1)
        posy = oldY + tan(ball.a) * (rPaddle.x - (0.5 * rPaddle.w) - oldX);
    else if (sign == -1)
        posy = oldY + tan(ball.a) * (lPaddle.x + (0.5 * lPaddle.w) - oldX);
    if (sign == 1 && posy >= rPaddle.y - 0.5 *  rPaddle.h && posy <= rPaddle.y + 0.5 * rPaddle.h)
        return 1;
    else if (sign == -1 && posy >= lPaddle.y - 0.5 * lPaddle.h && posy <= lPaddle.y + 0.5 * lPaddle.h)
        return 2;
    return 0;
}

void	Environment::bounceAngle(t_paddle & paddle, std::string const & side) {
    double const ratio = (ball.y - paddle.y) / (paddle.h / 2);
    ball.s = ball.is + 0.5 * ball.is * abs(ratio);
    ball.a = M_PI * 0.25 * ratio;
    if (side == "right")
        ball.a = M_PI - ball.a;
    norAngle();
    return ;
}

void	Environment::norAngle( void ) {
	if (this->ball.a < 0.) {
        this->ball.a += 2. * M_PI;
    }
    if (this->ball.a > (2. * M_PI)) {
        this->ball.a -= 2. * M_PI;
    }
	return ;
}

std::array<double,6>    Environment::getState( void ) {
    return std::array<double,6>{this->ball.x,\
                                this->ball.y,\
                                this->rPaddle.x,\
                                this->rPaddle.y,\
                                this->lPaddle.x,
                                this->lPaddle.y};
}

std::vector<double>*	Environment::getStateVector(std::array<double,6> const & act, std::array<double,6> const & old) {
	auto state = std::vector<std::vector<double>>(HEIGHT/DS_R, std::vector<double>(WIDTH/DS_R));

	std::array<int, 2>	idx = {static_cast<int>(act[0]/DS_R), static_cast<int>(act[1]/DS_R)};
	std::array<int, 2>	lastIdx = {static_cast<int>(old[0]/DS_R), static_cast<int>(old[1]/DS_R)};
	lineDrag(state, idx, lastIdx);
	std::array<double, 4>	pCoord = {act[2], act[3], act[4], act[5]}; 
	paddle(state, std::array<int, 2>{static_cast<int>(pCoord[0]/DS_R), static_cast<int>(pCoord[1]/DS_R)});
	paddle(state, std::array<int, 2>{static_cast<int>(pCoord[2]/DS_R), static_cast<int>(pCoord[3]/DS_R)});
    auto stateFlat = Math::flatten2D(state);
	return stateFlat;
}

void	Environment::lineDrag(std::vector<std::vector<double>> & simulation, std::array<int, 2> actxy, std::array<int, 2> oldxy) {
	int				diffX = std::abs(actxy[0] - oldxy[0]);
	int const		diffY = std::abs(actxy[1] - oldxy[1]);
	int const		direcX = actxy[0] < oldxy[0] ? 1 : -1;
	int const		direcY = actxy[1] < oldxy[1] ? 1 : -1;
	int	const		valuePerRow = diffY == 0 ? diffX : diffX / diffY;
	double const	offset = 1. / diffX;
	int				countValue = 0;
	double			value = 1.;
	while (diffX > 0) {
		simulation.at(actxy[1]).at(actxy[0]) = value;
		if (countValue < valuePerRow) {
			actxy[0] += direcX;
			++countValue;
		} else {
			actxy[1] += direcY;
			countValue = 0;
		}
		value -= offset;
		--diffX;
	}
	return ;
}

void	Environment::paddle(std::vector<std::vector<double>> & simulation, std::array<int, 2> pxy) {
	int	size = this->rPaddle.h / DS_R; // si ya un probleme c est ici
	
	pxy[1] -= size / 2;
	while (size) {
		simulation.at(pxy[1]).at(pxy[0]) = 0.5;
		++pxy[1];
		--size;
	}
	return ;
}

int	Environment::randInt( void ) const {
	static std::random_device				rd;
    static std::mt19937						gen(rd());
	static std::discrete_distribution<int>	dist({2, 1});
	return dist(gen);	
}

double	Environment::randDouble( void ) const {
	static std::random_device 					rd;
    static std::mt19937 						gen(rd());  
    static std::uniform_real_distribution<double>	dist(0., 1.);
	return dist(gen);
}
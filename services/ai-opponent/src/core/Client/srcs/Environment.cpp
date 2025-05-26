/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/22 14:39:43 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"

#include "TypeDefinition.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

Environment::Environment( void ) : \
	ball(std::array<double, 6>{WIDTH/2,HEIGHT,2,0,10,10}), \
	rPaddle(std::array<double, 5>{WIDTH-30,HEIGHT/2,20,200,10}), \
	lPaddle(std::array<double, 5>{30,HEIGHT/2,20,200,10}) {
    this->reset(this->ball, this->lPaddle, this->rPaddle);
	return ;
}

Environment::~Environment( void ) {}

void	Environment::action(int action) {
    int const   timeStamp = 6;
	auto		state = std::make_shared<std::vector<double>>(std::vector<double>(N_NEURON_INPUT));
    double num = 0.16;
    for (int lap = 0; lap < timeStamp; lap++, num += 0.16) {
        this->moovePaddle(action);
        this->mooveBall();
		auto	act = this->getGameState();
		this->drawBall(*state, std::floor(act.bx/DS_R), std::floor(act.by/DS_R), num);
		this->drawPaddle(*state, std::floor(act.rPx/DS_R), std::floor(act.rPy/DS_R), 200, num);
		this->drawPaddle(*state, std::floor(act.lPx/DS_R), std::floor(act.lPy/DS_R), 200, num);
    }
	this->_state = state;
	return ;
}

void	Environment::moovePaddle(int const action) {
	if (action == 0)
		this->rPaddle.y -= this->rPaddle.s;
	else if (action == 1)
		this->rPaddle.y += this->rPaddle.s;
	
	this->moovelPaddle();
	
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

void	Environment::moovelPaddle( void ) {
	double diff = ball.y - lPaddle.y;

	double tolerance = 10.0;

    if (std::abs(diff) > tolerance) {
		double direction = (diff > 0) ? 1.0 : -1.0;
		lPaddle.y += direction * lPaddle.s * 0.6;
	}
}

void	Environment::mooveBall( void ) {
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
        ball.x = WIDTH - (ball.x - WIDTH);
        ball.a = M_PI - ball.a;
    } else if (ball.x < 0) {
        ball.x = -ball.x;
        ball.a = M_PI - ball.a;
    }
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

void	Environment::reset(t_ball const & ball, t_paddle const & lPaddle, t_paddle const & rPaddle) {
    memcpy(&this->ball, &ball, sizeof(t_ball));
    memcpy(&this->lPaddle, &lPaddle, sizeof(t_paddle));
    memcpy(&this->rPaddle, &rPaddle, sizeof(t_paddle));
	auto		state = std::make_shared<std::vector<double>>(std::vector<double>(N_NEURON_INPUT));
	this->drawBall(*state, std::floor(this->ball.x/DS_R), std::floor(this->ball.y/DS_R), 1.0);
	this->drawPaddle(*state, std::floor(this->rPaddle.x/DS_R), std::floor(this->rPaddle.y/DS_R), 200, 1.0);
	this->drawPaddle(*state, std::floor(this->lPaddle.x/DS_R), std::floor(this->lPaddle.y/DS_R), 200, 1.0);
	this->_state = state;
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
    ball.s = ball.is + 0.5 * ball.is * std::abs(ratio);
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

std::shared_ptr<std::vector<double>>	Environment::getState( void ) {
	return	this->_state;
}

t_gState    Environment::getGameState( void ) {
    return t_gState(std::array<double,6>{this->ball.x,\
                                this->ball.y,\
                                this->rPaddle.x,\
                                this->rPaddle.y,\
                                this->lPaddle.x,\
                                this->lPaddle.y});
}

void	Environment::drawBall(std::vector<double> & s, int x, int y, double num) {
	int i = y * DOWN_SCALE_W + x;
	
	if (i < N_NEURON_INPUT && i >= 0)
		s.at(i) = num;
	return ;
}

void	Environment::drawPaddle(std::vector<double> & s, int x, int y, int sizePaddle, double num) {
	int	size = sizePaddle / DS_R;
	int start = (y - (size / 2)) * DOWN_SCALE_W;
	
	while (size && start < 600 && start >= 0) {
		s.at(start + x) = num;
		start += DOWN_SCALE_W;
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
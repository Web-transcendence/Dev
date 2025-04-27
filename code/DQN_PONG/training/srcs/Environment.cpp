/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.cpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:57:44 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/27 13:57:53 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Environment.class.hpp"

#include "TypeDefinition.hpp"
#include "Math.namespace.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>

Environment::Environment(unsigned int const maxStep) : _maxEpStep(maxStep), \
	ball(std::array<double, 6>{WIDTH/2,HEIGHT,2,0,10,10}), \
	lPaddle(std::array<double, 5>{30,HEIGHT/2,20,200,10}), \
	rPaddle(std::array<double, 5>{WIDTH-30,HEIGHT/2,20,200,10}) {
	this->_state = std::vector<double>();
	return ;
}

Environment::~Environment( void ) {}

void	Environment::action(t_exp * exp) {
	
	this->mooveBall();
	char const	place = this->_myMap[state];
	if (place == 'G' || place == 'H') {
		exp->done = true;
		if (place == 'G')
			++exp->reward;
	}
	exp->nextState.at(state) = 1.0;
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

double	Environment::mooveBall( void ) {
	double	oldX = ball.x;
    double	oldY = ball.y;
    int		collision = 0;
    ball.x += Math.cos(ball.angle) * ball.speed;
    ball.y += Math.sin(ball.angle) * ball.speed;
    if ((ball.x > rPaddle.x - 0.5 * rPaddle.width && (ball.angle < 0.5 * Math.PI || ball.angle > 1.5 * Math.PI)) || (ball.x < lPaddle.x + 0.5 * lPaddle.width && (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)))
        collision = checkCollision(oldX, oldY); // 0 = nothing || 1 = right || 2 = left
    if (collision === 1) {
        oldY = oldY + Math.tan(ball.angle) * (rPaddle.x - (0.5 * rPaddle.width) - oldX);
        oldX = rPaddle.x - (0.5 * rPaddle.width);
        bounceAngle(rPaddle, "right");
        ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
    } else if (collision === 2) {
        oldY =  oldY - Math.tan(ball.angle) * (lPaddle.x + (0.5 * lPaddle.width) - oldX);
        oldX = lPaddle.x + (0.5 * lPaddle.width);
        bounceAngle(lPaddle, "left");
        ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
    }
    if (ball.x > WIDTH) {
        reset();
		return -1.;
    }
    else if (ball.x < 0) {
        reset();
		return 1.;
    }
    if (ball.x > WIDTH) {
        ball.x = WIDTH - (ball.x - WIDTH);
        ball.angle = Math.PI - ball.a;
    } else if (ball.x < 0) {
        ball.x = -ball.x;
        ball.angle = Math.PI - ball.angle;
    }
    if (ball.y > canvas.height) {
        ball.y = canvas.height - (ball.y - canvas.height);
        ball.angle = 2 * Math.PI - ball.angle;
    } else if (ball.y < 0) {
        ball.y = -ball.y;
        ball.angle = 2 * Math.PI - ball.angle;
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

void	Environment::norAngle( void ) {
	if (this->ball.a < 0.)
        this->ball.a += 2. * M_PI;
    if (this->ball.a > 2. * M_PI)
        this->ball.a -= 2. * M_PI;
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
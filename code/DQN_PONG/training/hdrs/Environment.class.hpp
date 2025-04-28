/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:41:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/28 01:33:11 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENVIRONMENT_CLASS_HPP
# define ENVIRONMENT_CLASS_HPP
# define WIDTH 1200
# define HEIGHT 800
# define DS_R 20
# include "TypeDefinition.hpp"
# include <vector>
# include <string>
# include <array>

typedef struct s_ball {
	double	x;
	double	y;
	double	a;
	double	s;
	double	is;
	double	r;
	s_ball(std::array<double, 6> const & i) : x(i[0]),y(i[1]),a(i[2]),s(i[3]),is(i[4]),r(i[5]) {}
}	t_ball;

typedef struct s_paddle {
	double	x;
	double	y;
	double	w;
	double	h;
	double	s;
	s_paddle(std::array<double, 5> const & i) : x(i[0]),y(i[1]),w(i[2]),h(i[3]),s(i[4]) {}
}	t_paddle;

class Environment {
public:
	Environment(unsigned int const maxStep);
	~Environment( void );

	void					action(s_exp * exp);

	void					reset( void );
	
	std::array<double,6>				getState( void );
	std::vector<double>*	getStateVector(std::array<double,6> const & act, std::array<double,6> const & old);

private:

	void	moovePaddle(int const action);
	void	mooveBall(t_exp * exp);

	void	norAngle(void);
	int		checkCollision(double oldX, double oldY);
	void	bounceAngle(t_paddle & paddle, std::string const & side);

	void	lineDrag(std::vector<std::vector<double>> & simulation, std::array<int, 2> actxy, std::array<int, 2> oldxy);
	void	paddle(std::vector<std::vector<double>> & simulation, std::array<int, 2> pxy);

	int		randInt(void) const;
	double	randDouble(void) const;

	t_ball				ball;
	t_paddle			rPaddle;
	t_paddle			lPaddle;

	std::vector<double>	_state;
	bool				_done;

	unsigned int const	_maxEpStep;

friend class Agent;
};

#endif

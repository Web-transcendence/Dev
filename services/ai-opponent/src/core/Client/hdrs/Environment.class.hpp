/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/07 13:05:38 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/07 14:20:31 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENVIRONMENT_CLASS_HPP
# define ENVIRONMENT_CLASS_HPP
# define WIDTH 1200
# define HEIGHT 800
# define DS_R 40
# define DOWN_SCALE_W 30
# define DOWN_SCALE_H 20
# include "TypeDefinition.hpp"
# include <vector>
# include <string>
# include <memory>
# include <array>

typedef struct s_gState {
	double	bx;
	double	by;
	double	rPx;
	double	rPy;
	double	lPx;
	double	lPy;
	s_gState(std::array<double, 6> const & i) : bx(i[0]),by(i[1]),rPx(i[2]),rPy(i[3]),lPx(i[4]),lPy(i[5]) {}
}	t_gState;

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
	Environment( void );
	~Environment( void );

	void					action(int action);

	void					reset(t_ball const & ball, t_paddle const & lPaddle, t_paddle const & rPaddle);
	
	std::shared_ptr<std::vector<double>>	getState(void);

private:

	t_gState	getGameState( void );

	void	moovePaddle(int const action);
	void	moovelPaddle(void);
	void	mooveBall(void);

	void	norAngle(void);
	int		checkCollision(double oldX, double oldY);
	void	bounceAngle(t_paddle & paddle, std::string const & side);

	void	drawBall(std::vector<double> & s, int x, int y, double num);
	void	drawPaddle(std::vector<double> & s, int x, int y, int sizePaddle, double num);

	int		randInt(void) const;
	double	randDouble(void) const;

	t_ball				ball;
	t_paddle			rPaddle;
	t_paddle			lPaddle;

	std::shared_ptr<std::vector<double>>	_state;
	bool									_done;

friend class Agent;
};

#endif

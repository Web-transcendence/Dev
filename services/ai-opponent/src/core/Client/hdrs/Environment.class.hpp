/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/05/07 13:05:38 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/27 13:50:18 by thibaud          ###   ########.fr       */
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

typedef struct s_ball {
	float	x;
	float	y;
	float	a;
	float	s;
	float	is;
	float	r;
	s_ball(std::array<float, 6> const & i) : x(i[0]),y(i[1]),a(i[2]),s(i[3]),is(i[4]),r(i[5]) {}
}	t_ball;

typedef struct s_paddle {
	float	x;
	float	y;
	float	w;
	float	h;
	float	s;
	s_paddle(std::array<float, 5> const & i) : x(i[0]),y(i[1]),w(i[2]),h(i[3]),s(i[4]) {}
}	t_paddle;

typedef struct s_gState {
	float	bx;
	float	by;
	float	rPx;
	float	rPy;
	float	lPx;
	float	lPy;
	s_gState(std::array<float, 6> const & i) : bx(i[0]),by(i[1]),rPx(i[2]),rPy(i[3]),lPx(i[4]),lPy(i[5]) {}
}	t_gState;

typedef struct s_aState {
	t_ball		ball;
	t_paddle	rPaddle;
	t_paddle	lPaddle;
}	t_aState;

class Environment {
public:
	Environment( void );
	~Environment( void );

	void					action(int action);

	void					reset(t_ball const & ball, t_paddle const & lPaddle, t_paddle const & rPaddle);
	void					reset( void );
	
	std::shared_ptr<std::vector<float>>	getState(void);

private:

	t_gState	getGameState( void );

	void	moovePaddle(int const action);
	void	moovelPaddle(void);
	void	mooveBall(void);

	void	norAngle(void);
	int		checkCollision(float oldX, float oldY);
	void	bounceAngle(t_paddle & paddle, std::string const & side);

	void	drawBall(std::vector<float> & s, int x, int y, float num);
	void	drawPaddle(std::vector<float> & s, int x, int y, int sizePaddle, float num);

	int		randInt(void) const;
	float	randfloat(void) const;

	t_ball				ball;
	t_paddle			rPaddle;
	t_paddle			lPaddle;

	std::shared_ptr<std::vector<float>>	_state;
	bool									_done;

friend class Agent;
};

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:41:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/27 13:55:48 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENVIRONMENT_CLASS_HPP
# define ENVIRONMENT_CLASS_HPP
# define WIDTH 1200
# define HEIGHT 800
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

	void			action(s_exp * exp);

	void			reset( void );
	
private:

	void	norAngle(void);
	void	moovePaddle(int const action);
	double	mooveBall(void);


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

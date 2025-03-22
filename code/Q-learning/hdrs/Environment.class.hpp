/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:41:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/22 15:44:24 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENVIRONMENT_CLASS_HPP
# define ENVIRONMENT_CLASS_HPP
# include <vector>
# define UP 0
# define DOWN 1
# define RIGHT 2
# define LEFT 3

class Environment {
public:
	Environment(int const col, int const row, double const rewardTo, unsigned int const maxStep);
	~Environment( void );

	std::array<int, 2>	action(int const act);
	void				reset( void );
	void				render( void );

private:
	int		randInt(void);

	bool							_done;
	
	int								_state;
	std::vector<char>				_myMap;
	unsigned int const				_row;
	unsigned int const				_col;
	double const					_rewardThreshold;
	unsigned int const				_maxEpStep;

friend class QAgent;
};

#endif

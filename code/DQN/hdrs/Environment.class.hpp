/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Environment.class.hpp                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 11:41:48 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/04 13:36:12 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef ENVIRONMENT_CLASS_HPP
# define ENVIRONMENT_CLASS_HPP
# include "TypeDefinition.hpp"
# include <vector>
# include <array>

class Environment {
public:
	Environment(int const col, int const row, double const rewardTo, unsigned int const maxStep);
	~Environment( void );

	void			action(s_exp * exp);

	void			reset( void );
	void			render( void );

	unsigned int	getUIntState(std::vector<double> const & src);

private:
	int		randInt(void);

	std::vector<double>	_state;
	bool				_done;

	std::vector<char>	_myMap;
	unsigned int const	_row;
	unsigned int const	_col;
	double const		_rewardThreshold;
	unsigned int const	_maxEpStep;

friend class Agent;
};

#endif

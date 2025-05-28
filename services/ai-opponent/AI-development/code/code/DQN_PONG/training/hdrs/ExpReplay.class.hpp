/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ExpReplay.class.hpp                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/31 09:07:10 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/04 09:31:34 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef EXPREPLAY_CLASS_HPP
# define EXPREPLAY_CLASS_HPP
# include "TypeDefinition.hpp"
# include <vector>

class ExpReplay {
public:
	ExpReplay( void );
	ExpReplay(unsigned int const max, unsigned int const min);
	~ExpReplay( void );

	void					add(t_exp*	experience);

	unsigned int			getMax( void ) const;
	unsigned int			getMin( void ) const;
	unsigned int			getNum( void ) const;
	std::vector<t_exp*>*	getBatch(unsigned int const size) const;
private:
	unsigned int const	_max;
	unsigned int const	_min;
	unsigned int		_size;
	std::vector<t_exp*>	_experiences;
};

#endif

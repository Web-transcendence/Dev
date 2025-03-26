/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ExpReplay.class.hpp                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/25 11:04:06 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/25 12:35:06 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef EXPREPLAY_CLASS_HPP
# define EXPREPLAY_CLASS_HPP
# include <vector>

typedef enum	s_action {
	UP,
	DOWN,
	RIGHT,
	LEFT
}	t_action;

typedef struct	s_exp {
	std::vector<double>	state;
	t_action			action;
	unsigned int		reward;
	std::vector<double>	nextState;
	bool				done;
}	t_exp;

class ExpReplay {
public:
	ExpReplay(unsigned int const maxExp, unsigned int const minExp);
	~ExpReplay( void ) {}
	
	std::vector<t_exp*>*	getBatch(unsigned int const batchSize);
	void					add(t_exp * experience);
private:
	unsigned int			maxExp;
	unsigned int			minExp;
	unsigned int			size;
	
	std::vector<t_exp*>		experiences;
};


#endif

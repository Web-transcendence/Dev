/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   QAgent.class.hpp                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:31:17 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/22 15:24:29 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef QAGENT_CLASS_HPP
# define QAGENT_CLASS_HPP

class Environment;

typedef enum e_mode {TRAIN, TEST} t_mode;

class QAgent {
public:
	QAgent(int const maxTraining, int const maxAct, double const learningRate, \
			double const discount, double const exploRate, double const exploDecay);
	~QAgent( void );

	void	train( void );
	void	test( void );
	int		policy(t_mode const mode);

	void	setMap(Environment & env) {this->_env = &env;};
	void	genQMatrix( void );
	
private:
	QAgent( void );
	
	int	randInt(void);

	Environment*						_env;
	
	std::vector<std::vector<double>>	_QMatrix;

	int const							_maxEpTraining;
	int const							_maxActions;
	double const						_learningRate;
	double const						_discount;
	double const						_explorationRate;
	double const						_explorationDecay;
};

#endif

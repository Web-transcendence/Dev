/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   DeepQAgent.class.hpp                               :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:31:17 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/27 17:35:55 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef DEEPQAGENT_CLASS_HPP
# define DEEPQAGENT_CLASS_HPP
# include <vector>

class Environment;
class Network;

typedef enum e_mode {TRAIN, TEST} t_mode;

class DeepQAgent {
public:
	DeepQAgent(int const maxTraining, int const maxAct, double const learningRate, \
			double const discount, double const exploRate, double const exploDecay);
	~DeepQAgent( void );

	void	train( void );
	void	test( void );
	int		policy(t_mode const mode);

	void	setMap(Environment & env) {this->_env = &env;};
	void	genQMatrix( void );
	void	genQNet( void );
	
private:
	DeepQAgent( void );
	
	int						randInt(void);
	std::vector<double>*	mapPlacement(int const state);
	bool					realisable( void );

	void					printQmatrix(void);

	Environment*						_env;
	Network*							_QNet;
	
	std::vector<std::vector<double>>	_QMatrix;

	int const							_maxEpTraining;
	int const							_maxActions;
	double const						_learningRate;
	double const						_discount;
	double const						_explorationRate;
	double const						_explorationDecay;

};

#endif

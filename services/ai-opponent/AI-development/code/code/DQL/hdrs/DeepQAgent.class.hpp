/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   DeepQAgent.class.hpp                               :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:31:17 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 18:19:39 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef DEEPQAGENT_CLASS_HPP
# define DEEPQAGENT_CLASS_HPP
# include <vector>
# include "Math.namespace.hpp"

class Environment;
class Network;

class DeepQAgent {
public:
	DeepQAgent(int const maxTraining, int const maxAct, double const learningRate, \
			double const discount, double const exploRate, double const exploDecay);
	~DeepQAgent( void );

	void	trainQMatrix( void );
	void	testQMatrix( void );
	int		policyQMatrix(t_mode const mode);
	
	void	trainQNetFromQMatrix( void );
	void	trainQNet( void );
	void	testQNet( void );

	void	setMap(Environment & env) {this->_env = &env;};
	void	genQMatrix( void );
	void	genQNet( void );
	
private:
	DeepQAgent( void );
	
	int						randInt(void);
	double					randDouble( void );

	std::vector<double>*	mapPlacement(int const state);

	void					printQMatrix(void);
	void					printQNet(void);

	Environment*						_env;
	
	std::vector<std::vector<double>>	_QMatrix;
	Network*							_QNet;

	int const							_maxEpTraining;
	int const							_maxActions;
	double const						_learningRate;
	double const						_discount;
	double const						_explorationRate;
	double const						_explorationDecay;

	int									_goalTraining;

};

#endif

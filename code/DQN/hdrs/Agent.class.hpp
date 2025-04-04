/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Agent.class.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/22 12:31:17 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/04 12:11:27 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef AGENT_CLASS_HPP
# define AGENT_CLASS_HPP
# include "TypeDefinition.hpp"
# include <vector>

class Environment;
class Network;
class ExpReplay;

class Agent {
public:
	Agent(int const maxTraining, int const maxAct, double const learningRate, \
			double const discount, double const exploRate, double const exploDecay);
	~Agent( void );

	void	train( void );
	void	test( void );

	void	trainQMatrix( void );
	int		policyQMatrix(t_mode const mode);

	void	setMap(Environment & env) {this->_env = &env;};
	void	genQNet(std::vector<unsigned int> const & sizes, t_actFunc hidden, t_actFunc output);
	void	genTNet(std::vector<unsigned int> const & sizes, t_actFunc hidden, t_actFunc output);
	void	genExpReplay(unsigned int const max, unsigned int const min);
	void	genQMatrix( void );
	
private:
	Agent( void );
	
	void	batchTrain(unsigned int const batchSize);				
	void	getAction(t_exp * exp, double exploRate) const;

	void	TNetUpdate( void );

	int		randInt(void) const;
	double	randDouble( void ) const;

	Environment*	_env;
	
	Network*		_QNet;
	Network*		_TNet;

	ExpReplay*		_xp;

	std::vector<std::vector<double>>	_QMatrix;
	
	int const		_maxEpTraining;
	int const		_maxActions;
	double const	_learningRate;
	double const	_discount;
	double const	_explorationRate;
	double const	_explorationDecay;

	int				_goalTraining;
};

#endif

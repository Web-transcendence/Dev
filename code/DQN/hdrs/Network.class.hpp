/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/11 11:24:02 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/03 20:41:52 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP
# include "Math.namespace.hpp"
# include <vector>

class Layer;

class Network {
public:
    Network(std::vector<unsigned int>sizes, t_actFunc actHiddenFunc, t_actFunc actOutputFunc);
    ~Network( void ) {}
    
    void                    SDG(t_tuple* trainingData, double const eta);
    void                    SDG(std::vector<t_tuple*>& trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data);
    std::vector<double>*	feedForward(std::vector<double> const & input);
    
    static void             displayProgress(int current, int max);

    void                    copyNetwork(Network const & src);

private:
    void    updateMiniBatch(std::vector<t_tuple*>& miniBatch, double const eta);
    void    backprop(std::vector<double>& input, std::vector<double>& expectedOutput);

    int     evaluate(std::vector<t_tuple*>& test_data);
    void    myShuffle(std::vector<t_tuple*>& myVector);
    void    updateWeight(double const eta, double const miniBatchSize);
    void    updateNabla_w( void );
    void    updateBias(double const eta, double const miniBatchSize);
    void    updateNabla_b( void );

    int const                   _num_layers;
    std::vector<unsigned int>&  _sizes;
    std::vector<Layer*>         _layers;

friend class Agent;
};

#endif


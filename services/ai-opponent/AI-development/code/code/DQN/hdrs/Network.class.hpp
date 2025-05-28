/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/11 11:24:02 by thibaud           #+#    #+#             */
/*   Updated: 2025/05/26 10:04:06 by thibaud          ###   ########.fr       */
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
    ~Network( void );
    
    void                    SDG(t_tuple* trainingData, double const eta);
    std::vector<double>*	feedForward(std::vector<double> const & input);
    
    static void             displayProgress(int current, int max);

    void                    copyNetwork(Network const & src);
    void                    printNetworkToJson(std::string const & outputFile);

private:
    void    backprop(std::vector<double>& input, std::vector<double>& expectedOutput);

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


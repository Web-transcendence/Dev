/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/11 11:24:02 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/12 16:44:38 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP
# include "neuron.hpp"

typedef int t_input;
typedef bool t_output;

typedef struct  s_tuple {
    t_input     input;
    t_output    expectedOutput;
}      t_tuple;

class Network {
    public:
        Network(std::vector<unsigned int>sizes);
        ~Network() {}
        
        void    SDG(std::vector<t_tuple*> &trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data);
        
    private:
        void    myShuffle(std::vector<t_tuple*>& myVector);
        void    updateMiniBatch(std::vector<t_tuple*>& miniBatch, double const eta);
        int     evaluate(std::vector<t_tuple*>& test_data);

        int const                           _num_layers;
        std::vector<unsigned int>&          _sizes;
        std::vector<std::vector<Neuron*>*>  _layers;
};

#endif


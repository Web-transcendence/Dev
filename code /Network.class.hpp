/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Network.class.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/11 11:24:02 by thibaud           #+#    #+#             */
/*   Updated: 2025/03/14 16:23:22 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef NETWORK_CLASS_HPP
# define NETWORK_CLASS_HPP
# include "Layer.class.hpp"
# include "Neuron.class.hpp"

typedef struct  s_tuple {
    std::vector<double> input;
    std::vector<double> expectedOutput;
}      t_tuple;

class Network {
    public:
        Network(std::vector<unsigned int>sizes);
        ~Network() {}
        
        void    SDG(std::vector<t_tuple*> trainingData, int const epoch, int const miniBatchSize, double const eta, std::vector<t_tuple*>* test_data);
        
    private:
        void                                                updateMiniBatch(std::vector<t_tuple*>& miniBatch, double const eta);
        std::vector<std::vector<std::vector<double>*>*>*    backprop(std::vector<double>& input, std::vector<double>& expectedOutput);
        int                                                 evaluate(std::vector<t_tuple*>& test_data);
        void                                                myShuffle(std::vector<t_tuple*>& myVector);
        std::vector<std::vector<std::vector<double>*>*>*    shapeBiases( void );
        std::vector<std::vector<std::vector<double>*>*>*    shapeWeights( void );

        int const                   _num_layers;
        std::vector<unsigned int>&  _sizes;
        std::vector<Layer*>         _layers;

};

#endif


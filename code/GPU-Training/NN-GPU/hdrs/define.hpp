
#ifndef	DEFINE_HPP
# define DEFINE_HPP

#define N_INPUT 10
#define N_OUTPUT 10
#define N_HIDDEN 30
#define L_HIDDEN 1
#define	L_GLOBAL 2
#define	WEIGHT_FIRST_HIDDEN (N_INPUT * N_HIDDEN)
#define	WEIGHT_HIDDEN (N_HIDDEN * N_HIDDEN)
#define	WEIGHT_OUTPUT (N_HIDDEN * N_OUTPUT)


class Layer {
public:
	Layer() {}
	virtual ~Layer() {}
};

class FirstHidden : Layer {
public:
	double	weights[WEIGHT_FIRST_HIDDEN];
	double	nabla_w[WEIGHT_FIRST_HIDDEN];
	double	deltaNabla_w[WEIGHT_FIRST_HIDDEN];

	double	biaises[N_HIDDEN];
	double	nabla_b[N_HIDDEN];
	double	deltaNabla_b[N_HIDDEN];

	unsigned int const	sizeWeights = N_INPUT;
	unsigned int const	sizeNeurons = N_HIDDEN;
};

class Hidden : Layer {
public:
	double	weights[WEIGHT_HIDDEN];
	double	nabla_w[WEIGHT_HIDDEN];
	double	deltaNabla_w[WEIGHT_HIDDEN];

	double	biaises[N_HIDDEN];
	double	nabla_b[N_HIDDEN];
	double	deltaNabla_b[N_HIDDEN];

	unsigned int const	sizeWeights = N_HIDDEN;
	unsigned int const	sizeNeurons = N_HIDDEN;
};

class Output : Layer {
public:
	double	weights[WEIGHT_OUTPUT];
	double	nabla_w[WEIGHT_OUTPUT];
	double	deltaNabla_w[WEIGHT_OUTPUT];

	double	biaises[N_OUTPUT];
	double	nabla_b[N_OUTPUT];
	double	deltaNabla_b[N_OUTPUT];

	unsigned int const	sizeWeights = N_HIDDEN;
	unsigned int const	sizeNeurons = N_OUTPUT;
};

class Network {
public:
	FirstHidden	layer0;
	Output		layer2;
};

#endif

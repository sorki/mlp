import math

class Layer(object):
    ''' Single layer in MLP '''
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def next_layer(self, layer):
        ''' Set following layer '''
        pass

    def init_weights(self):
        ''' Initialize weight matrix between this and following layer '''
        pass

class MLP(object):
    ''' Multi layer perceptron implementation '''
    def __init__(self, activation_fn=lambda x: math.tanh(x)):
        self.activation_fn = activation_fn

    def add_layer(self, layer_instance):
        ''' Add MLP layer - first layer is treated as an input layer,
        last as an output, rest are hidden layers '''
        pass

    def init_network(self):
        ''' Initialize random weights between layers '''
        pass

    def train(self, patterns):
        ''' Use list of patterns to train the network '''
        pass

    def run(self, inp):
        ''' Get result using `inp` as input '''
        pass

    def _activate_(self):
        ''' Run activation process for each layer '''
        pass

    def _back_propagate(self):
        ''' Run back propagation process for each layer '''
        pass

def main():
    pass

if __name__ == '__main__':
    main()


import math

class Layer(object):
    ''' Single layer in MLP '''
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.next = None
        self.weights = None

    def next_layer(self, layer_instance):
        ''' Set following layer '''
        assert isinstance(layer_instance, Layer)
        self.next = layer_instance

    def init_weights(self):
        ''' Initialize weight matrix between this and following layer '''
        self.weights = []
        for i in range(self.num_neurons):
            self.weights.append([0 for _ in range(self.next.num_neurons)])

class MLP(object):
    ''' Multi layer perceptron implementation '''
    def __init__(self, activation_fn=lambda x: math.tanh(x)):
        self.activation_fn = activation_fn
        self.layers = []

    def add_layer(self, layer_instance):
        ''' Add MLP layer - first layer is treated as an input layer,
        last as an output, rest are hidden layers '''
        assert isinstance(layer_instance, Layer)
        self.layers.append(layer_instance)

    def init_network(self):
        ''' Initialize weights between layers '''
        for i in range(len(self.layers)-1):
            self.layers[i].next_layer(self.layers[i+1])
            self.layers[i].init_weights()

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

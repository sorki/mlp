import math

class Layer(object):
    ''' Single layer in MLP '''
    def __init__(self, num_neurons):
        if num_neurons <= 0:
            raise ValueError
        self.num_neurons = num_neurons
        self.next = None
        self.prev = None
        self.weights = None
        self.values = [0 for _ in range(self.num_neurons)]

    def next_layer(self, layer_instance):
        ''' Set following layer '''
        assert isinstance(layer_instance, Layer)
        self.next = layer_instance

    def prev_layer(self, layer_instance):
        ''' Set preceding layer '''
        assert isinstance(layer_instance, Layer)
        self.prev = layer_instance

    def init_weights(self):
        ''' Initialize weight matrix between this and following layer '''
        if self.next is not None:
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

        for i in range(1, len(self.layers)):
            self.layers[i].prev_layer(self.layers[i-1])

    def train(self, patterns):
        ''' Use list of patterns to train the network '''
        pass

    def run(self, inp):
        ''' Get result using `inp` as input '''
        if len(self.layers[0].values) != len(inp):
            raise ValueError

        self.layers[0].values = inp
        self._activate()
        return self.layers[-1].values

    def _activate(self):
        ''' Run activation process for each layer '''
        for layer in self.layers[1:]:
            for idx in range(layer.num_neurons):
                val = .0
                for h_idx, h_neuron_value in enumerate(layer.prev.values):
                    val = val + h_neuron_value * layer.prev.weights[h_idx][idx]
                layer.values[idx] = self.activation_fn(val)

    def _back_propagate(self):
        ''' Run back propagation process for each layer '''
        pass

import random

class Layer(object):
    ''' Single layer in MLP '''
    def __init__(self, num_neurons,
            weight_function=lambda: random.uniform(-0.2, 0.2)):
        if num_neurons <= 0:
            raise ValueError

        self.num_neurons = num_neurons
        self.weight_function = weight_function
        self.next = None
        self.prev = None
        self.weights = None
        self.weight_changes = None
        self.difs = None
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
            self.weight_changes = []
            for i in range(self.num_neurons):
                self.weights.append([self.weight_function()
                    for _ in range(self.next.num_neurons)])
                self.weight_changes.append([0
                    for _ in range(self.next.num_neurons)])

    def __str__(self):
        out = '  V: %s\n' % self.values
        if self.weights:
            out += '  W: %s\n' % self.weights
        if self.weight_changes:
            out += '  C: %s\n' % self.weight_changes
        if self.difs:
            out += '  D: %s\n' % self.difs
        out += '\n'
        return out


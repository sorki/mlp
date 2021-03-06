import math
from layer import Layer

class MLP(object):
    ''' Multi layer perceptron implementation '''
    def __init__(self, activation_fn=lambda x: math.tanh(x),
            derivative_fn=lambda x: 1.0 - x**2 ):
        self.activation_fn = activation_fn
        self.derivative_fn = derivative_fn
        self.layers = []
        self.step = 0.3
        self.moment = 0.1
        self.verbose = False

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

        for lay in self.layers:
            lay.init_values()

    def add_bias(self):
        ''' Add bias node to first layer '''
        zero_layer = self.layers[0]
        zero_layer.num_neurons += 1
        zero_layer.has_bias = True

    def train(self, patterns, iters=1000):
        ''' Use list of patterns to train the network '''
        for i in range(iters):
            error = 0.
            for inp, target in patterns:
                self.run(inp)
                error += self._back_propagate(target)

            if self.verbose:
                print '%d %.8f' % ( i, error )
        return error

    def train_target(self, patterns,
            target_error=0.01,
            max_iterations=10000):
        ''' Train until error is lower then `target_error` or
            `max_iterations` limit is reached '''

        iters = 0
        while True:
            error = 0.
            iters += 1
            for inp, target in patterns:
                self.run(inp)
                error += self._back_propagate(target)
            if self.verbose:
                print '%d %.8f' % ( iters, error )
            if error < target_error or iters == max_iterations:
                break
        return (error, iters)

    def run(self, inp):
        ''' Get result using `inp` as input '''
        zero_layer = self.layers[0]
        required = zero_layer.num_neurons
        if zero_layer.has_bias:
            required -= 1
        if required != len(inp):
            raise ValueError

        zero_layer = self.layers[0]
        if zero_layer.has_bias:
            for i in range(zero_layer.num_neurons-1):
                zero_layer.values[i] = inp[i]
        else:
            zero_layer.values = inp
        self._activate()
        return self.layers[-1].values

    def _activate(self):
        ''' Run activation process for each layer '''
        for layer in self.layers[1:]:
            lim = layer.num_neurons
            if layer.has_bias:
                lim -= 1
            for idx in range(lim):
                val = .0
                for h_idx, h_neuron_value in enumerate(layer.prev.values):
                    val = val + h_neuron_value * layer.prev.weights[h_idx][idx]
                layer.values[idx] = self.activation_fn(val)

    def _back_propagate(self, desired):
        ''' Run back propagation process for each layer '''
        difs = []
        total_error = 0.

        for layer in reversed(self.layers):
            layer.difs = []
            if layer.next is None:
                for idx, value in enumerate(layer.values):
                    err = desired[idx] - value
                    total_error = (err**2)/2
                    layer.difs.append(err)
            else:
                for idx, value in enumerate(layer.values):
                    dif = 0.
                    err = 0.
                    for l_idx, l_dif in enumerate(layer.next.difs):
                        err += l_dif * layer.weights[idx][l_idx]

                    dif = self.derivative_fn(value) * err
                    layer.difs.append(dif)

        for layer in self.layers:
            if layer.next is None:
                continue
            for i in range(layer.num_neurons):
                for j in range(layer.next.num_neurons):
                    weight_change = layer.values[i] * layer.next.difs[j]
                    layer.weights[i][j] += self.step * weight_change + \
                        self.moment * layer.weight_changes[i][j]
                    layer.weight_changes[i][j] = weight_change

        return total_error

    def __str__(self):
        out = 'MLP:\n'
        for layer in self.layers:
            out += '%s' % layer
        return out

    def as_graph(self):
        ''' Output dot graph representation '''
        out = 'digraph mlp { '
        for layer_id, layer in enumerate(self.layers):
            out += 'subgraph cluster_%d {' % layer_id
            for neu_id, value in enumerate(layer.values):
                out+= 'N%d_%d [label="N%d_%d=%.2f"];' % (
                    layer_id, neu_id, layer_id, neu_id, value)
            out += '}'
        for layer_id, layer in enumerate(self.layers):
            if layer.next is None:
                break
            for neu_id in range(layer.num_neurons):
                for next_id in range(layer.next.num_neurons):
                    label = ''
                    if layer.weights:
                        label = '[label="%.2f"]' % (
                            layer.weights[neu_id][next_id],)
                    out += 'N%d_%d -> N%d_%d %s;' % (layer_id, neu_id,
                        layer_id+1, next_id, label)
        out += '}'
        return out

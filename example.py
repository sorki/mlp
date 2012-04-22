#!/usr/bin/env python

import mlp

a = mlp.MLP()
a.add_layer(mlp.Layer(2))
a.add_layer(mlp.Layer(2))
a.add_layer(mlp.Layer(1))

a.init_network()
a.layers[0].values = [1,1]
a.layers[0].weights[0]=[0.5,0.2]
a.layers[0].weights[1]=[0.1,-0.7]
a.layers[1].weights[0]=[0.1]
a.layers[1].weights[1]=[-0.1]
a._activate()

print a

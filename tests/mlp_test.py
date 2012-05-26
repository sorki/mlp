#!/usr/bin/env python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(".."))

from mlp import MLP, Layer

class LayerTest(unittest.TestCase):
    def test_new_layer(self):
        with self.assertRaises(ValueError):
            Layer(0)

    def test_next_layer(self):
        with self.assertRaises(AssertionError):
            Layer(15).next_layer('')

        x = Layer(1)
        y = Layer(2)
        x.next_layer(y)
        x.next is y

    def test_prev_layer(self):
        with self.assertRaises(AssertionError):
            Layer(15).prev_layer('')

        x = Layer(1)
        y = Layer(2)
        y.prev_layer(x)
        y.prev is x

    def test_init_weights(self):
        x = Layer(1)
        x.init_weights()
        self.assertIsNone(x.weights)

        cnf = lambda: 0

        x = Layer(1, cnf)
        y = Layer(1, cnf)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0]])

        x = Layer(1, cnf)
        y = Layer(2, cnf)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0, 0]])

        x = Layer(2, cnf)
        y = Layer(1, cnf)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0], [0]])

        x = Layer(2, cnf)
        y = Layer(2, cnf)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0, 0], [0, 0]])

class MLPTest(unittest.TestCase):
    def test_add_layer(self):
        a = MLP()
        with self.assertRaises(AssertionError):
            a.add_layer('')

        a.add_layer(Layer(1))
        a.add_layer(Layer(2))
        a.add_layer(Layer(3))
        self.assertEqual(len(a.layers), 3)
        for l in a.layers:
            self.assertIsInstance(l, Layer)

    def test_init_empty_network(self):
        a = MLP()
        a.init_network()

    def test_init_network(self):
        a = MLP()
        a.add_layer(Layer(1))
        a.add_layer(Layer(2))
        a.add_layer(Layer(3))
        a.init_network()
        self.assertIsNone(a.layers[0].prev)
        self.assertIsNotNone(a.layers[0].weights)
        self.assertIsNotNone(a.layers[0].next)
        self.assertIsNotNone(a.layers[1].prev)
        self.assertIsNotNone(a.layers[1].weights)
        self.assertIsNotNone(a.layers[1].next)
        self.assertIsNotNone(a.layers[2].prev)
        self.assertIsNone(a.layers[2].weights)
        self.assertIsNone(a.layers[2].next)

    def test_activate(self):
        a = MLP()
        a.add_layer(Layer(3))
        a.add_layer(Layer(2))
        a.init_network()
        a.layers[0].values = [1, 1, 1]
        a.layers[0].weights[0][0] = 1
        a.layers[0].weights[1][0] = -1
        a.layers[0].weights[2][0] = 1
        a.layers[0].weights[0][1] = -0.1
        a.layers[0].weights[1][1] = -0.5
        a.layers[0].weights[2][1] = 1
        a._activate()
        self.assertGreater(a.layers[1].values[0], 0.5)
        self.assertLess(a.layers[1].values[1], 0.5)

if __name__ == '__main__':
    unittest.main()

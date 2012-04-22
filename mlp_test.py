#!/usr/bin/env python
import unittest
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

    def test_init_weights(self):
        x = Layer(1)
        x.init_weights()
        self.assertIsNone(x.weights)

        x = Layer(1)
        y = Layer(1)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0]])

        x = Layer(1)
        y = Layer(2)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0, 0]])

        x = Layer(2)
        y = Layer(1)
        x.next_layer(y)
        x.init_weights()
        self.assertEqual(x.weights, [[0], [0]])

        x = Layer(2)
        y = Layer(2)
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
        self.assertIsNotNone(a.layers[0].weights)
        self.assertIsNotNone(a.layers[0].next)
        self.assertIsNotNone(a.layers[1].weights)
        self.assertIsNotNone(a.layers[1].next)
        self.assertIsNone(a.layers[2].weights)
        self.assertIsNone(a.layers[2].next)

if __name__ == '__main__':
    unittest.main()

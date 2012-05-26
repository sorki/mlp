#!/usr/bin/env python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(".."))

from mlp import MLP, Layer

class XorTest(unittest.TestCase):
    def setUp(self):
        xor = MLP()
        xor.add_layer(Layer(2))
        xor.add_layer(Layer(2))
        xor.add_layer(Layer(1))

        xor.init_network()

        xor.patterns = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]
        self.xor = xor

    def test_xor(self):
        print self.xor.train_target(self.xor.patterns, 0.01, 2000)
        self.validate()

    def test_xor_biased(self):
        self.xor.add_bias()
        self.xor.init_network()

        print self.xor.train_target(self.xor.patterns, 0.01, 2000)
        self.validate()

    def test_xor_biased_high_precision(self):
        self.xor.add_bias()
        self.xor.init_network()

        print self.xor.train_target(self.xor.patterns, 0.0001, 5000)
        self.validate()

    def validate(self):
        for inp, target in self.xor.patterns:
            tolerance = 0.1
            computed = self.xor.run(inp)
            error = abs(computed[0] - target[0])
            print 'input: %s target: %s, output: %s, error: %.4f' % (inp,
                target, computed, error)
            self.assertGreater(tolerance, error)


if __name__ == '__main__':
    unittest.main()

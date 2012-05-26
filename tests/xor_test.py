#!/usr/bin/env python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(".."))

from mlp import MLP, Layer

class XorTest(unittest.TestCase):
    def test_xor(self):
        xor = MLP()
        xor.add_layer(Layer(2))
        xor.add_layer(Layer(2))
        xor.add_layer(Layer(1))

        xor.init_network()

        xor_patterns = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]

        xor.train(xor_patterns)
        for inp, target in xor_patterns:
            tolerance = 0.1
            computed = xor.run(inp)
            error = abs(computed[0] - target[0])
            print 'input: %s target: %s, output: %s, error: %.4f' % (inp,
                target, computed, error)
            self.assertGreater(tolerance, error)

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
import unittest
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
        for inp, outp in xor_patterns:
            self.assertEqual(xor.run(inp), outp)

if __name__ == '__main__':
    unittest.main()

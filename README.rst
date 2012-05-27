MLP library
-----------

Provides flexible multi-layer perceptron implementation.
Network is highly configurable, allows usage of numerous layers
with different number of neurons.

Implementation is rather slow and should be used only for demonstration
purposes.

Requirements:
 - python >= 2.7
 - PIL
 - python-mnist (https://github.com/sorki/python-mnist)

For usage information, check provided examples in examples directory.

Examples:
 - xor.py (boolean xor function computation)
 - xor_bias.py (same as xor.py with added network bias)
 - img.py (recognition of font digits in examples/font_samples)
 - mn.py (recognition of digits from MNIST database [incomplete])

Examples require setting `PYTHONPATH` variable to '..' to be able to find
mlp directory - use `PYTHONPATH=.. ./xor.py` to run xor example. MNIST example
also requires placing `python-mnist` on `PYTHONPATH`.

Other features:
 - .dot graph output (`as_graph` method)
 - two training methods (defined desired error or number of iterations)
 - unit tests

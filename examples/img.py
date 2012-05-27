#!/usr/bin/env python
import os
import random
from PIL import Image
from mlp import MLP, Layer

DEBUG = False

def main():
    imres = MLP()
    num_points = 400
    imres.add_layer(Layer(num_points))
    imres.add_layer(Layer(20))
    imres.add_layer(Layer(10))

    imres.add_bias()
    imres.init_network()

    imres.step = 0.01
    imres.moment = imres.step / 10
    imres.verbose = True
    target_error = 0.01

    imres.patterns = []
    imres._patterns = []
    imres.test_patterns = []
    imres._test_patterns = []

    def norm(inp):
        def fn(x):
            return x/255
        return map(fn, inp)

    sample_dirs = set(os.listdir('font_samples'))
    train = set(random.sample(sample_dirs, len(sample_dirs)-1))
    test = sample_dirs - train
    for j in train:
        for i in range(0, 10):
            gim = Image.open('font_samples/%s/%d.png' % (j, i)).convert('L')
            imdata = norm(list(gim.getdata()))
            outvect = [0]*10
            outvect[i] = 1
            imres.patterns.append((imdata, outvect))
            imres._patterns.append((imdata, i, outvect))

    for j in test:
        for i in range(0, 10):
            gim = Image.open('font_samples/%s/%d.png' % (j, i)).convert('L')
            imdata = norm(list(gim.getdata()))
            outvect = [0]*10
            outvect[i] = 1
            imres.test_patterns.append((imdata, outvect))
            imres._test_patterns.append((imdata, i, outvect))


    print 'Training samples: %d (%s)' %  (len(imres.patterns),
        ' '.join(train))
    print 'Testing samples: %d (%s)' %  (len(imres.test_patterns),
        ' '.join(test))
    print 'Target error: %.4f' % target_error

    final_err, steps = imres.train_target(imres.patterns,
        target_error=target_error)

    print 'Training done in %d steps with final error of %.6f' % (steps,
        final_err)


    print '----- Detailed test output -----'
    total_tests = len(imres._test_patterns)
    total_fails = 0
    for inp, num, target in imres._test_patterns:
        computed = imres.run(inp)
        error = abs(computed[0] - target[0])
        computed = map(lambda x: round(x, 4), computed)
        maxn = computed[0]
        pos = 0
        for i in range(len(computed)):
            if computed[i] > maxn:
                maxn = computed[i]
                pos = i

        if num != pos:
            total_fails += 1
        print 'in: %d, out: %d' % (num, pos)
        print 'target: %s \noutput: %s' % (target, computed)

    print '-----'
    print 'Testing done - %d of %d samples classified incorrectly' % (
        total_fails, total_tests)


if __name__ == "__main__":
    main()

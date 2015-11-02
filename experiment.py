#!/usr/bin/env python

from neon.backends import gen_backend

from optimizers import (
    gradient_descent,
    newton_method,
    conjugate_gradients,
)

if __name__ == '__main__':
    be = gen_backend()
    x_init = be.array([0])
    f = lambda x: x*2*x - 3*x

    res_descent = gradient_descent(x_init, f, be, learning_rate=0.1)
    res_newton = newton_method(x_init, f, be)
    res_conjugate = conjugate_gradients(x_init, f, be)

    print 'Gradient Descent: ', res_descent.get()
    # print 'Newton\'s Method: ', res_newton.get()
    # print 'Gradient Descent: ', res_conjugate.get()


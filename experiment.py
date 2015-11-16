#!/usr/bin/env python

from neon.backends import gen_backend

from optimizers import (
    gradient_descent,
    newton_method,
    newton_backtracking,
    conjugate_gradients,
    fletcher_reeves,
)

if __name__ == '__main__':
    be = gen_backend()
    x_init = be.array([-2, -2])
    f = lambda x: x*2*x - 3*x
    f = lambda x: be.dot(be.dot(x.T, be.array([[2, 1], [1, 2]])), x)
    # f = lambda x: (x[0]**4 + x[1]**4 - 6*x[0]**2 * x[1]**2 - 1 )**2  + (4*x[0]**3*x[1] - 4*x[0]*x[1]**3)**2

    res_descent, val_descent = gradient_descent(x_init, f, be, learning_rate=0.1)
    # res_newton, val_newton = newton_method(x_init, f, be)
    # res_conj, val_conj = conjugate_gradients(x_init, f, be)

    print 'Gradient Descent: ', res_descent.get(), ' with value:', val_descent.get()
    # print 'Newton\'s Method: ', res_newton.get(), ' with value:', val_newton.get()
    # print 'Gradient Descent: ', res_conjugate.get()


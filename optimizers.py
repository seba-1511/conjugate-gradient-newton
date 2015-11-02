#!/usr/bin/env python

from neon.backends import Autodiff


def conv_test(f_init, f_new, be):
    """ A simple convergence test """
    convergence = be.empty((1, 1))
    convergence[:] = f_init
    # Avoids numerical error:
    convergence[:] = max(1, convergence.get())
    convergence[:] = be.absolute(f_new - f_init) / convergence
    return convergence.get()[0]


def conv_vec_test(x_init, x_new, be):
    """ A simple convergence test using vector values"""
    convergence = be.empty((1, 1))
    convergence[:] = be.sqrt(x_init * x_init)
    # Avoids numerical error:
    convergence[:] = max(1, convergence.get())
    convergence[:] = be.absolute(x_new - x_init) / convergence
    return convergence.get()[0]


def gradient_descent(x_init, f, be, learning_rate=0.1, epsilon=10e-9):
    x_new = be.zeros_like(x_init)
    f_init = f(x_init)
    f_new = f(x_new)
    grad_f = Autodiff(f_init, be=be, next_error=None)
    while True:
        x_new[:] = x_init - learning_rate * grad_f.get_grad_tensor([x_init])[0]
        if conv_vec_test(x_init, x_new, be) < epsilon:
        # if conv_test(f_init, f_new, be) < epsilon:
            f_val = be.empty((1, 1))
            f_val[:] = f_new
            return x_new, f_val
        x_init[:] = x_new


def newton_method(x_init, f, be, epsilon=1e-3):
    x_new = be.zeros_like(x_init)
    f_init = f(x_init)
    f_new = f(x_new)
    grad_f = Autodiff(f_init, be=be, next_error=None)
    grad_f = grad_f.get_grad_op_tree([x_init])[0]
    hessian_f = Autodiff(grad_f, be=be, next_error=None)
    hessian_f = hessian_f.get_grad_op_tree([x_init])[0]
    while True:
        x_new[:] = x_init - grad_f / hessian_f
        # if conv_vec_test(x_init, x_new, be) < epsilon:
        if conv_test(f_init, f_new, be) < epsilon:
            f_val = be.empty((1, 1))
            f_val[:] = f_new
            return x_new, f_val
        x_init[:] = x_new


def conjugate_gradients(x_init, f, epsilon=1e-3):
    return 0, 0

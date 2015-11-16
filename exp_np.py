#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm, inv

np.seterr(all='raise')


def conv_test(z_init, z_new):
    # print 'top: ', norm(z_new - z_init)
    # print 'bottom: ', max(1.0, norm(z_init))
    return norm(z_new - z_init) / max(1.0, norm(z_init))


def gradient_descent(z_init, f, df, learning_rate=0.01, epsilon=1e-3):
    steps = 0
    while True:
        steps += 1
        z_new = z_init - (learning_rate * df(z_init))
        if conv_test(z_init, z_new) < epsilon:
            return f(z_new), steps
        z_init = z_new


def newton_backtrack(z_init, f, df, Hf, epsilon=1e-3):
    pass


if __name__ == '__main__':
    """
    The gradient is:
    {   8*x*((x**6) + 3*(x**4)*(y**2) + (x**2)*(3*(y**4) - 1) + (y**2)*((y**4) + 3)),
        8*y*((x**6) + 3*(x**4)*(y**2) + 3*(x**2)*((y**4) + 1) + (y**2)*((y**4) - 1))
    }
    """
    f = lambda z: (z[0]**4 + z[1]**4 -6*(z[0]**2)*(z[1]**2) - 1)**2 + (4*(z[0]**3)*z[1] - 4*z[0]*(z[1]**3))**2
    # df = lambda z: np.array([
        # 8*z[0]*((z[0]**6) + 3*(z[0]**4)*(z[1]**2) + (z[0]**2)*(3*(z[1]**4) - 1) + (z[1]**2)*((z[1]**4) + 3)),
        # 8*z[1]*((z[0]**6) + 3*(z[0]**4)*(z[1]**2) + 3*(z[0]**2)*((z[1]**4) + 1) + (z[1]**2)*((z[1]**4) - 1))
    # ], dtype=np.float64)
    df = lambda z: np.array([
        24*(z[0]**5)*(z[1]**2) + 24*(z[0]**3)*(z[1]**4) + 8*z[0]*(z[1]**6) + 8*(z[0]**7) - 8*(z[0]**3) + 24*z[0]*(z[1]**2),
        8*(z[0]**6)*z[1] + 24*(z[0]**4)*(z[1]**3) + 24*(z[0]**2)*(z[1]**5) + 8*(z[1]**7) - 8*(z[1]**3) + 24*(z[0]**2)*z[1],
    ], dtype=np.float64)
    Hf = lambda z: np.array([
        [
            120*(x**4)*(y**2) + 72*(x**2)*(y**4) + 8*(y**6) + 56*(x**6) - 24*(x**2) + 24*(y**2),
            48*(x**5)*y + 96*(x**3)*(y**3) + 48*x*(y**5) + 48*x*y,
        ],
        [
            48*(x**5)*y + 96*(x**3)*(y**3) + 48*x*(y**5) + 48*x*y,
            8*(x**6) + 72*(x**4)*(y**2) + 120*(x**2)*(y**4) + 56*(y**6) - 24*(y**2) + 24*(x**2),
        ],
    ])

    desc_res = np.zeros((100, 100))
    newton_res = np.zeros((100, 100))
    conj_res = np.zeros((100, 100))

    desc_steps = np.zeros((100, 100))
    newton_steps = np.zeros((100, 100))
    conj_steps = np.zeros((100, 100))

    interval = 0.5

    for k in xrange(100):
        x = -interval + (2 * interval * k / 100.0)
        for j in xrange(100):
            y = -interval + (2 * interval * j / 100.0)
            z = np.array([x, y])

            r_d, r_s = gradient_descent(z, f, df)
            print r_d
            desc_res[k, j] = r_d
            desc_steps[k, j] = r_s

            # n_d, n_s = newton_backtrack(z, f, df, Hf)
            # newton_res[k, j] = n_d
            # newton_steps[k, j] = n_s

    import pdb; pdb.set_trace()

import numpy as np
import math


def f(w, b, x):
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)


def error(X, Y, w, b):
    total = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        total += 0.5*(fx - y) ** 2
    return total


def do_adam(X, Y, init_w, init_b, max_epochs):
    w, b, eta = init_w, init_b, 0.01
    m_w, m_b, v_w, v_b, m_w_hat, m_b_hat, v_w_hat, v_b_hat, eps, beta1, beta2 = 0, 0, 0, 0, 0, 0, 0, 0, 1e-08, 0.9, 0.99
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        m_w = beta1 * m_w + (1 - beta1)*dw
        m_b = beta1 * m_b + (1 - beta1)*db

        v_w = beta2 * v_w + (1 - beta2) * dw**2
        v_b = beta2 * v_b + (1 - beta2) * db**2

        m_w_hat = m_w / (1 - math.pow(beta1, i + 1))
        m_b_hat = m_b / (1 - math.pow(beta1, i + 1))

        v_w_hat = v_w / (1 - math.pow(beta2, i + 1))
        v_b_hat = v_b / (1 - math.pow(beta2, i + 1))

        w = w - (eta/ np.sqrt(v_w_hat + eps)) * m_w_hat
        b = b - (eta / np.sqrt(v_b_hat + eps)) * m_b_hat
        #print(error(X, Y, w, b), w, b)
    return w, b


if __name__ == "__main__":
    fp = open("A4_Q7_data.csv", 'r')

    fp.readline()
    X, Y = [0 for i in range(40)], [0 for i in range(40)]
    for i in range(40):
        buff = fp.readline().strip()
        X[i], Y[i] = map(eval, buff.split(','))

    w, b = 1, 1
    print(error(X, Y, w, b), w, b)
    w, b = do_adam(X, Y, w, b, 100)
    print(error(X, Y, w, b), w, b)
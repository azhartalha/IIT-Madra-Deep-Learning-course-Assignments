import numpy as np


def f(w, b, x):
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


def error(X, Y, w, b):
    total = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        total += 0.5*(fx - y) ** 2
    return total


if __name__ == '__main__':
    w, b, eta = 1, 1, 0.01
    number_of_iterations = 100

    fp = open('A2_Q4_data.csv', 'r')
    fp.readline()
    X, Y = [0 for i in range(40)], [0 for i in range(40)]
    for i in range(40):
        buff = fp.readline().strip()
        X[i], Y[i] = map(eval, buff.split(','))
    print(error(X, Y, w, b), w, b)
    for a0 in range(number_of_iterations):
        w_gradient, b_gradient = 0, 0
        for i in range(40):
            fx = f(w, b, X[i])
            tmp = (fx - Y[i]) * fx * (1 - fx)
            w_gradient += tmp * X[i]
            b_gradient += tmp
        w -= eta * w_gradient
        b -= eta * b_gradient
        print(error(X, Y, w, b), w, b)

import numpy as np
import matplotlib.pyplot as plt


def f(w, x):
    return 1.0/ (1.0 + np.exp(-(np.dot(np.transpose(w), x))))


def error(X, Y, w):
    n = len(Y)
    total = 0.0
    for x, y in zip(X, Y):
        fx = f(w, x)
        total += 0.5 * (fx - y) ** 2
    return total/n


if __name__ == "__main__":
    TX, TY, VX, VY = [], [], [], []

    fp = open("training_data.csv", 'r')
    fp.readline()

    for line in fp.readlines():
        tmp = list(map(float, line.strip().split(',')))
        TX.append(np.array(tmp[:-1]))
        TY.append(tmp[-1])
    fp.close()

    fp = open("validation_data.csv", 'r')
    fp.readline()

    for line in fp.readlines():
        tmp = list(map(float, line.strip().split(',')))
        VX.append(np.array(tmp[:-1]))
        VY.append(tmp[-1])
    fp.close()

    TE, VE = [], []
    for i in range(10):
        W = open("./weights_in_csv/weights_after_epoch_"+str(i)+".csv", 'r')
        W = np.array(list(map(float, W.readline().strip().split(','))))
        TE.append(error(TX, TY, W))
        VE.append(error(VX, VY, W))

    plt.plot(list(range(10)), TE, label="Training loss", marker='o',)

    plt.plot(list(range(10)), VE, label="Validation loss", marker='o',)

    plt.xlabel('# Epoch')

    plt.ylabel('Mean squared loss')

    plt.title('Comparison between train and validation loss')

    plt.legend()

    plt.grid(True)

    plt.show()

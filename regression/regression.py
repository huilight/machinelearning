import math
import numpy as np
from matplotlib import pyplot as plt

# def load_data():
#     xaxis = []
#     yaxis = []

#     with open("prices.txt", 'r') as f:
#         for i in f:
#             dat = i.strip().split(',')
#             xaxis.append(int(dat[0]))
#             yaxis.append(int(dat[1]))

#     x = np.array(xaxis)
#     y = np.array(yaxis)
#     return y, x, len(x)


def load_data():
    x, y = np.loadtxt("prices.txt", delimiter=',', usecols=(0, 1), unpack=True)
    # return np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), 5

    return x, y, len(x)


def h(t0, t1, x):
    # print(f"In h function {t0} {t1} {x}")
    return t0 + t1 * x


def cost(t0, t1, m, x, y):
    # results = 0
    # for _, __ in zip(x, y):
        # results = results + (h(t0, t1, _) - __) ** 2
    results = np.sum(((t0 + t1 * x) - y ) **2)
    print(results, results/(2*m))
    return results / (2 * m)


def muladd(func, *args):
    results = 0
    x, y = args[2], args[3]

    for xi, yi in zip(x,y):
        results += func(args[0], args[1], xi, yi)

    return results


def tfun0(*args):
    """t0, t1, x, y"""
    return h(args[0], args[1], args[2]) - args[3]

def tfun1(*args):
    return (h(args[0], args[1], args[2]) - args[3]) * args[2]


def gradient(t0, t1, a, m, x, y):

    temp0 = t0 - a*1000 * np.sum((t0 + t1 * x)-y) / m
    temp1 = t1 - a * np.sum(((t0 + t1*x)-y)*x) / m
    # temp0 = t0 - a * (muladd(tfun0, t0, t1, x, y) / m)
    # temp1 = t1 - a * (muladd(tfun1, t0, t1, x, y) / m)
    t0 = temp0
    t1 = temp1
    # t0 = round(temp0, 100)
    # t1 = round(temp1, 100)
    return t0, t1


def plot(t0, t1, x, y):
    # plt.xlim((min(x), max(x)))
    # plt.ylim((min(y), max(y)))
    plt.plot(x, y, "ob")
    dots = []
    for _ in x:
        dots.append(h(t0, t1, _))
    cost(t0, t1, len(x), x, y)
    plt.plot(x, dots)
    plt.show()
    print(t0, t1)


def main(a, t0=44, t1=1):
    x, y, m = load_data()

    while True:
        tm0, tm1 = t0, t1
        t0, t1 = gradient(t0, t1, a, m, x, y)
        if math.isinf(t0) or math.isinf(t1):
            break
        if math.isnan(t0) or math.isnan(t1):
            break

        # print(f'in gradient {t0:.30f}, {t1: .30f}, {tc}')
        if abs(t0 - tm0) < 1e-10 and abs(t1 - tm1) < 1e-10:
            break

    # tm0 = 43.92337078446014
    # tm1 = 0.1483948395523574
    plot(tm0, tm1, x, y)


if __name__ == '__main__':
    main(0.0000001, 44, 0)

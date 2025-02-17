import numpy as np
import math


def f(arr: np.ndarray[np.float32]) -> float | np.ndarray[np.float32]:
    x = arr.T
    return 2 * x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1] ** 2 + x[0] - 3 * x[1]


def main():
    basis: np.ndarray[np.float32] = np.array([5., 5.])
    n = 2
    d = 100
    h = 0.2
    m = 1.5
    it = 0
    epsilon = 0.000001

    while True:
        new_x = basis.copy()
        while True:
            print(f'Итерация {it}')
            it += 1
            for i in range(n):
                new_x[i] += h
                if f(new_x) < f(basis):
                    continue
                new_x[i] -= 2 * h
                if f(new_x) < f(basis):
                    continue
                new_x[i] += h
            if (new_x == basis).all():
                h /= d
            else:
                break
        pattern = new_x + m * (new_x - basis)
        if f(pattern) < f(new_x):
            basis = pattern
        else:
            basis = new_x
        print(basis)
        print(f(basis))
        
        if h <= epsilon:
            print((lambda x: np.round(x, 3))(basis))
            print(f(basis))
            break




if __name__ == '__main__':
    main()
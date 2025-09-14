import numpy as np

def f(arr: np.ndarray[np.float32]) -> float | np.ndarray[np.float32]:
    x = arr.T
    return 2 * x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1] ** 2 + x[0] - 3 * x[1]


def main():
    basis: np.ndarray[np.float32] = np.array([1., 1.])
    n = 2
    d = 10
    h = 0.2
    m = 2
    it = 0
    epsilon = 0.0001

    while True:
        new_x = basis.copy()
        while True:
            print(f'Итерация {it}')
            it += 1
            for i in range(n):
                basis = new_x.copy()
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

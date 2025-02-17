import numpy as np


def f(arr: np.ndarray[np.float32]) -> float | np.ndarray[np.float32]:
    x = arr.T
    return 2 * x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1] ** 2 + x[0] - 3 * x[1]


def gradient(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    result = np.array([4 * x[0] - 2 * x[1] + 1,
                       -2 * x[0] + 6 * x[1] - 3])
    return result


def main():
    x = np.array([5., 5.])
    it = 0
    h = 0.2
    epsilon = 0.000001

    while np.linalg.norm(gradient(x)) > epsilon:
        while True:
            print(f'Итерация {it}')
            it += 1

            new_x = x - h * gradient(x)
            if f(new_x) < f(x):
                x = new_x.copy()
                break
            else:
                h /= 2
    
    print(np.round(x, 2))
    print(f(x))


if __name__ == '__main__':
    main()
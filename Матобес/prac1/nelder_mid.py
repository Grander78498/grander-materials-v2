import numpy as np
import math


def f(arr: np.ndarray[np.float32]) -> float:
    x = arr.T
    return 2 * x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1] ** 2 + x[0] - 3 * x[1]


def calc_center(arr: np.ndarray[np.float32], k: None | int = None) -> np.ndarray[np.float32]:
    if k is not None:
        arr = np.concat([arr[:k], arr[k + 1:]])
    center = arr.mean(axis=0)
    
    return center


def max_func_value(arr: np.ndarray[np.float32]) -> int:
    return np.where(f(arr) == np.max(f(arr)))[0][0]


def min_func_value(arr: np.ndarray[np.float32]) -> int:
    return np.where(f(arr) == np.min(f(arr)))[0][0]


def main():
    array_x: np.ndarray[np.float32] = np.array([[1, 1]])
    n = 2
    m = 0.5
    it = 0
    epsilon = 0.001

    delta1 = m * ((math.sqrt(n + 1) - 1) / (n * math.sqrt(2)))
    delta2 = m * ((math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)))
    deltas = np.eye(n) * delta1 + (1 - np.eye(n)) * delta2

    new_points = array_x.repeat(n, axis=0) + deltas
    array_x = np.concat([array_x, new_points])

    while True:
        print(f'Итерация iter = {it}')
        it += 1

        k = max_func_value(array_x)
        center = calc_center(array_x, k=k)
        new_x: np.ndarray[np.float32] = 2 * center - array_x[k]
        if f(new_x) >= f(array_x[k]):
            r = min_func_value(array_x)
            array_x = array_x[r] + 0.5 * (array_x - array_x[r])
        else:
            array_x[k] = new_x.copy()

        print(array_x)
        x_center = calc_center(array_x)
        flag = True
        for i in range(n + 1):
            if abs(f(array_x[i]) - f(x_center)) >= epsilon:
                flag = False
                break
        
        if flag or it == 10:
            r = min_func_value(array_x)
            print(f(array_x[r]))
            break



if __name__ == '__main__':
    main()
import numpy as np

delta_x = 0.0001


def f(arr: np.ndarray[np.float32]) -> float | np.ndarray[np.float32]:
    x = arr.T
    return 2 * x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1] ** 2 + x[0] - 3 * x[1]


def first_partial(x: np.ndarray[np.float32], i: int):
    u = x.copy()
    u[i] += delta_x
    u = f(u)

    return (u - f(x)) / delta_x


def second_partial(x: np.ndarray[np.float32], i: int):
    u1 = x.copy()
    u1[i] += delta_x
    u1 = f(u1)

    u2 = f(x)

    u3 = x.copy()
    u3[i] -= delta_x
    u3 = f(u3)

    return (u1 - 2 * u2 + u3) / delta_x ** 2


def mixed_partial(x: np.ndarray[np.float32], i: int, j: int):
    u1 = f(x)

    u2 = x.copy()
    u2[i] -= delta_x
    u2 = f(u2)

    u3 = x.copy()
    u3[j] -= delta_x
    u3 = f(u3)

    u4 = x.copy()
    u4[i] -= delta_x
    u4[j] -= delta_x
    u4 = f(u4)

    return (u1 - u2 - u3 + u4) / delta_x ** 2


def hesse_matrix(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    matrix = []
    for i in range(len(x)):
        matrix.append([])
        for j in range(len(x)):
            if i == j:
                matrix[i].append(second_partial(x, i))
            else:
                matrix[i].append(mixed_partial(x, i, j))
    return np.array(matrix)


def check_positive(matrix: np.ndarray[np.float32]):
    minors = [] # A MINOOOOOOOOOOOOOOOOOOR
    for i in range(len(matrix)):
        minors.append(np.linalg.det(matrix[:i + 1, :i + 1]))
    if all(x > 0 for x in minors):
        return True
    else:
        flag = True
        for i in range(len(minors)):
            if minors[i] * (-1) ** (i + 1) > 0:
                pass
            else:
                flag = False
                break
        if flag:
            return False
        else:
            raise Exception('Матрица Гессе не является положительно или отрицательно определённой')



def calc_step(x: np.ndarray[np.float32]):
    result = np.square(np.linalg.norm(x)) / np.dot(hesse_matrix(x) @ gradient(x).T, gradient(x).T)
    return result


def gradient(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    result = np.array([first_partial(x, 0), first_partial(x, 1)])
    return result


def main():
    x = np.array([1., 1.])
    it = 0
    epsilon = 0.0001

    while np.linalg.norm(gradient(x)) > epsilon:
        print(f'Итерация {it}')
        it += 1

        hesse = hesse_matrix(x)
        if check_positive(hesse):
            x = x - np.linalg.inv(hesse) @ gradient(x).T
        else:
            x = x - calc_step(x) * gradient(x)
    
    print(np.round(x, 3))
    print(f(x))


if __name__ == '__main__':
    main()
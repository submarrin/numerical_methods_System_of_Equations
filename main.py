from math import sqrt
import numpy as np

end_iter = 21

# initialize the matrix
A = np.array([[1.5, -0.2, 0.1],
              [-0.1, 0.99985, -0.1],
              [-0.2995, 0.2, -0.495]])
b = np.array([2.165, 1.482715, -1.38699])
eps = 0.5 * 10 ** (-4)


def jacobi(A, b):
    count = 0
    x = [0]*len(b)
    for iter_count in range(end_iter):
        print("Решение на шаге:", x)
        new = [0]*len(x)
        count += 1
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            new[i] = (b[i] - s1 - s2) / A[i, i]
        [a1, a2, a3] = new
        x = new
    print()
    print("Решение по методу Якоби :" + str(x) + '' + str(count))


def zeidel(A, b):
    n = len(A)
    x = [.0 for i in range(len(A))]
    count = 0
    converge = False
    while not converge:
        new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            new[i] = (b[i] - s1 - s2) / A[i][i]
            count += 1
        converge = sqrt(sum((new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = new
    return print('Решение по методу Зейделя: ' + str(x) + str(count))


jacobi(A, b)
zeidel(A, b)
import numpy as np
from typing import Tuple

from multiplication import multiply_matrices
from inversion import inverse
from utils import (
    split_matrix_into_quadrants,
    get_zeros_matrix,
    substract_matrices,
    merge_matrices,
    timeit
)


LU = Tuple[np.ndarray, np.ndarray]


def LU_factorization_2D(matrix: np.ndarray, l:int) -> LU:  # noqa E741
    A_11 = matrix[0][0]
    A_12 = matrix[0][1]
    A_21 = matrix[1][0]
    A_22 = matrix[1][1]

    L_11, U_11 = LU_factorization(A_11, l)

    U_11_inv = inverse(U_11, l)
    L_21 = A_21 * U_11_inv
    L_11_inv = inverse(L_11, l)
    U_12 = L_11_inv * A_12
    S = A_22 - A_21 * U_11_inv * L_11_inv * A_12
    Ls, Us = LU_factorization(S, l)
    U_22, L_22 = Us, Ls

    return (
        np.array([[L_11, 0], [L_21, L_22]]),
        np.array([[U_11, U_12], [0, U_22]])
    )


def LU_factorization(matrix: np.ndarray, l: int):  # noqa E741
    if isinstance(matrix, np.int32) or isinstance(matrix, np.float64):
        return (np.array(1), np.array(matrix))
    elif matrix.shape == (2, 2):
        return LU_factorization_2D(matrix, l)
    else:
        A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(matrix)

        L_11, U_11 = LU_factorization(A_11, l)
        U_11_inv = inverse(U_11, l)
        L_21 = multiply_matrices(A_21, U_11_inv, l)
        L_11_inv = inverse(L_11, l)
        U_12 = multiply_matrices(L_11_inv, A_12, l)
        S = substract_matrices(
            A_22,
            multiply_matrices(multiply_matrices(multiply_matrices(A_21, U_11_inv, l), L_11_inv, l), A_12, l)  # noqa: 501
        )
        Ls, Us = LU_factorization(S, l)
        U_22, L_22 = Us, Ls

        return (
            merge_matrices(L_11, get_zeros_matrix(L_11.shape[0]), L_21, L_22),
            merge_matrices(U_11, U_12, get_zeros_matrix(U_11.shape[0]), U_22)
        )


def calculate_determinant_based_on_LU_matrices(
    L_result: np.ndarray, U_result: np.ndarray
) -> np.float64:
    return np.prod(L_result.diagonal()) * np.prod(U_result.diagonal())


@timeit
def matrix_factorization_interface(
    matrix: np.ndarray, l: int  # noqa: E741
) -> LU:
    return LU_factorization(matrix, l)

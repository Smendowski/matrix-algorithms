import numpy as np

from multiplication import multiply_matrices
from utils import (
    split_matrix_into_quadrants,
    get_identity_matrix,
    substract_matrices,
    merge_matrices,
    timeit
)

import config


class ZeroElementInversionError(Exception):
    """Custom error that is raised when zero element tries to be inverted."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def inverse_2D(matrix: np.ndarray, l: int) -> np.array:  # noqa: E741
    config.FLOATING_POINT_OPERATIONS += 14

    A_11 = matrix[0][0]
    A_12 = matrix[0][1]
    A_21 = matrix[1][0]
    A_22 = matrix[1][1]

    A_11_inv = inverse(A_11, l)
    S_22 = A_22 - A_21 * A_11_inv * A_12
    S_22_inv = inverse(S_22, l)
    B_11 = A_11_inv * (1 + A_12 * S_22_inv * A_21 * A_11_inv)
    B_12 = -A_11_inv * A_12 * S_22_inv
    B_21 = -S_22_inv * A_21 * A_11_inv
    B_22 = S_22_inv

    return np.array([[B_11, B_12], [B_21, B_22]])


def inverse(matrix: np.ndarray, l:int) -> np.ndarray:  # noqa E741

    try:
        x, y = matrix.shape
    except ValueError:
        config.FLOATING_POINT_OPERATIONS += 1
        if matrix == 0:
            raise ZeroElementInversionError("Can't handle zero-like element!")
        else:
            return 1/matrix
    else:
        if matrix.shape == (2, 2):
            return inverse_2D(matrix, l)
        else:
            config.FLOATING_POINT_OPERATIONS += 4
            A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(matrix)
            i_matrix = get_identity_matrix(A_11.shape[0])

            A_11_inv = inverse(A_11, l)

            S_22 = substract_matrices(
                A_22,
                multiply_matrices(
                    multiply_matrices(A_21, A_11_inv, l), A_12, l
                )
            )

            S_22_inv = inverse(S_22, l)

            B_11 = multiply_matrices(
                A_11_inv,
                (i_matrix + multiply_matrices(multiply_matrices(multiply_matrices(A_12, S_22_inv, l), A_21, l), A_11_inv, l)), l  # noqa: E501
            )

            B_12 = multiply_matrices(
                multiply_matrices(-A_11_inv, A_12, l), S_22_inv, l)

            B_21 = multiply_matrices(
                multiply_matrices(-S_22_inv, A_21, l), A_11_inv, l)

            B_22 = S_22_inv

            return merge_matrices(B_11, B_12, B_21, B_22)


@timeit
def call_matrix_inversion_interface(
    matrix: np.ndarray, l: int   # noqa E741
) -> np.ndarray:
    return inverse(matrix, l)

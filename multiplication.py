import numpy as np

from utils import (
    split_matrix_into_quadrants,
    add_matrices,
    substract_matrices,
    timeit,
    check_matrices_preconditions,
    log2
)

import config


def traditional_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    final_matrix = []

    for row_idx_A in range(A.shape[0]):
        final_row = []

        for col_idx_B in range(B.shape[1]):
            row_col_product = 0
            col_elem_B = 0

            for row_elem_A in range(A.shape[0]):
                row_col_product += \
                    A[row_idx_A][row_elem_A]*B[col_elem_B][col_idx_B]

                config.FLOATING_POINT_OPERATIONS += 1

                col_elem_B = col_elem_B + 1

            config.FLOATING_POINT_OPERATIONS += 1
            final_row.append(row_col_product)
        final_matrix.append(final_row)

    return np.array(final_matrix)


def strassen_2D(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    P1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    P2 = (A[1][0] + A[1][1]) * B[0][0]
    P3 = A[0][0] * (B[0][1] - B[1][1])
    P4 = A[1][1] * (B[1][0] - B[0][0])
    P5 = (A[0][0] + A[0][1]) * B[1][1]
    P6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
    P7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])

    return np.array([
                [P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]
           ])


def strassen_recursive_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    config.FLOATING_POINT_OPERATIONS += 25

    # Base Case
    if A.shape == (2, 2):
        return strassen_2D(A, B)

    A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(A)
    B_11, B_12, B_21, B_22 = split_matrix_into_quadrants(B)

    P1 = strassen_recursive_algorithm(
        add_matrices(A_11, A_22),
        add_matrices(B_11, B_22)
    )

    P2 = strassen_recursive_algorithm(
        add_matrices(A_21, A_22), B_11
    )

    P3 = strassen_recursive_algorithm(
        A_11, substract_matrices(B_12, B_22)
    )

    P4 = strassen_recursive_algorithm(
        A_22, substract_matrices(B_21, B_11)
    )

    P5 = strassen_recursive_algorithm(
        add_matrices(A_11, A_12), B_22
    )

    P6 = strassen_recursive_algorithm(
        substract_matrices(A_21, A_11),
        add_matrices(B_11, B_12)
    )

    P7 = strassen_recursive_algorithm(
        substract_matrices(A_12, A_22),
        add_matrices(B_21, B_22)
    )

    # Top Left Quadrant
    C_11 = add_matrices(
        P1, add_matrices(substract_matrices(P4, P5), P7))

    # Top Right Quadrant
    C_12 = add_matrices(P3, P5)

    # Bottom Left Quadrant
    C_21 = add_matrices(P2, P4)

    # Bottom Right Quadrant
    C_22 = add_matrices(
        substract_matrices(P1, P2),
        add_matrices(P3, P6)
    )

    final_matrix = []

    for idx in range(len(C_12)):
        # Construct the top of the final matrix
        # Top = Top Left Quadrant + Top Right Quadrant
        final_matrix.append(list(C_11[idx]) + list(C_12[idx]))

    for idx in range(len(C_22)):
        # Construct the bottom of the final matrix
        # Bottom = Bottom Left Quadrant + Bottom Right Quadrant
        final_matrix.append(list(C_21[idx]) + list(C_22[idx]))

    return np.array(final_matrix)


def multiply_matrices(
        A: np.ndarray,
        B: np.ndarray,
        l: int  # noqa: E741
) -> np.array:
    if len(A) == 1 and len(B) == 1:
        return np.array([list(A)[0] * list(B)[0]])

    check_matrices_preconditions(A, B)

    matrix_order = A.shape[0]
    k = log2(matrix_order)

    if k <= l:
        return traditional_algorithm(A, B)
    elif k > l:
        return strassen_recursive_algorithm(A, B)


@timeit
def call_matrix_mutliplication_interface(
    A: np.ndarray, B: np.ndarray,
    l: int  # noqa: E741
) -> np.array:
    return multiply_matrices(A, B, l)

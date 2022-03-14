import sys
import math
import threading
import numpy as np
import time
from functools import wraps

from typing import Tuple

sys.setrecursionlimit(10**7)
threading.stack_size(2**27)

global FLOATING_POINT_OPERATIONS


def timeit(func):
    is_evaluating = False

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        nonlocal is_evaluating
        if is_evaluating:
            return func(*args, **kwargs)
        else:
            start_time = time.perf_counter()
            is_evaluating = True
            try:
                result = func(*args, **kwargs)
            finally:
                is_evaluating = False

            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{func.__name__} took {total_time:.4f} seconds')
            return result

    return timeit_wrapper


@timeit
def traditional_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    global FLOATING_POINT_OPERATIONS

    final_matrix = []

    for row_idx_A in range(A.shape[0]):
        final_row = []

        for col_idx_B in range(B.shape[1]):
            row_col_product = 0
            col_elem_B = 0

            for row_elem_A in range(A.shape[0]):
                row_col_product += \
                    A[row_idx_A][row_elem_A]*B[col_elem_B][col_idx_B]

                col_elem_B = col_elem_B + 1

                # Multiplication
                FLOATING_POINT_OPERATIONS += 1
            # Addition
            FLOATING_POINT_OPERATIONS += 1

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


def split_matrix_into_quadrants(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(A)
    split_point = size // 2

    top_left = [
        [A[row_idx][col_idx] for col_idx in range(split_point)]
        for row_idx in range(split_point)
    ]

    top_right = [
        [A[row_idx][col_idx] for col_idx in range(split_point, size)]
        for row_idx in range(split_point)
    ]

    bottom_left = [
        [A[row_idx][col_idx] for col_idx in range(split_point)]
        for row_idx in range(split_point, size)
    ]

    bottom_right = [
        [A[row_idx][col_idx] for col_idx in range(split_point, size)]
        for row_idx in range(split_point, size)
    ]

    return np.array(top_left), np.array(top_right), \
        np.array(bottom_left), np.array(bottom_right)


def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    final_matrix = [
        [
            A[row_idx][col_idx] + B[row_idx][col_idx]
            for col_idx in range(len(A[row_idx]))
        ]
        for row_idx in range(len(A))
    ]

    return np.array(final_matrix)


def substract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    final_matrix = [
        [
            A[row_idx][col_idx] - B[row_idx][col_idx]
            for col_idx in range(len(A[row_idx]))
        ]
        for row_idx in range(len(A))
    ]

    return np.array(final_matrix)


@timeit
def strassen_recursive_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    global FLOATING_POINT_OPERATIONS

    # Each invokation of matrix multiplication using Strassen Algorithm
    # is a recursive multiplication: 7 multiplications and 18 additions
    FLOATING_POINT_OPERATIONS += 25

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
        add_matrices(A_11, A_22), B_22
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
    C_11 = substract_matrices(
        add_matrices(P1, P4),
        add_matrices(P5, P7)
    )

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


def log2(x: int) -> float:
    return math.log10(x) / math.log10(2)


def is_power_of_two(n: int) -> bool:
    return math.ceil(log2(n)) == math.floor(log2(n))


def check_matrix_is_square(A: np.ndarray) -> bool:
    return A.shape[0] == A.shape[1]


def check_matrix_order_is_power_of_two(A: np.ndarray) -> bool:
    return is_power_of_two(A.shape[0]) and is_power_of_two(A.shape[1])


def check_matrix_mul_condition(A: np.ndarray, B: np.ndarray) -> bool:
    return A.shape[1] == B.shape[0]


def check_matrices_preconditions(A: np.ndarray, B: np.ndarray) -> bool:
    # Ensure matrices are square ones
    assert check_matrix_is_square(A)
    assert check_matrix_is_square(B)

    # Ensure matrices order is a power of two
    assert check_matrix_order_is_power_of_two(A)
    assert check_matrix_order_is_power_of_two(B)

    # Ensure matrix multiplication condition is satisfied
    assert check_matrix_mul_condition(A, B)


def multiply_matrices(A: np.ndarray, B: np.ndarray, l: int) -> np.array:
    # Handle trivial case
    if len(A) == 1 and len(B) == 1:
        return np.array([list(A)[0] * list(B)[0]])

    check_matrices_preconditions(A, B)

    matrix_order = A.shape[0]
    k = log2(matrix_order)

    if k <= l:
        print("Using Traditional Algorithm.")
        return traditional_algorithm(A, B)
    elif k > l:
        print("Using Recurent Strassen Algorithm.")
        return strassen_recursive_algorithm(A, B)


if __name__ == "__main__":
    FLOATING_POINT_OPERATIONS = 0

    A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    result = multiply_matrices(A, B, l=1)

    if result is not None:
        print(f"Result of Matrix Multiplication: \n {result}")
        print(f"Result shape: {result.shape})")
        print(f"Floating Point operations: {FLOATING_POINT_OPERATIONS}")

        # Post-multiplication check
        if A.shape != (1,) and B.shape != (1,):
            assert result.shape[0] == A.shape[0]
            assert result.shape[1] == A.shape[1]

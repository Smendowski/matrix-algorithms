import time
import math
import numpy as np
from typing import Tuple
from functools import wraps


Quadrants = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def log2(x: int) -> float:
    return math.log10(x) / math.log10(2)


def is_power_of_two(n: int) -> bool:
    return math.ceil(log2(n)) == math.floor(log2(n))


def get_identity_matrix(shape: int) -> np.ndarray:
    return np.identity(shape)


def get_zeros_matrix(shape: int) -> np.ndarray:
    return np.zeros((shape, shape))


def check_matrix_is_square(A: np.ndarray) -> bool:
    return A.shape[0] == A.shape[1]


def check_matrix_order_is_power_of_two(A: np.ndarray) -> bool:
    return is_power_of_two(A.shape[0]) and is_power_of_two(A.shape[1])


def check_matrix_mul_condition(A: np.ndarray, B: np.ndarray) -> bool:
    return A.shape[1] == B.shape[0]


def check_matrices_preconditions(A: np.ndarray, B: np.ndarray) -> bool:
    assert check_matrix_is_square(A)
    assert check_matrix_is_square(B)

    assert check_matrix_order_is_power_of_two(A)
    assert check_matrix_order_is_power_of_two(B)

    assert check_matrix_mul_condition(A, B)


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


def split_matrix_into_quadrants(A: np.ndarray) -> Quadrants:
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


def merge_matrices(
    first_matrix: np.ndarray, second_matrix: np.ndarray,
    third_matrix: np.ndarray, forth_matrix: np.ndarray
 ) -> np.ndarray:
    first_row = np.concatenate((first_matrix, second_matrix), axis=1)
    second_row = np.concatenate((third_matrix, forth_matrix), axis=1)

    return np.concatenate((first_row, second_row), axis=0)


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
            print(f'{func.__name__} took {total_time:.4} seconds.')
            return result

    return timeit_wrapper

import sys
import threading
from enum import Enum

import numpy as np

from multiplication import call_matrix_mutliplication_interface
from inversion import call_matrix_inversion_interface
from factorization import call_matrix_factorization_interface, \
     calculate_determinant_based_on_LU_matrices
import config

sys.setrecursionlimit(10**7)
threading.stack_size(2**27)
np.set_printoptions(precision=2)


class Operations(Enum):
    Multiplication = 1
    Inversion = 2
    Factorization = 3


def main() -> None:
    operation = Operations.Factorization

    if operation.name == Operations.Multiplication.name:
        run_multiplication_logic()
    elif operation.name == Operations.Inversion.name:
        run_inversion_logic()
    elif operation.name == Operations.Factorization.name:
        run_factorization_logic()
    else:
        print(f"Incorrect operation: {operation}")


def run_multiplication_logic():
    matrix_mul_thr = 3
    A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    if config.MEASURE_MULTIPLICATION_TIME:
        result = call_matrix_mutliplication_interface(
            A, B, matrix_mul_thr)
    else:
        result = call_matrix_mutliplication_interface.__wrapped__(
            A, B, matrix_mul_thr)

    if result is not None:
        print(f"Result \n {result}")
        print(f"Result Shape: {result.shape})")
        print(f"FlOps: {config.FLOATING_POINT_OPERATIONS}")

        # Post-multiplication condition check
        if A.shape != (1,) and B.shape != (1,):
            assert result.shape[0] == A.shape[0]
            assert result.shape[1] == B.shape[1]


def run_inversion_logic():
    matrix_mul_thr = 3
    matrix = np.random.randint(1, 10, size=(2**5, 2**5))

    if config.MEASURE_INVERSION_TIME:
        result = call_matrix_inversion_interface(
            matrix, matrix_mul_thr)
    else:
        result = call_matrix_inversion_interface.__wrapped__(
            matrix, matrix_mul_thr)

    if result is not None:
        print(f"Result \n {result}")
        print(f"Result Shape: {result.shape})")
        print(f"FlOps: {config.FLOATING_POINT_OPERATIONS}")


def run_factorization_logic():
    matrix_mul_thr = 3
    matrix = np.random.randint(1, 10, size=(2**5, 2**5))

    if config.MEASURE_FACTORIZATION_TIME:
        L_result, U_result = \
            call_matrix_factorization_interface(
                matrix, matrix_mul_thr)
    else:
        L_result, U_result = \
            call_matrix_factorization_interface.__wrapped__(
                matrix, matrix_mul_thr)

    if L_result is not None and U_result is not None:
        print(f"L \n {L_result}")
        print(f"L Shape: {L_result.shape})")
        print(f"U \n {U_result}")
        print(f"U Shape: {U_result.shape})")
        print(f"FlOps: {config.FLOATING_POINT_OPERATIONS}")
        determinant = calculate_determinant_based_on_LU_matrices(
            L_result, U_result)
        print(f"Determinant: {determinant:.4f}")


if __name__ == '__main__':
    main()

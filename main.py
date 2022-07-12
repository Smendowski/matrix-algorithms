import sys
import threading

import numpy as np

from multiplication import matrix_mutliplication_interface
from inversion import matrix_inversion_interface
from factorization import matrix_factorization_interface, \
     calculate_determinant_based_on_LU_matrices
import config

sys.setrecursionlimit(10**7)
threading.stack_size(2**27)
np.set_printoptions(precision=2)


def main() -> None:
    operation = "factorization"
    if operation == "multiplication":
        run_multiplication_logic()
    elif operation == "inversion":
        run_inversion_logic()
    elif operation == "factorization":
        run_factorization_logic()
    else:
        print(f"Incorrect operation: {operation}")


def run_multiplication_logic():
    A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    result = matrix_mutliplication_interface(A, B, 3)

    if result is not None:
        print(f"Result \n {result}")
        print(f"Result Shape: {result.shape})")
        print(f"FlOps: {config.FLOATING_POINT_OPERATIONS}")

        # Post-multiplication condition check
        if A.shape != (1,) and B.shape != (1,):
            assert result.shape[0] == A.shape[0]
            assert result.shape[1] == B.shape[1]


def run_inversion_logic():
    matrix = np.random.randint(1, 10, size=(2**5, 2**5))

    if config.MEASURE_INVERSION_TIME:
        result = matrix_inversion_interface(matrix, 3)
    else:
        result = matrix_inversion_interface.__wrapped__(matrix, 3)

    if result is not None:
        print(f"Result \n {result}")
        print(f"Result Shape: {result.shape})")
        print(f"FlOps: {config.FLOATING_POINT_OPERATIONS}")


def run_factorization_logic():
    matrix = np.random.randint(1, 10, size=(2**5, 2**5))

    L_result, U_result = matrix_factorization_interface(matrix, l=3)

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

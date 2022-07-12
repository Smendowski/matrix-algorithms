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
    A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    k = 5
    l = 5
    operation = "multiplication"

    if operation == "multiplication":
        run_multiplication_logic()
    elif operation == "inversion":
        run_inversion_logic()
    elif operation == "factorization":
        run_factorization_logic()


if __name__ == '__main__':
    main()


# SECTION FOR MATRIX MULTIPLIATION -> dodać argparse itd.
# if __name__ == "__main__":

#     A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
#     B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

#     # A = np.array([[5, 5], [2, 4]])
#     # B = np.array([[5, 5], [2, 4]])

#     result, elapsed_time, function_name = multiply_matrices(A, B, l=1)

#     if result is not None:
#         print(f"Using: {function_name}, elapsed time: {elapsed_time:.4f}")
#         print(f"Result of Matrix Multiplication: \n {result}")
#         print(f"Result shape: {result.shape})")
#         print(f"Floating Point operations: {config.FLOATING_POINT_OPERATIONS}")

#         # Post-multiplication check
#         if A.shape != (1,) and B.shape != (1,):
#             assert result.shape[0] == A.shape[0]
#             assert result.shape[1] == A.shape[1]


# FOR INVERSION
# if __name__ == "__main__":
#     FLOATING_POINT_OPERATIONS = 0


#     matrix = np.random.randint(1, 10, size=(2**2, 2**2))
#     # matrix = np.array([[5, 5, 5, 100], [5, 5, 5, 200], [1000, 5, 5, 5], [5, 5, 2, 4]])
#     # matrix = np.array([[2, 3], [23, 1]])
#     np.set_printoptions(precision=2)
#     print(f"Input shape: {matrix.shape}")

#     result = inverse_matrix_interface(matrix, l=0) if config.MEASURE_INVERSION_TIME else inverse_matrix_interface.__wrapped__(matrix, l=0)

#     if result is not None:
#         print(matrix)
#         print("\n")
#         print(f"Result of Matrix Inversion: \n {result}")
#         print(f"Result shape: {result.shape})")
#         print(f"Floating Point operations: {FLOATING_POINT_OPERATIONS}")


# FOR FACTORIZATION
# if __name__ == "__main__":
#     k = 3
#     l = 5
#     matrix = np.random.randint(1, 10, size=(2**k, 2**k))

#     print(f"Input Matrix:\n {matrix}")
#     np.set_printoptions(precision=2)
#     print(f"Input shape: {matrix.shape}")
#     L_result, U_result = matrix_factorization_interface(matrix, l=0)
#     print(f"Determinant: {calculate_determinant_based_on_LU_matrices(L_result, U_result):.2f}")


#     if L_result is not None and U_result is not None:
#         print(f"Result of Matrix LU Factorization:")
#         print(f"Matrix L:\n {L_result}")
#         print(f"Matrix U:\n {U_result}")
#         print(f"L shape: {L_result.shape})")
#         print(f"U shape: {U_result.shape})")
#         print(f"Floating Point operations: {config.FLOATING_POINT_OPERATIONS}")
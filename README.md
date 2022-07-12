## 1. Matrix Multiplication
Implementation of matrix muliplication algorithms according to the following description: For matrices smaller than or equal to 2<sup>l</sup> × 2<sup>l</sup> use
traditional algorithm. For matrices larger than 2<sup>l</sup> × 2<sup>l</sup> use Strassen recursive algorithm. <br>
```python
k = log2(matrix_order)

if k <= l:
    return traditional_algorithm(A, B)
elif k > l:
    return strassen_recursive_algorithm(A, B)
```

## 2. Matrix Inversion
Implementation of matrix inversion algorithm with block and recursive approach.

## 3. Matrix LU Factorization
Implementation of matrix LU factorization algorithm with block and recursive approach.

## 4. Usage
### 1. Command Line Interface
```powershell
$ python main.py --help

(...)

options:
  -h, --help
  -m, --multiplication
  -i, --inversion
  -f, --factorization
```
### 2. Configuration file
```python
FLOATING_POINT_OPERATIONS: int = 0
MEASURE_MULTIPLICATION_TIME = False
MEASURE_INVERSION_TIME = False
MEASURE_FACTORIZATION_TIME = False
```

### 3. Define matrix/matrices and threshold l
Recursive inversion and LU factorization use matrix multiplication. Specification on threshold l allow to choose the matrix multiplication method, according to the logic in the *Matrix Multiplication* section. <br>
Functions *run_multiplication_logic*, *run_inversion_logic* and *run_factorization_logic* allow to specify aforementioned threshold and declare matrix/matrices to operate on. Example:
```python
def run_multiplication_logic():
    matrix_mul_thr = 3
    A = np.array([
        [1, 2, 3, 4], [1, 2, 3, 4],
        [1, 2, 3, 4], [1, 2, 3, 4]])
    B = np.array([
        [1, 2, 3, 4], [1, 2, 3, 4],
        [1, 2, 3, 4], [1, 2, 3, 4]])

(...)
```

### 4. Sample Output
- Multiplication
```powershell
Input matrix A:
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
Input matrix B:
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]
A*B Multiplication Result: 
 [10 20 30 40]
 [10 20 30 40]
 [10 20 30 40]
 [10 20 30 40]
Result Shape: (4, 4))
FlOps: 80
```
- Inversion
```powershell
Input matrix A:
 [6 8 1 1]
 [2 6 8 4]
 [1 1 5 9]
 [8 3 5 4]
Inversion Result:
 [ 0.   -0.07 -0.03  0.14]
 [ 0.13  0.04  0.02 -0.11]
 [-0.13  0.15 -0.08  0.07]
 [ 0.06 -0.08  0.16 -0.04]
Result Shape: (4, 4))
FlOps: 156
```

- LU Factorization
```powershell
Input matrix A:
 [5 2 3 9]
 [2 4 6 8]
 [7 4 8 4]
 [3 4 5 5]
L Result:
 [ 1.    0.    0.    0.  ]
 [ 0.4   1.    0.    0.  ]
 [ 1.4   0.38  1.    0.  ]
 [ 0.6   0.88 -0.5   1.  ]
L Shape: (4, 4)
U Result:
 [  5.     2.     3.     9.  ]
 [  0.     3.2    4.8    4.4 ]
 [  0.     0.     2.   -10.25]
 [  0.     0.     0.    -9.38]
U Shape: (4, 4)
FlOps: 96
Determinant: -300.00
```

Each output is associated with the number of floating point operations (FlOps). Furthermore, to enrich results with execution time, provide necessary changes in *config.py*. Example
```python
MEASURE_MULTIPLICATION_TIME = True
```

### 5. Vital conclusion
In the case of the Strassen algorithm, the theoretical computational complexity is significantly less than that of the traditional method. Due to the non-zero time of data transfer between the memory and the processor cache, the profit from the use of the algorithm in practice is visible only for large matrices. The analyzed algorithms are topping examples of confronting theoretical assumptions with their actual implementation.



# PyTorch for Beginners:

## Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's widely used in artificial intelligence applications, particularly in deep learning and neural networks. PyTorch is known for its simplicity, flexibility, and dynamic computational graph, which allows for easier debugging and more intuitive development of complex models.

### Key Use Cases of PyTorch in AI:

1. Deep Learning Research
2. Computer Vision
3. Natural Language Processing
4. Reinforcement Learning
5. Generative Models (e.g., GANs)

PyTorch's popularity stems from its pythonic nature and ease of use, making it a favorite among researchers and developers alike.

## Installation

Let's start by installing PyTorch. The installation process depends on your operating system and whether you want to use CUDA for GPU acceleration.

### Step 1: Check your Python version

PyTorch requires Python 3.6 or later. Open a terminal and run:

```bash
python --version
```

### Step 2: Install PyTorch

For this tutorial, we'll use pip to install PyTorch. Run the following command:

```bash
pip install torch torchvision torchaudio
```

This command installs PyTorch along with torchvision and torchaudio, which are commonly used companion libraries.

To verify the installation, open a Python interpreter and run:

```python
import torch
print(torch.__version__)
```

If it prints a version number without any errors, you've successfully installed PyTorch!

## Basic Usage

Now that we have PyTorch installed, let's start with some basic operations.

### Tensors

Tensors are the fundamental data structure in PyTorch. They're similar to NumPy arrays but can be used on GPUs for accelerated computing.

```python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])
print(x)

# Create a 2D tensor
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(y)

# Get the shape of a tensor
print(x.shape)
print(y.shape)

# Basic operations
z = x + 10
print(z)

# Matrix multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.matmul(a, b)
print(c)
```

---

# Basic Linear Algebra with PyTorch

PyTorch provides powerful tools for linear algebra operations, which are fundamental to many machine learning algorithms.
we'll explore how to perform basic linear algebra operations using PyTorch.

## 1. Creating Vectors and Matrices

Let's start by creating vectors and matrices using PyTorch tensors.

```python
import torch

# Create a vector
v = torch.tensor([1, 2, 3, 4, 5])
print("Vector v:", v)

# Create a matrix
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print("Matrix A:")
print(A)

# Create a 2x3 matrix of ones
B = torch.ones(2, 3)
print("Matrix B:")
print(B)

# Create a 3x3 identity matrix
I = torch.eye(3)
print("Identity matrix I:")
print(I)
```

## 2. Vector Operations

Now, let's perform some common vector operations.

```python
# Vector addition
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
v_sum = v1 + v2
print("v1 + v2 =", v_sum)

# Scalar multiplication
scalar = 2
v_scaled = scalar * v1
print("2 * v1 =", v_scaled)

# Dot product
dot_product = torch.dot(v1, v2)
print("v1 · v2 =", dot_product)

# Vector norm (magnitude)
norm = torch.norm(v1)
print("||v1|| =", norm)

# Normalize a vector
v_normalized = v1 / norm
print("Normalized v1:", v_normalized)
```

## 3. Matrix Operations

Let's explore matrix operations in PyTorch.

```python
# Matrix addition
C = torch.tensor([[1, 2], [3, 4]])
D = torch.tensor([[5, 6], [7, 8]])
M_sum = C + D
print("C + D:")
print(M_sum)

# Matrix multiplication
M_product = torch.matmul(C, D)
print("C * D:")
print(M_product)

# Element-wise multiplication
M_element_wise = C * D
print("C .* D (element-wise):")
print(M_element_wise)

# Matrix transpose
C_transpose = C.t()
print("C^T (transpose of C):")
print(C_transpose)

# Matrix inverse
C_inv = torch.inverse(C)
print("C^-1 (inverse of C):")
print(C_inv)

# Verify inverse
I = torch.matmul(C, C_inv)
print("C * C^-1 (should be close to identity):")
print(I)
```

## 4. Solving Linear Systems

PyTorch can be used to solve systems of linear equations. Let's solve the system Ax = b.

```python
# Define matrix A and vector b
A = torch.tensor([[2, 1], [1, 3]], dtype=torch.float)
b = torch.tensor([4, 5], dtype=torch.float)

# Solve the system Ax = b
x = torch.linalg.solve(A, b)
print("Solution to Ax = b:")
print(x)

# Verify the solution
b_check = torch.matmul(A, x)
print("A * x (should be close to b):")
print(b_check)
```

## 5. Eigenvalues and Eigenvectors

PyTorch can compute eigenvalues and eigenvectors of a matrix.

```python
# Define a matrix
A = torch.tensor([[1, 2], [2, 3]], dtype=torch.float)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)

# Verify Av = λv for the first eigenpair
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
Av1 = torch.matmul(A, v1)
lambda_v1 = lambda1 * v1

print("Av1:")
print(Av1)
print("λ1 * v1:")
print(lambda_v1)
```

## 6. Matrix Decompositions

PyTorch provides functions for various matrix decompositions. Let's look at the Singular Value Decomposition (SVD).

```python
# Create a matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)

# Perform SVD
U, S, V = torch.svd(A)

print("U:")
print(U)
print("S:")
print(S)
print("V:")
print(V)

# Reconstruct the original matrix
A_reconstructed = torch.matmul(U, torch.matmul(torch.diag(S), V.t()))
print("Reconstructed A:")
print(A_reconstructed)

# Check if the reconstruction is close to the original
is_close = torch.allclose(A, A_reconstructed, atol=1e-6)
print("Is the reconstruction close to the original?", is_close)
```

## Conclusion

This tutorial has covered the basics of linear algebra using PyTorch, including:

1. Creating vectors and matrices
2. Vector operations
3. Matrix operations
4. Solving linear systems
5. Computing eigenvalues and eigenvectors
6. Matrix decompositions (SVD)

These operations form the foundation of many machine learning algorithms and are crucial for understanding how neural networks operate at a low level. As you progress in your PyTorch journey, you'll find these operations being used extensively in more complex models and algorithms.

Remember that PyTorch can leverage GPU acceleration for these operations, making it very efficient for large-scale linear algebra computations in deep learning applications.

---

Certainly! I'll create a tutorial on different data types in PyTorch. This tutorial will cover the various tensor types available in PyTorch, their properties, and when to use each type.

# PyTorch Data Types Tutorial

PyTorch offers a variety of data types for tensors, allowing you to optimize your models for both performance and memory usage. Understanding these data types is crucial for efficient deep learning development.

## 1. Introduction to PyTorch Data Types

PyTorch tensors can hold various data types, similar to NumPy arrays. The main categories are:

- Floating-point types
- Integer types
- Boolean type
- Complex number types

Let's explore each of these categories in detail.

## 2. Floating-Point Types

Floating-point types are used for real numbers and are the most common in deep learning.

```python
import torch

# Float32 (default)
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Default float type: {x.dtype}")

# Float64 (double precision)
y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print(f"Double precision: {y.dtype}")

# Float16 (half precision)
z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
print(f"Half precision: {z.dtype}")

# BFloat16 (brain floating point)
w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
print(f"BFloat16: {w.dtype}")
```

- `torch.float32` (default): Standard 32-bit floating-point, good balance of precision and efficiency.
- `torch.float64`: 64-bit double precision, highest precision but more memory usage.
- `torch.float16`: 16-bit half precision, saves memory but less precise.
- `torch.bfloat16`: 16-bit brain floating point, designed for machine learning with a larger dynamic range than float16.

## 3. Integer Types

Integer types are used for whole numbers and can be signed or unsigned.

```python
# Int32 (default integer type)
a = torch.tensor([1, 2, 3])
print(f"Default int type: {a.dtype}")

# Int64 (long)
b = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"64-bit integer: {b.dtype}")

# Int16 (short)
c = torch.tensor([1, 2, 3], dtype=torch.int16)
print(f"16-bit integer: {c.dtype}")

# Int8
d = torch.tensor([1, 2, 3], dtype=torch.int8)
print(f"8-bit integer: {d.dtype}")

# Uint8 (unsigned 8-bit)
e = torch.tensor([1, 2, 3], dtype=torch.uint8)
print(f"Unsigned 8-bit integer: {e.dtype}")
```

- `torch.int32`: 32-bit signed integer.
- `torch.int64`: 64-bit signed integer, useful for very large numbers.
- `torch.int16`: 16-bit signed integer.
- `torch.int8`: 8-bit signed integer.
- `torch.uint8`: 8-bit unsigned integer, useful for image data.

## 4. Boolean Type

Boolean type is used for logical operations and comparisons.

```python
# Boolean
f = torch.tensor([True, False, True])
print(f"Boolean type: {f.dtype}")

# Create boolean tensor from comparison
g = torch.tensor([1, 2, 3]) > 1
print(f"Result of comparison: {g}")
print(f"Type of comparison result: {g.dtype}")
```

## 5. Complex Number Types

PyTorch supports complex numbers for advanced mathematical operations.

```python
# Complex64
h = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(f"Complex64 type: {h.dtype}")

# Complex128
i = torch.tensor([1+2j, 3+4j], dtype=torch.complex128)
print(f"Complex128 type: {i.dtype}")
```

- `torch.complex64`: Complex number with 32-bit real and 32-bit imaginary parts.
- `torch.complex128`: Complex number with 64-bit real and 64-bit imaginary parts.

## 6. Type Conversion

You can convert between different data types using the `to()` method or type-specific casting functions.

```python
# Original tensor
original = torch.tensor([1.5, 2.5, 3.5])
print(f"Original: {original} (dtype: {original.dtype})")

# Convert to int32
int_tensor = original.to(torch.int32)
print(f"To int32: {int_tensor} (dtype: {int_tensor.dtype})")

# Convert to float64
double_tensor = original.double()
print(f"To float64: {double_tensor} (dtype: {double_tensor.dtype})")

# Convert to uint8
uint8_tensor = original.to(torch.uint8)
print(f"To uint8: {uint8_tensor} (dtype: {uint8_tensor.dtype})")
```

## 7. Default Types and Changing Defaults

PyTorch has default types for new tensors. You can check and change these defaults.

```python
# Check default types
print(f"Default float type: {torch.get_default_dtype()}")

# Change default float type
torch.set_default_dtype(torch.float64)
print(f"New default float type: {torch.get_default_dtype()}")

# Create a new tensor to verify
new_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"New tensor dtype: {new_tensor.dtype}")

# Reset to default
torch.set_default_dtype(torch.float32)
```

## 8. Choosing the Right Data Type

Choosing the appropriate data type is crucial for model performance and memory efficiency:

- Use `float32` for most cases as it balances precision and efficiency.
- Consider `float16` or `bfloat16` for large models to save memory, especially when using GPU.
- Use `int64` for large indices or when working with very large tensors.
- Use `uint8` for image data or when you need to save memory and only require values from 0 to 255.
- Use boolean type for masks and logical operations.

## Conclusion

Understanding and correctly using PyTorch's data types is essential for efficient deep learning model development. 
Proper data type selection can significantly impact your model's **memory usage**, **computational speed**, and **numerical precision**. As you develop more complex models, you'll find that choosing the right data types becomes increasingly important, especially when working with **limited GPU memory** or when **fine-tuning** model performance.

---
# Complex Numbers in PyTorch: A Comprehensive Tutorial

Complex numbers are essential in various fields of science and engineering, including signal processing, quantum computing, and certain types of neural networks. PyTorch provides robust support for complex numbers, allowing you to perform computations efficiently on both CPU and GPU.

## 1. Introduction to Complex Numbers in PyTorch

In PyTorch, complex numbers are represented using two main data types:

- `torch.complex64`: Complex number with 32-bit float real and imaginary parts
- `torch.complex128`: Complex number with 64-bit float real and imaginary parts

## 2. Creating Complex Tensors

Let's start by creating complex tensors in different ways:

```python
import torch

# Create a complex tensor from a list of complex numbers
c1 = torch.tensor([1+2j, 3-4j, 5+6j])
print("c1:", c1)
print("c1 dtype:", c1.dtype)

# Create a complex tensor from real and imaginary parts
real = torch.tensor([1., 2., 3.])
imag = torch.tensor([4., 5., 6.])
c2 = torch.complex(real, imag)
print("c2:", c2)
print("c2 dtype:", c2.dtype)

# Create a complex tensor with specific dtype
c3 = torch.tensor([1+2j, 3-4j, 5+6j], dtype=torch.complex128)
print("c3:", c3)
print("c3 dtype:", c3.dtype)

# Create a complex tensor from polar coordinates
magnitude = torch.tensor([1., 2., 3.])
angle = torch.tensor([0., torch.pi/2, torch.pi])
c4 = torch.polar(magnitude, angle)
print("c4:", c4)
print("c4 dtype:", c4.dtype)
```

## 3. Accessing Real and Imaginary Parts

You can easily access the real and imaginary parts of complex tensors:

```python
c = torch.tensor([1+2j, 3-4j, 5+6j])

# Access real part
print("Real part:", c.real)

# Access imaginary part
print("Imaginary part:", c.imag)

# Access magnitude (absolute value)
print("Magnitude:", c.abs())

# Access phase angle
print("Phase angle:", c.angle())
```

## 4. Basic Operations with Complex Numbers

PyTorch supports various operations on complex tensors:

```python
a = torch.tensor([1+2j, 3-4j])
b = torch.tensor([5+6j, 7-8j])

# Addition
print("a + b =", a + b)

# Subtraction
print("a - b =", a - b)

# Multiplication
print("a * b =", a * b)

# Division
print("a / b =", a / b)

# Power
print("a ** 2 =", a ** 2)

# Conjugate
print("Conjugate of a:", a.conj())

# Exponential
print("exp(a) =", torch.exp(a))

# Natural logarithm
print("log(a) =", torch.log(a))
```

## 5. Complex Mathematical Functions

PyTorch provides various mathematical functions that work with complex numbers:

```python
z = torch.tensor([1+1j, 2-2j, 3+3j])

print("Sin:", torch.sin(z))
print("Cos:", torch.cos(z))
print("Tan:", torch.tan(z))

print("Sinh:", torch.sinh(z))
print("Cosh:", torch.cosh(z))
print("Tanh:", torch.tanh(z))

print("Square root:", torch.sqrt(z))
```

## 6. Linear Algebra with Complex Tensors

PyTorch's linear algebra operations support complex tensors:

```python
# Complex matrix
A = torch.tensor([[1+1j, 2+2j], [3+3j, 4+4j]])
# Complex vector
v = torch.tensor([1+1j, 2+2j])

# Matrix-vector multiplication
print("A * v =", torch.matmul(A, v))

# Matrix multiplication
B = torch.tensor([[5+5j, 6+6j], [7+7j, 8+8j]])
print("A * B =", torch.matmul(A, B))

# Determinant
print("det(A) =", torch.det(A))

# Inverse
print("inv(A) =", torch.inverse(A))

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
```

## Conclusion

This tutorial has covered the basics of working with complex numbers in PyTorch, including creation, manipulation, mathematical operations, and applications in deep learning. Complex numbers are powerful tools in various scientific and engineering domains, and PyTorch's support for complex tensors allows you to leverage this power in your deep learning models and computations.

As you explore more advanced topics in signal processing, quantum computing, or specialized neural network architectures, you'll find that understanding complex numbers in PyTorch becomes increasingly valuable.


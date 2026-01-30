# Toán Ứng Dụng trong Trí Tuệ Nhân Tạo

## Giới thiệu
Toán học là nền tảng cốt lõi của Trí tuệ nhân tạo (AI) và Machine Learning (ML). Tài liệu này cung cấp tổng quan chi tiết về các lĩnh vực toán học quan trọng cần thiết để hiểu và phát triển các hệ thống AI.

---

## 📚 Mục lục
1. [Đại số tuyến tính (Linear Algebra)](#1-đại-số-tuyến-tính)
2. [Giải tích (Calculus)](#2-giải-tích)
3. [Xác suất và Thống kê (Probability & Statistics)](#3-xác-suất-và-thống-kê)
4. [Tối ưu hóa (Optimization)](#4-tối-ưu-hóa)
5. [Lý thuyết thông tin (Information Theory)](#5-lý-thuyết-thông-tin)
6. [Lộ trình học tập](#6-lộ-trình-học-tập)
7. [Tài nguyên học tập](#7-tài-nguyên-học-tập)

---

## 1. Đại số tuyến tính

### Tầm quan trọng
Đại số tuyến tính là cơ sở của hầu hết các thuật toán ML/AI, từ xử lý dữ liệu đến deep learning.

### Các khái niệm chính

#### 1.1 Vector và Ma trận
- **Vector**: Đại diện cho điểm dữ liệu trong không gian n chiều
  ```
  v = [x₁, x₂, ..., xₙ]
  ```
- **Ma trận**: Lưu trữ dữ liệu, trọng số trong neural networks
  ```
  A = [a₁₁  a₁₂  ...  a₁ₙ]
      [a₂₁  a₂₂  ...  a₂ₙ]
      [...  ...  ...  ...]
      [aₘ₁  aₘ₂  ...  aₘₙ]
  ```

**Ví dụ Python**:
```python
import numpy as np

# Tạo vector
v = np.array([1, 2, 3, 4])
print(f"Vector: {v}")
print(f"Chiều: {v.shape}")  # (4,)

# Tạo ma trận
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"Ma trận:\n{A}")
print(f"Kích thước: {A.shape}")  # (3, 3)

# Vector row và column
row_vector = np.array([[1, 2, 3]])  # Shape: (1, 3)
col_vector = np.array([[1], [2], [3]])  # Shape: (3, 1)
```

#### 1.2 Các phép toán cơ bản

**1. Cộng vector/ma trận**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B  # [[6, 8], [10, 12]]
```

**2. Nhân vô hướng (Scalar multiplication)**:
```python
A = np.array([[1, 2], [3, 4]])
scaled = 2 * A  # [[2, 4], [6, 8]]
```

**3. Nhân ma trận**:
```python
# Matrix multiplication: (m×n) × (n×p) = (m×p)
A = np.array([[1, 2], [3, 4]])  # 2×2
B = np.array([[5, 6], [7, 8]])  # 2×2
C = np.dot(A, B)  # hoặc A @ B
# C = [[19, 22], [43, 50]]

# Trong Neural Networks
X = np.random.randn(100, 784)  # 100 samples, 784 features
W = np.random.randn(784, 128)  # Weights
b = np.random.randn(128)       # Bias
Y = X @ W + b  # Output: (100, 128)
```

**4. Chuyển vị (Transpose)**:
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T  # [[1, 4],
           #  [2, 5],
           #  [3, 6]]
```

**5. Tích vô hướng (Dot product)**:
```python
# Đo độ tương đồng giữa 2 vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

similarity = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Cosine similarity
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

#### 1.3 Eigenvalues và Eigenvectors

**Định nghĩa**:
```
Av = λv
```
Trong đó:
- A: Ma trận vuông (n×n)
- v: Eigenvector (không thay đổi hướng khi nhân với A)
- λ: Eigenvalue (hệ số scale)

**Ý nghĩa**:
- Eigenvector: Hướng mà ma trận chỉ scale, không xoay
- Eigenvalue: Độ lớn của scaling

**Code Python**:
```python
import numpy as np

# Ma trận
A = np.array([[4, 2],
              [1, 3]])

# Tính eigenvalues và eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")   # [5. 2.]
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    Av = A @ v
    lam_v = lam * v
    
    print(f"Av = {Av}")
    print(f"λv = {lam_v}")
    print(f"Equal: {np.allclose(Av, lam_v)}")
```

**Ứng dụng**: 
- **PCA (Principal Component Analysis)**: Tìm hướng có variance lớn nhất
- **Spectral clustering**: Phân cụm dựa trên eigenvalues
- **Google PageRank**: Eigenvector của ma trận link
- **Stability analysis**: Kiểm tra hệ thống động

#### 1.4 SVD (Singular Value Decomposition)

**Phân rã ma trận**: 
```
A = UΣV^T
```
Trong đó:
- A: Ma trận gốc (m×n)
- U: Ma trận trực giao trái (m×m) - left singular vectors
- Σ: Ma trận đường chéo (m×n) - singular values
- V^T: Ma trận trực giao phải (n×n) - right singular vectors

**Code Python**:
```python
import numpy as np
from numpy.linalg import svd

# Ma trận dữ liệu
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print(f"Shape A: {A.shape}")  # (4, 3)

# SVD
U, S, VT = svd(A, full_matrices=False)

print(f"U shape: {U.shape}")    # (4, 3)
print(f"S shape: {S.shape}")    # (3,)
print(f"VT shape: {VT.shape}")  # (3, 3)

# Reconstruct A
S_matrix = np.diag(S)
A_reconstructed = U @ S_matrix @ VT
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# Low-rank approximation (compression)
k = 2  # Giữ lại 2 singular values lớn nhất
A_compressed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
print(f"Compression ratio: {k * (U.shape[0] + VT.shape[1]) / A.size}")
```

**Ứng dụng thực tế**:

1. **Image Compression**:
```python
from PIL import Image

# Load image
img = np.array(Image.open('image.jpg').convert('L'))
U, S, VT = svd(img, full_matrices=False)

# Compress với k singular values
k = 50
img_compressed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Save compressed image
Image.fromarray(img_compressed.astype(np.uint8)).save('compressed.jpg')
```

2. **Recommendation Systems (Collaborative Filtering)**:
```python
# User-Item rating matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

U, S, VT = svd(ratings, full_matrices=False)

# Low-rank approximation
k = 2
predicted_ratings = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Predict missing ratings
print("Predicted ratings:")
print(predicted_ratings)
```

3. **Dimensionality Reduction**:
- Giống PCA nhưng không cần center data
- Giảm từ n dimensions xuống k dimensions

#### 1.5 Norms và Distances
- **L1 norm** (Manhattan): |x₁| + |x₂| + ... + |xₙ|
- **L2 norm** (Euclidean): √(x₁² + x₂² + ... + xₙ²)
- **Ứng dụng**: Regularization, đo khoảng cách giữa các điểm dữ liệu

---

## 2. Giải tích

### Tầm quan trọng
Giải tích là nền tảng để tối ưu hóa các mô hình ML thông qua gradient descent và backpropagation.

### Các khái niệm chính

#### 2.1 Đạo hàm (Derivatives)
- **Định nghĩa**: Tốc độ thay đổi của hàm số
  ```
  f'(x) = lim[h→0] (f(x+h) - f(x))/h
  ```
- **Ứng dụng**: Tìm độ dốc để tối ưu hóa hàm loss

#### 2.2 Đạo hàm riêng (Partial Derivatives)
- **Định nghĩa**: Đạo hàm theo một biến khi giữ các biến khác cố định
  ```
  ∂f/∂x, ∂f/∂y
  ```
- **Ứng dụng**: Tính gradient trong không gian đa chiều

#### 2.3 Gradient
- **Vector gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
- **Ý nghĩa**: Hướng tăng nhanh nhất của hàm số
- **Ứng dụng**: Gradient Descent Algorithm

#### 2.4 Chain Rule (Quy tắc dây chuyền)
- **Công thức**: (f∘g)'(x) = f'(g(x)) · g'(x)
- **Ứng dụng**: 
  - Backpropagation trong neural networks
  - Tính đạo hàm của hàm hợp

#### 2.5 Gradient Descent
```python
# Thuật toán cơ bản
θ = θ - α · ∇J(θ)

# θ: tham số cần tối ưu
# α: learning rate
# ∇J(θ): gradient của hàm loss
```

**Các biến thể**:
- **Batch Gradient Descent**: Sử dụng toàn bộ dữ liệu
- **Stochastic Gradient Descent (SGD)**: Sử dụng từng mẫu dữ liệu
- **Mini-batch Gradient Descent**: Sử dụng batch nhỏ

#### 2.6 Taylor Series
- **Công thức**: 
  ```
  f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
  ```
- **Ứng dụng**: Xấp xỉ hàm số phức tạp

#### 2.7 Matrix Calculus

**Jacobian Matrix**:
Ma trận đạo hàm của vector function
```
f: ℝⁿ → ℝᵐ
J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
    [...      ...      ...  ...    ]
    [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ]
```

**Code Python**:
```python
import numpy as np

def f(x):
    """Vector function f: R² → R²"""
    return np.array([
        x[0]**2 + x[1]**2,
        x[0] * x[1]
    ])

def jacobian(x):
    """Jacobian của f tại x"""
    return np.array([
        [2*x[0], 2*x[1]],
        [x[1], x[0]]
    ])

x = np.array([3.0, 4.0])
J = jacobian(x)
print(f"Jacobian tại {x}:\n{J}")

# Numerical verification
h = 1e-7
J_numerical = np.zeros((2, 2))
for i in range(2):
    x_plus = x.copy()
    x_plus[i] += h
    J_numerical[:, i] = (f(x_plus) - f(x)) / h

print(f"Jacobian numerical:\n{J_numerical}")
```

**Hessian Matrix**:
Ma trận đạo hàm bậc 2
```
f: ℝⁿ → ℝ
H = [∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ]
    [∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ]
    [...         ...        ...  ...       ]
    [∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²  ]
```

**Ứng dụng Hessian**:
- **Newton's method**: Tối ưu bậc 2
  ```
  x_{n+1} = x_n - H⁻¹∇f(x_n)
  ```
- **Kiểm tra convexity**: Nếu H positive definite → hàm convex
- **Second-order optimization**: L-BFGS

```python
def f(x):
    """f(x) = x₁² + 2x₂²"""
    return x[0]**2 + 2*x[1]**2

def hessian(x):
    """Hessian của f"""
    return np.array([
        [2, 0],
        [0, 4]
    ])

x = np.array([1.0, 1.0])
H = hessian(x)
print(f"Hessian:\n{H}")

# Check positive definite (convex)
eigenvalues = np.linalg.eigvals(H)
print(f"Eigenvalues: {eigenvalues}")
print(f"Positive definite: {all(eigenvalues > 0)}")
```

#### 2.8 Backpropagation Chi Tiết

**Neural Network Forward Pass**:
```
Layer 1: z₁ = W₁x + b₁
         a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂
         a₂ = σ(z₂)
Output:  ŷ = a₂
Loss:    L = (y - ŷ)²
```

**Backward Pass (Chain Rule)**:
```
∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂W₂
       = ∂L/∂ŷ · σ'(z₂) · a₁ᵀ

∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
       = ∂L/∂ŷ · σ'(z₂) · W₂ᵀ · σ'(z₁) · xᵀ
```

**Code Implementation**:
```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer gradient
        dL_da2 = 2 * (self.a2 - y) / m
        dL_dz2 = dL_da2 * self.sigmoid_derivative(self.z2)
        
        # Hidden layer gradient
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.z1)
        
        # Weight gradients
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)
            
            # Loss
            loss = np.mean((y - y_pred)**2)
            
            # Backward
            self.backward(X, y)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=5000)

# Test
predictions = nn.forward(X)
print(f"Predictions:\n{predictions}")
```

---

## 3. Xác suất và Thống kê

### Tầm quan trọng
Xác suất là cơ sở của machine learning, đặc biệt trong xử lý uncertainty và inference.

### Các khái niệm chính

#### 3.1 Xác suất cơ bản
- **Xác suất**: P(A) ∈ [0, 1]
- **Xác suất có điều kiện**: P(A|B) = P(A∩B)/P(B)
- **Định lý Bayes**: 
  ```
  P(A|B) = P(B|A) · P(A) / P(B)
  ```

#### 3.2 Biến ngẫu nhiên
- **Rời rạc**: Số lần tung đồng xu
- **Liên tục**: Chiều cao, cân nặng

#### 3.3 Các phân phối xác suất quan trọng

**Phân phối Bernoulli**:
- Mô tả thí nghiệm có 2 kết quả (0 hoặc 1)
- P(X=1) = p, P(X=0) = 1-p

**Phân phối Gaussian (Normal)**:
```
f(x) = (1/√(2πσ²)) · e^(-(x-μ)²/(2σ²))
```
- **Ứng dụng**: Mô hình hóa dữ liệu tự nhiên, noise

**Phân phối Multinomial**:
- Mở rộng của Bernoulli cho nhiều lớp
- **Ứng dụng**: Classification problems

**Phân phối Poisson**:
```
P(X=k) = (λᵏ · e⁻ᵏ) / k!
```
- **Ứng dụng**: Đếm số events trong khoảng thời gian
- Ví dụ: Số email nhận trong 1 giờ

**Phân phối Exponential**:
```
f(x) = λe⁻ᵏˣ, x ≥ 0
```
- **Ứng dụng**: Thời gian chờ giữa các events
- Ví dụ: Thời gian giữa 2 cuộc gọi

**Phân phối Beta**:
```
f(x) = x^(α-1)(1-x)^(β-1) / B(α,β)
```
- **Ứng dụng**: Prior distribution cho xác suất trong Bayesian
- Ví dụ: Mô hình A/B testing

**Code Visualization**:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Gaussian
x = np.linspace(-5, 5, 1000)
y_gaussian = stats.norm.pdf(x, loc=0, scale=1)

# Poisson
k = np.arange(0, 20)
y_poisson = stats.poisson.pmf(k, mu=5)

# Exponential
x_exp = np.linspace(0, 5, 1000)
y_exp = stats.expon.pdf(x_exp, scale=1)

# Beta
x_beta = np.linspace(0, 1, 1000)
y_beta = stats.beta.pdf(x_beta, a=2, b=5)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, y_gaussian)
axes[0, 0].set_title('Gaussian Distribution')

axes[0, 1].bar(k, y_poisson)
axes[0, 1].set_title('Poisson Distribution')

axes[1, 0].plot(x_exp, y_exp)
axes[1, 0].set_title('Exponential Distribution')

axes[1, 1].plot(x_beta, y_beta)
axes[1, 1].set_title('Beta Distribution')

plt.tight_layout()
plt.show()
```

#### 3.4 Các thống kê mô tả
- **Mean (Trung bình)**: μ = E[X]
- **Variance (Phương sai)**: σ² = E[(X-μ)²]
- **Standard Deviation (Độ lệch chuẩn)**: σ = √σ²
- **Covariance**: Đo mối quan hệ giữa 2 biến
  ```
  Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
  ```

#### 3.5 Maximum Likelihood Estimation (MLE)
- **Mục tiêu**: Tìm tham số θ sao cho dữ liệu quan sát được có xác suất cao nhất
  ```
  θ̂ = argmax L(θ|data)
  ```

#### 3.6 Bayes Networks
- **Mô hình đồ thị xác suất**: Biểu diễn mối quan hệ có điều kiện
- **Ứng dụng**: 
  - Spam filtering
  - Diagnosis systems
  - Naive Bayes classifier

#### 3.7 Hypothesis Testing

**Quy trình kiểm định giả thuyết**:
1. Đặt giả thuyết null (H₀) và alternative (H₁)
2. Chọn mức significance (α, thường là 0.05)
3. Tính test statistic
4. Tính p-value
5. Kết luận: reject H₀ nếu p-value < α

**T-test**:
```python
from scipy import stats

# One-sample t-test
data = np.array([1.2, 2.3, 1.8, 2.5, 1.9])
population_mean = 2.0
t_stat, p_value = stats.ttest_1samp(data, population_mean)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Two-sample t-test
group1 = np.array([1.2, 2.3, 1.8, 2.5, 1.9])
group2 = np.array([2.1, 2.8, 2.3, 3.0, 2.6])
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

**Chi-Square Test**:
```python
# Test independence của 2 categorical variables
observed = np.array([[10, 20, 30],
                     [6, 9, 17]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-square: {chi2}, P-value: {p_value}")
```

#### 3.8 Confidence Intervals

**Công thức**:
```
CI = x̄ ± z(α/2) · (σ/√n)
```

**Code Python**:
```python
from scipy import stats
import numpy as np

data = np.array([1.2, 2.3, 1.8, 2.5, 1.9, 2.1, 2.0])

# Tính confidence interval 95%
confidence = 0.95
mean = np.mean(data)
std_err = stats.sem(data)
ci = stats.t.interval(confidence, len(data)-1, 
                      loc=mean, scale=std_err)

print(f"Mean: {mean:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

#### 3.9 Correlation và Causation

**Pearson Correlation**:
```
r = Cov(X,Y) / (σₓ · σᵧ)
r ∈ [-1, 1]
```

**Code**:
```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Pearson correlation
r_pearson, p_value = pearsonr(x, y)
print(f"Pearson r: {r_pearson:.3f}, p-value: {p_value:.3f}")

# Spearman correlation (rank-based)
r_spearman, p_value = spearmanr(x, y)
print(f"Spearman r: {r_spearman:.3f}, p-value: {p_value:.3f}")

# Visualization
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'r')
plt.title(f'Correlation: r = {r_pearson:.3f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

**Lưu ý quan trọng**:
- **Correlation ≠ Causation**: Hai biến tương quan không có nghĩa là một biến gây ra biến kia
- Ví dụ: Ice cream sales và drowning deaths có correlation cao (do cùng tăng vào mùa hè)

---

## 4. Tối ưu hóa

### Tầm quan trọng
Tối ưu hóa là cốt lõi của việc training các mô hình ML - tìm tham số tốt nhất để minimize loss.

### Các khái niệm chính

#### 4.1 Convex Optimization
- **Hàm lồi**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- **Ưu điểm**: Có global minimum duy nhất
- **Ứng dụng**: Linear regression, SVM

#### 4.2 Loss Functions

**Regression**:
- **MSE**: L = (1/n)Σ(yᵢ - ŷᵢ)²
- **MAE**: L = (1/n)Σ|yᵢ - ŷᵢ|

**Classification**:
- **Cross-Entropy**: L = -Σyᵢlog(ŷᵢ)
- **Hinge Loss**: L = max(0, 1 - y·ŷ)

#### 4.3 Regularization
**L1 Regularization (Lasso)**:
```
J(θ) = Loss(θ) + λΣ|θᵢ|
```
- **Đặc điểm**: Tạo sparsity (nhiều trọng số = 0)

**L2 Regularization (Ridge)**:
```
J(θ) = Loss(θ) + λΣθᵢ²
```
- **Đặc điểm**: Giảm độ lớn của trọng số

#### 4.4 Advanced Optimization Algorithms
- **Momentum**: Tăng tốc trong hướng nhất quán
- **Adam**: Adaptive learning rate cho từng tham số
- **RMSprop**: Điều chỉnh learning rate dựa trên gradient gần đây

#### 4.5 Constrained Optimization
- **Lagrange Multipliers**: Tối ưu với ràng buộc
  ```
  L(x,λ) = f(x) + λg(x)
  ```
- **Ứng dụng**: SVM optimization

---

## 5. Lý thuyết thông tin

### Tầm quan trọng
Lý thuyết thông tin cung cấp framework để đo lường uncertainty và information content.

### Các khái niệm chính

#### 5.1 Entropy
- **Shannon Entropy**: 
  ```
  H(X) = -Σ P(x)log₂P(x)
  ```
- **Ý nghĩa**: Đo độ bất định/surprise trung bình
- **Ứng dụng**: Decision trees (information gain)

#### 5.2 Cross-Entropy
```
H(p,q) = -Σ p(x)log q(x)
```
- **Ứng dụng**: Loss function trong classification

#### 5.3 KL-Divergence
```
D_KL(P||Q) = Σ P(x)log(P(x)/Q(x))
```
- **Ý nghĩa**: Đo khoảng cách giữa 2 phân phối
- **Ứng dụng**: 
  - Variational autoencoders (VAE)
  - Model evaluation

#### 5.4 Mutual Information
```
I(X;Y) = H(X) - H(X|Y)
```
- **Ý nghĩa**: Lượng thông tin chung giữa 2 biến
- **Ứng dụng**: Feature selection

---

## 6. Lộ trình học tập

### Giai đoạn 1: Nền tảng (2-3 tháng)
1. **Đại số tuyến tính cơ bản**
   - Vector, ma trận, phép toán cơ bản
   - Eigenvalues, eigenvectors
   - Thực hành với NumPy

2. **Giải tích**
   - Đạo hàm, đạo hàm riêng
   - Gradient
   - Chain rule

3. **Xác suất cơ bản**
   - Xác suất, xác suất có điều kiện
   - Định lý Bayes
   - Các phân phối cơ bản

### Giai đoạn 2: Ứng dụng (3-4 tháng)
1. **Tối ưu hóa**
   - Gradient descent và các biến thể
   - Loss functions
   - Regularization

2. **Thống kê nâng cao**
   - MLE
   - Hypothesis testing
   - Bayesian inference

3. **Thực hành**
   - Implement các thuật toán từ đầu
   - Linear regression
   - Logistic regression

### Giai đoạn 3: Nâng cao (3-4 tháng)
1. **Deep Learning Math**
   - Backpropagation chi tiết
   - Optimization algorithms (Adam, RMSprop)
   - Batch normalization

2. **Advanced Topics**
   - Information theory
   - Convex optimization
   - Matrix calculus

---

## 7. Tài nguyên học tập

### Sách giáo khoa
1. **"Mathematics for Machine Learning"** - Marc Peter Deisenroth
   - Free PDF: https://mml-book.github.io/
   - Bao quát toàn diện các chủ đề

2. **"Deep Learning"** - Ian Goodfellow
   - Chapters 2-4 về math foundations
   - Link: https://www.deeplearningbook.org/

3. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - Xác suất và thống kê cho ML

### Khóa học online
1. **Khan Academy**
   - Linear Algebra
   - Calculus
   - Statistics

2. **3Blue1Brown (YouTube)**
   - Essence of Linear Algebra
   - Essence of Calculus
   - Visualization tuyệt vời

3. **Coursera**
   - Mathematics for Machine Learning Specialization
   - Imperial College London

### Công cụ thực hành
```python
# NumPy - Đại số tuyến tính
import numpy as np

# Ma trận
A = np.array([[1, 2], [3, 4]])
# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# SciPy - Tối ưu hóa
from scipy.optimize import minimize

# Matplotlib - Visualization
import matplotlib.pyplot as plt

# TensorFlow/PyTorch - Deep Learning
import tensorflow as tf
import torch
```

### Websites hữu ích
1. **Distill.pub** - Giải thích ML với visualization
2. **colah.github.io** - Blog về neural networks
3. **Towards Data Science** - Tutorials và articles

---

## 📝 Bài tập thực hành

### Đại số tuyến tính
1. Implement matrix multiplication từ đầu
2. Viết thuật toán PCA
3. Tính eigenvalues và eigenvectors

### Giải tích
1. Implement gradient descent cho linear regression
2. Tính gradient của neural network đơn giản
3. Visualize gradient descent trên các hàm khác nhau

### Xác suất
1. Implement Naive Bayes classifier
2. Tính posterior probability với Bayes' theorem
3. Visualize các phân phối xác suất

### Tối ưu hóa
1. So sánh SGD vs Mini-batch GD vs Batch GD
2. Implement Adam optimizer
3. Thêm L1/L2 regularization vào model

---

## 🎯 Checklist kiến thức

### Đại số tuyến tính
- [ ] Hiểu vector và ma trận
- [ ] Thành thạo các phép toán ma trận
- [ ] Eigenvalues và eigenvectors
- [ ] SVD
- [ ] PCA

### Giải tích
- [ ] Đạo hàm và đạo hàm riêng
- [ ] Gradient và chain rule
- [ ] Gradient descent
- [ ] Backpropagation

### Xác suất
- [ ] Xác suất cơ bản và có điều kiện
- [ ] Định lý Bayes
- [ ] Các phân phối xác suất
- [ ] MLE
- [ ] Bayesian inference

### Tối ưu hóa
- [ ] Loss functions
- [ ] Regularization
- [ ] Optimization algorithms
- [ ] Convex optimization

---

## 💡 Tips học tập

1. **Học lý thuyết kết hợp thực hành**
   - Đừng chỉ đọc công thức
   - Implement từ đầu với Python

2. **Visualization**
   - Vẽ đồ thị để hiểu concepts
   - Sử dụng Matplotlib, Plotly

3. **Làm projects**
   - Áp dụng toán vào bài toán thực tế
   - Kaggle competitions

4. **Học theo nhóm**
   - Giải thích cho người khác
   - Thảo luận các vấn đề khó

5. **Kiên nhẫn**
   - Toán học cần thời gian
   - Ôn tập thường xuyên

---

## 📚 Kết luận

Toán học là công cụ không thể thiếu trong AI/ML. Không cần phải master tất cả trước khi bắt đầu, nhưng nên hiểu sâu các concepts cơ bản:

1. **Đại số tuyến tính** - Xử lý dữ liệu đa chiều
2. **Giải tích** - Tối ưu hóa models
3. **Xác suất** - Xử lý uncertainty
4. **Tối ưu hóa** - Training algorithms

Hãy học song song giữa lý thuyết và thực hành, và đừng ngại implement các thuật toán từ đầu để hiểu sâu bên trong!

---

**Tạo bởi**: GitHub Copilot  
**Ngày**: 30/01/2026  
**Phiên bản**: 1.0

# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 1)

## NumPy 고급 정답 (문제 1-20)

### 문제 1 정답: 복잡한 배열 인덱싱

```python
import numpy as np

# 100x100 랜덤 배열 생성
arr = np.random.randint(0, 256, size=(100, 100))

# 방법 1: 인덱싱 사용
chess_pattern = arr[::2, ::2].sum() + arr[1::2, 1::2].sum()
chess_count = arr[::2, ::2].size + arr[1::2, 1::2].size
chess_mean = chess_pattern / chess_count

# 방법 2: 마스크 사용
row_idx, col_idx = np.indices(arr.shape)
mask = (row_idx % 2) == (col_idx % 2)
chess_mean2 = arr[mask].mean()

print(f"체스판 패턴 평균: {chess_mean:.2f}")
print(f"마스크 방법 평균: {chess_mean2:.2f}")
```

---

### 문제 2 정답: 이동 평균

```python
import numpy as np

# 랜덤 배열 생성
arr = np.random.randn(1000)
window_size = 50

# 방법 1: np.convolve 사용
moving_avg = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

# 방법 2: stride_tricks 사용 (더 효율적)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(arr, window_size)
moving_avg2 = windows.mean(axis=1)

print(f"이동 평균 배열 길이: {len(moving_avg)}")
print(f"첫 5개 이동 평균: {moving_avg[:5]}")
print(f"stride_tricks 결과 동일 여부: {np.allclose(moving_avg, moving_avg2)}")
```

---

### 문제 3 정답: 행렬 분해와 재구성

```python
import numpy as np

# 5x5 랜덤 행렬
A = np.random.randn(5, 5)

# SVD 수행
U, s, Vt = np.linalg.svd(A)

# 상위 3개 특이값만 사용
k = 3
s_reduced = np.zeros_like(A, dtype=float)
s_reduced[:k, :k] = np.diag(s[:k])

# 근사 재구성
A_approx = U @ s_reduced @ Vt

# Frobenius norm 오차
error = np.linalg.norm(A - A_approx, 'fro')
relative_error = error / np.linalg.norm(A, 'fro')

print(f"원본 행렬:\n{A}")
print(f"\n특이값: {s}")
print(f"\n근사 행렬:\n{A_approx}")
print(f"\nFrobenius norm 오차: {error:.6f}")
print(f"상대 오차: {relative_error:.6%}")
```

---

### 문제 4 정답: 복잡한 브로드캐스팅

```python
import numpy as np

# 배열 생성
a = np.arange(1, 101).reshape(100, 1)
b = np.arange(1, 101).reshape(1, 100)

# 외적 계산 (브로드캐스팅)
outer = a * b

# 5칸 떨어진 대각선들의 합
diag_sums = []
for offset in [-5, 5]:
    diag = np.diagonal(outer, offset=offset)
    diag_sums.append(diag.sum())

print(f"외적 행렬 shape: {outer.shape}")
print(f"5칸 위 대각선 합: {diag_sums[0]}")
print(f"5칸 아래 대각선 합: {diag_sums[1]}")
print(f"두 대각선 합의 총합: {sum(diag_sums)}")
```

---

### 문제 5 정답: 커스텀 정렬

```python
import numpy as np

# 10x10 랜덤 배열
arr = np.random.randint(1, 101, size=(10, 10))
print("원본 배열:")
print(arr)

# 각 행에 대해 처리
sorted_arr = np.zeros_like(arr)
for i, row in enumerate(arr):
    median = np.median(row)
    smaller = row[row < median]
    equal = row[row == median]
    larger = row[row > median]
    
    # 작은 값들은 내림차순, 큰 값들은 오름차순
    smaller_sorted = np.sort(smaller)[::-1]
    larger_sorted = np.sort(larger)
    
    # 결합
    sorted_arr[i] = np.concatenate([smaller_sorted, equal, larger_sorted])

print("\n정렬된 배열:")
print(sorted_arr)

print(f"\n첫 번째 행 중앙값: {np.median(arr[0])}")
print(f"원본 첫 행: {arr[0]}")
print(f"정렬 후 첫 행: {sorted_arr[0]}")
```

---

### 문제 6 정답: 다차원 히스토그램

```python
import numpy as np

# 정규분포 배열 생성
x = np.random.randn(1000)
y = np.random.randn(1000)

# 2D 히스토그램 계산
H, xedges, yedges = np.histogram2d(x, y, bins=20)

# 가장 빈도가 높은 빈 찾기
max_freq = H.max()
max_idx = np.unravel_index(H.argmax(), H.shape)

# 빈의 중심 좌표 계산
x_center = (xedges[max_idx[0]] + xedges[max_idx[0] + 1]) / 2
y_center = (yedges[max_idx[1]] + yedges[max_idx[1] + 1]) / 2

print(f"2D 히스토그램 shape: {H.shape}")
print(f"최대 빈도수: {max_freq}")
print(f"최대 빈도 빈의 인덱스: {max_idx}")
print(f"최대 빈도 빈의 중심 좌표: ({x_center:.2f}, {y_center:.2f})")
print(f"\n전체 히스토그램 합: {H.sum()}")
```

---

### 문제 7 정답: 행렬의 고유값 문제

```python
import numpy as np

# 랜덤 행렬 생성
A = np.random.randn(7, 7)

# 대칭 행렬로 만들기
A = (A + A.T) / 2

# 양의 정부호 행렬로 만들기
A_positive = A.T @ A + np.eye(7) * 0.1

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A_positive)

# 가장 큰 고유값의 인덱스
max_idx = np.argmax(eigenvalues)
max_eigenvalue = eigenvalues[max_idx]
max_eigenvector = eigenvectors[:, max_idx]

# 고유벡터 정규화
max_eigenvector_normalized = max_eigenvector / np.linalg.norm(max_eigenvector)

print(f"모든 고유값이 양수인지 확인: {np.all(eigenvalues > 0)}")
print(f"\n고유값들: {eigenvalues}")
print(f"\n가장 큰 고유값: {max_eigenvalue:.6f}")
print(f"\n해당 고유벡터 (정규화):\n{max_eigenvector_normalized}")
print(f"\n정규화 검증 (norm=1): {np.linalg.norm(max_eigenvector_normalized):.6f}")

# 고유벡터 검증
verification = A_positive @ max_eigenvector_normalized
expected = max_eigenvalue * max_eigenvector_normalized
print(f"\n고유벡터 방정식 검증 (오차): {np.linalg.norm(verification - expected):.10f}")
```

---

### 문제 8 정답: 복잡한 마스킹

```python
import numpy as np

# 50x50 랜덤 배열
arr = np.random.randn(50, 50)

# 중심점
center = (25, 25)

# 각 점의 좌표 생성
y, x = np.ogrid[:50, :50]

# 중심으로부터의 거리 계산
distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)

# 원형 고리 마스크 (10 <= distance <= 20)
annulus_mask = (distances >= 10) & (distances <= 20)

# 해당 영역의 값들 추출
annulus_values = arr[annulus_mask]

print(f"원형 고리 내 요소 개수: {len(annulus_values)}")
print(f"원형 고리 내 평균: {annulus_values.mean():.6f}")
print(f"원형 고리 내 표준편차: {annulus_values.std():.6f}")
print(f"\n전체 배열 평균: {arr.mean():.6f}")
print(f"\n마스크 True 비율: {annulus_mask.sum() / annulus_mask.size:.2%}")
```

---

### 문제 9 정답: 텐서 연산

```python
import numpy as np

# 3차원 배열 생성
tensor = np.random.randn(10, 20, 30)

# 각 슬라이스에서 10x10 부분행렬 추출 및 행렬식 계산
determinants = []
for i in range(30):
    slice_2d = tensor[:, :10, i]  # 10x10 부분행렬
    det = np.linalg.det(slice_2d)
    determinants.append(det)

determinants = np.array(determinants)

# 절댓값 기준 상위 5개 인덱스
top5_indices = np.argsort(np.abs(determinants))[-5:][::-1]
top5_values = determinants[top5_indices]

print(f"30개 행렬식 값 통계:")
print(f"  평균: {determinants.mean():.6f}")
print(f"  표준편차: {determinants.std():.6f}")
print(f"  최대값: {determinants.max():.6f}")
print(f"  최소값: {determinants.min():.6f}")

print(f"\n상위 5개 행렬식의 인덱스: {top5_indices}")
print(f"상위 5개 행렬식의 값: {top5_values}")
```

---

### 문제 10 정답: 조건부 배열 생성

```python
import numpy as np

def is_prime(n):
    """소수 판별 함수"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def sum_of_digit_squares(n):
    """각 자릿수의 제곱의 합"""
    return sum(int(digit)**2 for digit in str(n))

# 1부터 10000까지
numbers = np.arange(1, 10001)

# 조건을 만족하는 숫자 찾기
result = []
for num in numbers:
    digit_sum = sum_of_digit_squares(num)
    if is_prime(digit_sum):
        result.append(num)

result = np.array(result)

print(f"조건을 만족하는 숫자 개수: {len(result)}")
print(f"첫 20개: {result[:20]}")
print(f"마지막 20개: {result[-20:]}")

# 예시 검증
test_num = result[0]
digit_sum = sum_of_digit_squares(test_num)
print(f"\n검증 - {test_num}의 자릿수 제곱 합: {digit_sum}, 소수 여부: {is_prime(digit_sum)}")
```

---

### 문제 11 정답: 복잡한 reshape와 전치

```python
import numpy as np

# 4차원 배열 생성
arr = np.arange(2*3*4*5).reshape(2, 3, 4, 5)
print(f"원본 shape: {arr.shape}")

# (3, 5, 2, 4)로 재배열
reshaped = arr.reshape(3, 5, 2, 4)
print(f"reshape 후: {reshaped.shape}")

# 축 순서 변경: (2, 0, 3, 1)
transposed = reshaped.transpose(2, 0, 3, 1)
print(f"transpose 후: {transposed.shape}")

# 검증
print(f"\n모든 단계에서 요소 개수: {arr.size}")
print(f"각 단계별 shape:")
print(f"  원본: {arr.shape} = {np.prod(arr.shape)}")
print(f"  Reshaped: {reshaped.shape} = {np.prod(reshaped.shape)}")
print(f"  Transposed: {transposed.shape} = {np.prod(transposed.shape)}")
```

---

### 문제 12 정답: 고급 인덱싱과 팬시 인덱싱

```python
import numpy as np

# 100x100 랜덤 배열
arr = np.random.randn(100, 100)

# 방법 1: triu_indices와 tril_indices 사용
upper_indices = np.triu_indices(100, k=1)  # 대각선 제외
lower_indices = np.tril_indices(100, k=-1)  # 대각선 제외

upper_elements = arr[upper_indices]
lower_elements = arr[lower_indices]

# 평균 차이
upper_mean = upper_elements.mean()
lower_mean = lower_elements.mean()
mean_diff = upper_mean - lower_mean

print(f"상삼각 요소 개수: {len(upper_elements)}")
print(f"하삼각 요소 개수: {len(lower_elements)}")
print(f"대각선 요소 개수: {np.diag(arr).size}")
print(f"\n상삼각 평균: {upper_mean:.6f}")
print(f"하삼각 평균: {lower_mean:.6f}")
print(f"평균 차이: {mean_diff:.6f}")

# 검증
expected_elements = (100 * 99) // 2
print(f"\n검증 - 예상 요소 개수: {expected_elements}")
```

---

### 문제 13 정답: 벡터화된 거리 계산

```python
import numpy as np
from scipy.spatial.distance import cdist

# 1000개의 2D 포인트 생성
points = np.random.randn(1000, 2)

# 브로드캐스팅을 이용한 거리 행렬 계산
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=2))

# 또는 scipy의 cdist 사용 (더 효율적)
distances = cdist(points, points, metric='euclidean')

# 각 포인트에서 자기 자신을 제외한 가장 가까운 5개
sorted_distances = np.sort(distances, axis=1)
nearest_5 = sorted_distances[:, 1:6]  # 0번째는 자기 자신

# 가장 가까운 5개 이웃의 평균 거리
avg_nearest_5 = nearest_5.mean(axis=1)

print(f"포인트 개수: {len(points)}")
print(f"거리 행렬 shape: {distances.shape}")
print(f"\n평균 거리 통계:")
print(f"  전체 평균: {avg_nearest_5.mean():.6f}")
print(f"  표준편차: {avg_nearest_5.std():.6f}")
print(f"  최소값: {avg_nearest_5.min():.6f}")
print(f"  최대값: {avg_nearest_5.max():.6f}")
```

---

### 문제 14 정답: 복잡한 집계 연산

```python
import numpy as np

# 100x50 랜덤 배열
arr = np.random.randn(100, 50)

def max_consecutive_positive(column):
    """한 열에서 연속된 양수 구간의 최대 길이"""
    is_positive = column > 0
    changes = np.diff(np.concatenate(([False], is_positive, [False])).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    if len(starts) == 0:
        return 0
    
    lengths = ends - starts
    return lengths.max()

# 각 열에 대해 적용
max_consecutive = np.apply_along_axis(max_consecutive_positive, axis=0, arr=arr)

print(f"배열 shape: {arr.shape}")
print(f"각 열의 최대 연속 양수 길이: {max_consecutive}")
print(f"\n통계:")
print(f"  평균: {max_consecutive.mean():.2f}")
print(f"  최대: {max_consecutive.max()}")
print(f"  최소: {max_consecutive.min()}")
```

---

### 문제 15 정답: 다항식 피팅

```python
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
true_coeffs = [2, -3, 5, -1]
y_true = 2*x**3 - 3*x**2 + 5*x - 1

# 노이즈 추가
np.random.seed(42)
noise = np.random.randn(100) * 10
y_noisy = y_true + noise

# 5차 다항식 피팅
fitted_coeffs = np.polyfit(x, y_noisy, deg=5)

# 피팅된 다항식으로 예측
y_fitted = np.polyval(fitted_coeffs, x)

# 오차 분석
mse = np.mean((y_fitted - y_true)**2)
r_squared = 1 - (np.sum((y_noisy - y_fitted)**2) / 
                  np.sum((y_noisy - y_noisy.mean())**2))

print("원래 계수 [x³, x², x¹, x⁰]: [2, -3, 5, -1]")
print(f"피팅된 계수 [x⁵, x⁴, x³, x², x¹, x⁰]:")
print(fitted_coeffs)
print(f"\n3차 계수 비교:")
print(f"  원래: 2.0, 피팅: {fitted_coeffs[2]:.4f}")
print(f"  원래: -3.0, 피팅: {fitted_coeffs[3]:.4f}")
print(f"  원래: 5.0, 피팅: {fitted_coeffs[4]:.4f}")
print(f"  원래: -1.0, 피팅: {fitted_coeffs[5]:.4f}")
print(f"\nMSE: {mse:.4f}")
print(f"R²: {r_squared:.4f}")
```

---

# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 2)

## NumPy 고급 정답 계속 (문제 16-20)

### 문제 16 정답: 메모리 효율적인 연산

```python
import numpy as np

# 시뮬레이션
total_size = 1_000_000
chunk_size = 100_000

# 온라인 알고리즘으로 평균과 분산 계산
n_total = 0
mean_online = 0
M2 = 0  # 분산 계산을 위한 누적값

for i in range(0, total_size, chunk_size):
    # 청크 생성
    chunk = np.random.randn(min(chunk_size, total_size - i))
    
    # 온라인 알고리즘 (Welford's algorithm)
    for value in chunk:
        n_total += 1
        delta = value - mean_online
        mean_online += delta / n_total
        M2 += delta * (value - mean_online)

variance_online = M2 / (n_total - 1)
std_online = np.sqrt(variance_online)

print(f"총 요소 개수: {n_total:,}")
print(f"청크 개수: {(total_size + chunk_size - 1) // chunk_size}")
print(f"\n온라인 알고리즘 결과:")
print(f"  평균: {mean_online:.6f}")
print(f"  표준편차: {std_online:.6f}")
```

---

### 문제 17 정답: 복잡한 조건부 연산

```python
import numpy as np
from scipy.ndimage import generic_filter

# 1000x1000 랜덤 배열
arr = np.random.randn(100, 100)

def median_filter_condition(window):
    """중심값이 중앙값보다 크면 유지, 아니면 0"""
    center_idx = len(window) // 2
    center_value = window[center_idx]
    median_value = np.median(window)
    return center_value if center_value > median_value else 0

# generic_filter 적용
result = generic_filter(arr, median_filter_condition, 
                       size=3, mode='constant', cval=0)

print(f"원본 배열 shape: {arr.shape}")
print(f"결과 shape: {result.shape}")

print(f"\n원본 통계:")
print(f"  0이 아닌 값: {np.count_nonzero(arr)}")
print(f"  평균: {arr.mean():.4f}")

print(f"\n필터 후 통계:")
print(f"  0이 아닌 값: {np.count_nonzero(result)}")
print(f"  평균: {result.mean():.4f}")
```

---

### 문제 18 정답: 고급 선형대수

```python
import numpy as np

# 행렬 생성
A = np.random.randn(100, 80)
B = np.random.randn(80, 60)

# 행렬 곱
AB = A @ B

# SVD
U, s, Vt = np.linalg.svd(AB, full_matrices=False)

# 조건수 계산
condition_number = np.linalg.cond(AB)
condition_number_manual = s[0] / s[-1]

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"AB shape: {AB.shape}")

print(f"\nSVD 결과:")
print(f"  U shape: {U.shape}")
print(f"  특이값 개수: {len(s)}")
print(f"  Vt shape: {Vt.shape}")

print(f"\n특이값 통계:")
print(f"  최대 특이값: {s[0]:.4f}")
print(f"  최소 특이값: {s[-1]:.4f}")
print(f"  특이값 범위: {s[0]/s[-1]:.4f}")

print(f"\n조건수: {condition_number:.4f}")
print(f"조건수 (수동 계산): {condition_number_manual:.4f}")

if condition_number > 100:
    print("\n⚠️ 조건수가 크므로 이 행렬은 ill-conditioned입니다.")
else:
    print("\n✓ 조건수가 작으므로 이 행렬은 well-conditioned입니다.")
```

---

### 문제 19 정답: 복잡한 배열 병합

```python
import numpy as np

# 3개의 배열 생성
arr1 = np.random.randn(100, 3)
arr2 = np.random.randn(100, 5)
arr3 = np.random.randn(100, 2)

# 열 방향 연결
combined = np.hstack([arr1, arr2, arr3])

# 각 행의 분산 계산
row_variances = np.var(combined, axis=1)

# 상위 20% 임계값
threshold = np.percentile(row_variances, 80)

# 상위 20% 행 추출
mask = row_variances >= threshold
top_20_percent_rows = combined[mask]

print(f"원본 배열 shapes: {arr1.shape}, {arr2.shape}, {arr3.shape}")
print(f"연결된 배열 shape: {combined.shape}")

print(f"\n행 분산 통계:")
print(f"  평균: {row_variances.mean():.4f}")
print(f"  표준편차: {row_variances.std():.4f}")
print(f"  80 백분위수: {threshold:.4f}")

print(f"\n상위 20% 행 개수: {top_20_percent_rows.shape[0]}")
print(f"상위 20% 행 shape: {top_20_percent_rows.shape}")
```

---

### 문제 20 정답: 고급 통계 연산

```python
import numpy as np

# 정규분포 샘플
np.random.seed(42)
data = np.random.randn(1000)

# 부트스트랩
n_bootstrap = 1000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    # 복원 추출
    resample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means[i] = resample.mean()

# 95% 신뢰구간
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

# 원본 통계
original_mean = data.mean()
original_std = data.std()

print(f"원본 데이터 통계:")
print(f"  샘플 크기: {len(data)}")
print(f"  평균: {original_mean:.4f}")
print(f"  표준편차: {original_std:.4f}")

print(f"\n부트스트랩 결과 ({n_bootstrap}회):")
print(f"  부트스트랩 평균들의 평균: {bootstrap_means.mean():.4f}")
print(f"  부트스트랩 평균들의 표준편차: {bootstrap_means.std():.4f}")

print(f"\n95% 신뢰구간:")
print(f"  하한: {ci_lower:.4f}")
print(f"  상한: {ci_upper:.4f}")
print(f"  구간 폭: {ci_upper - ci_lower:.4f}")

# 이론적 신뢰구간과 비교
theoretical_se = original_std / np.sqrt(len(data))
theoretical_ci = (original_mean - 1.96*theoretical_se, 
                  original_mean + 1.96*theoretical_se)
print(f"\n이론적 95% 신뢰구간:")
print(f"  하한: {theoretical_ci[0]:.4f}")
print(f"  상한: {theoretical_ci[1]:.4f}")
```

---

## Pandas 고급 정답 (문제 21-30)

### 문제 21 정답: 복잡한 GroupBy 연산

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
n_rows = 100_000

df = pd.DataFrame({
    '날짜': pd.date_range('2020-01-01', periods=n_rows//500, freq='D').repeat(500),
    '제품': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n_rows),
    '지역': np.random.choice(['서울', '부산', '대구', '인천', '광주'], n_rows),
    '매출': np.random.randint(1000, 100000, n_rows)
})

# 정렬
df = df.sort_values(['제품', '지역', '날짜']).reset_index(drop=True)

# 30일 이동 평균 계산
df['매출_30일_이동평균'] = df.groupby(['제품', '지역'])['매출'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)

print(f"DataFrame shape: {df.shape}")
print(f"\n첫 50행:")
print(df.head(50))

print(f"\n이동평균 통계:")
print(df['매출_30일_이동평균'].describe())
```

---

### 문제 22 정답: 복잡한 Merge 연산

```python
import pandas as pd
import numpy as np

# 고객 DataFrame
customers = pd.DataFrame({
    '고객ID': range(1, 101),
    '이름': [f'고객{i}' for i in range(1, 101)],
    '지역': np.random.choice(['서울', '부산', '대구'], 100)
})

# 주문 DataFrame
np.random.seed(42)
orders = pd.DataFrame({
    '주문ID': range(1, 501),
    '고객ID': np.random.randint(1, 101, 500),
    '제품ID': np.random.randint(1, 21, 500),
    '수량': np.random.randint(1, 10, 500)
})

# 제품 DataFrame
products = pd.DataFrame({
    '제품ID': range(1, 21),
    '제품명': [f'제품{i}' for i in range(1, 21)],
    '가격': np.random.randint(10000, 100000, 20)
})

# 주문금액 계산
orders_with_price = orders.merge(products, on='제품ID')
orders_with_price['주문금액'] = orders_with_price['수량'] * orders_with_price['가격']

# 고객당 총 주문금액
customer_total = orders_with_price.groupby('고객ID').agg({
    '주문금액': 'sum'
}).rename(columns={'주문금액': '총주문금액'})

# 고객당 가장 많이 구매한 제품
customer_favorite = orders_with_price.groupby(['고객ID', '제품명'])['수량'].sum().reset_index()
customer_favorite = customer_favorite.sort_values('수량', ascending=False).groupby('고객ID').first()
customer_favorite = customer_favorite.rename(columns={'제품명': '최다구매제품', '수량': '최다구매수량'})

# 최종 병합
result = customers.merge(customer_total, on='고객ID', how='left')
result = result.merge(customer_favorite, on='고객ID', how='left')
result['총주문금액'] = result['총주문금액'].fillna(0)

print(f"고객 수: {len(customers)}")
print(f"주문 수: {len(orders)}")
print(f"제품 수: {len(products)}")

print(f"\n최종 결과:")
print(result.head(20))

print(f"\n총주문금액 상위 10명:")
print(result.nlargest(10, '총주문금액')[['이름', '지역', '총주문금액', '최다구매제품']])
```

---

### 문제 23 정답: 시계열 리샘플링과 집계

```python
import pandas as pd
import numpy as np

# 시간별 센서 데이터 생성
date_range = pd.date_range('2024-01-01', periods=8760, freq='h')
df = pd.DataFrame({
    '시간': date_range,
    '온도': 20 + 10 * np.sin(np.arange(8760) * 2 * np.pi / 24) + np.random.randn(8760) * 2,
    '습도': 50 + 20 * np.sin(np.arange(8760) * 2 * np.pi / 24 + np.pi/2) + np.random.randn(8760) * 5,
    '압력': 1013 + np.random.randn(8760) * 5
})

df.set_index('시간', inplace=True)

# 일별 리샘플링
daily = df.resample('D').agg(['mean', 'max', 'min'])

# 주별 리샘플링
weekly = df.resample('W').agg(['mean', 'max', 'min'])

# 월별 리샘플링
monthly = df.resample('ME').agg(['mean', 'max', 'min'])

print("원본 데이터 (시간별):")
print(df.head(24))
print(f"Shape: {df.shape}")

print("\n일별 집계:")
print(daily.head(10))
print(f"Shape: {daily.shape}")

print("\n주별 집계:")
print(weekly.head())
print(f"Shape: {weekly.shape}")

print("\n월별 집계:")
print(monthly)
print(f"Shape: {monthly.shape}")
```

---

### 문제 24 정답: 피봇과 멜트

```python
import pandas as pd
import numpy as np

# Wide format 데이터 생성
np.random.seed(42)
students = [f'학생{i}' for i in range(1, 51)]
df_wide = pd.DataFrame({
    '학생': students,
    '수학': np.random.randint(60, 100, 50),
    '영어': np.random.randint(60, 100, 50),
    '과학': np.random.randint(60, 100, 50),
    '국어': np.random.randint(60, 100, 50),
    '사회': np.random.randint(60, 100, 50)
})

print("원본 (Wide format):")
print(df_wide.head(10))

# Long format으로 변환
df_long = df_wide.melt(id_vars='학생', var_name='과목', value_name='점수')
df_long = df_long.sort_values(['학생', '과목']).reset_index(drop=True)

print("\nLong format:")
print(df_long.head(15))

# 학생별 평균 계산
student_avg = df_long.groupby('학생')['점수'].mean().reset_index()
student_avg.columns = ['학생', '평균']

# Wide format으로 재구성 + 평균 추가
df_wide_with_avg = df_long.pivot(index='학생', columns='과목', values='점수')
df_wide_with_avg = df_wide_with_avg.reset_index()
df_wide_with_avg = df_wide_with_avg.merge(student_avg, on='학생')

# 열 순서 정리
cols = ['학생'] + [col for col in df_wide_with_avg.columns if col not in ['학생', '평균']] + ['평균']
df_wide_with_avg = df_wide_with_avg[cols]

print("\nWide format with 평균:")
print(df_wide_with_avg.head(10))

# 상위 10명
top_10 = df_wide_with_avg.nlargest(10, '평균')
print("\n평균 점수 상위 10명:")
print(top_10)
```

---

### 문제 25 정답: 복잡한 문자열 처리

```python
import pandas as pd
import numpy as np
import random
import string

# 이메일 데이터 생성
np.random.seed(42)
random.seed(42)

def generate_email():
    username_length = random.randint(5, 15)
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    domain = random.choice(['gmail.com', 'naver.com', 'daum.net', 'yahoo.com', 'outlook.com'])
    return f"{username}@{domain}"

df = pd.DataFrame({
    '사용자ID': range(1, 1001),
    '이메일': [generate_email() for _ in range(1000)]
})

# 도메인 추출
df['도메인'] = df['이메일'].str.split('@').str[1]

# 도메인별 사용자 수
domain_counts = df['도메인'].value_counts()

# 가장 긴 이메일
longest_email = df.loc[df['이메일'].str.len().idxmax(), '이메일']
longest_length = len(longest_email)

# 숫자 포함 이메일 비율
has_digit = df['이메일'].str.contains(r'\d', regex=True)
digit_ratio = has_digit.sum() / len(df)

print("데이터 샘플:")
print(df.head(10))

print("\n도메인별 사용자 수:")
print(domain_counts)

print(f"\n가장 긴 이메일: {longest_email}")
print(f"길이: {longest_length}")

print(f"\n숫자 포함 이메일 비율: {digit_ratio:.2%}")
print(f"숫자 포함 이메일 개수: {has_digit.sum()}")

# username 길이 분포
df['username_길이'] = df['이메일'].str.split('@').str[0].str.len()
print("\nUsername 길이 통계:")
print(df['username_길이'].describe())
```

---

# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 3)

## Pandas 고급 정답 계속 (문제 26-40)

### 문제 26 정답: 결측치 고급 처리

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    '연속형1': np.random.randn(n),
    '연속형2': np.random.exponential(2, n),
    '범주형': np.random.choice(['A', 'B', 'C', 'D'], n),
    '시계열': range(n),
    '정수형': np.random.randint(1, 100, n)
})

# 랜덤 결측치 삽입
for col in df.columns:
    missing_idx = np.random.choice(df.index, size=int(0.1 * n), replace=False)
    df.loc[missing_idx, col] = np.nan

print("결측치 현황:")
print(df.isnull().sum())

# 자동 결측치 처리 함수
def auto_fill_missing(series):
    if series.isnull().sum() == 0:
        return series
    
    # 범주형 데이터: 최빈값
    if series.dtype == 'object':
        return series.fillna(series.mode()[0])
    
    # 연속형 데이터
    skewness = series.dropna().skew()
    
    # 왜도가 작으면 평균, 크면 중앙값
    if abs(skewness) < 0.5:
        return series.fillna(series.mean())
    else:
        return series.fillna(series.median())

# 열별 적용
df_filled = df.copy()
for col in df_filled.columns:
    if col == '시계열':
        # 시계열 데이터는 선형 보간
        df_filled[col] = df_filled[col].interpolate(method='linear')
    else:
        df_filled[col] = auto_fill_missing(df_filled[col])

print("\n처리 후 결측치:")
print(df_filled.isnull().sum())

print("\n각 열의 왜도:")
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"{col}: {df[col].dropna().skew():.4f}")
```

---

### 문제 27 정답: 복잡한 날짜/시간 연산

```python
import pandas as pd
import numpy as np

# 거래 데이터 생성
np.random.seed(42)
n_transactions = 5000

df = pd.DataFrame({
    '거래ID': range(1, n_transactions + 1),
    '고객ID': np.random.randint(1, 201, n_transactions),
    '거래일시': pd.date_range('2023-01-01', periods=n_transactions, freq='3h'),
    '금액': np.random.randint(10000, 500000, n_transactions)
})

df['거래일시'] = pd.to_datetime(np.random.choice(df['거래일시'], size=n_transactions, replace=False))
df = df.sort_values('거래일시').reset_index(drop=True)

# 요일 추가
df['요일'] = df['거래일시'].dt.day_name()

# 고객별 분석
customer_analysis = df.groupby('고객ID').agg({
    '거래일시': ['min', 'max', 'count']
}).reset_index()

customer_analysis.columns = ['고객ID', '첫구매일', '마지막구매일', '거래횟수']

# 평균 구매 간격 계산
def calculate_avg_interval(group):
    if len(group) < 2:
        return np.nan
    dates = group['거래일시'].sort_values()
    intervals = dates.diff().dropna()
    return intervals.mean().days

avg_intervals = df.groupby('고객ID').apply(calculate_avg_interval).reset_index()
avg_intervals.columns = ['고객ID', '평균구매간격_일']

# 요일 분포
weekday_dist = df.groupby('고객ID')['요일'].apply(
    lambda x: x.value_counts().to_dict()
).reset_index()
weekday_dist.columns = ['고객ID', '요일분포']

# 최종 병합
result = customer_analysis.merge(avg_intervals, on='고객ID')
result = result.merge(weekday_dist, on='고객ID')

# 고객 생애 기간
result['고객생애기간_일'] = (result['마지막구매일'] - result['첫구매일']).dt.days

print("고객별 분석 결과:")
print(result.head(20))

print("\n전체 통계:")
print(f"평균 구매 간격: {result['평균구매간격_일'].mean():.2f}일")
print(f"평균 고객 생애 기간: {result['고객생애기간_일'].mean():.2f}일")
print(f"평균 거래 횟수: {result['거래횟수'].mean():.2f}회")
```

---

### 문제 28 정답: 윈도우 함수와 순위

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365),
    '카테고리': np.random.choice(['전자', '의류', '식품', '도서'], 365),
    '매출': np.random.randint(100000, 1000000, 365)
})

df = df.sort_values(['카테고리', '날짜']).reset_index(drop=True)

# 카테고리별 윈도우 연산
df['누적합계'] = df.groupby('카테고리')['매출'].cumsum()
df['7일이동평균'] = df.groupby('카테고리')['매출'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
df['30일이동평균'] = df.groupby('카테고리')['매출'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)

# 카테고리 내 순위
df['카테고리내순위'] = df.groupby('카테고리')['매출'].rank(ascending=False, method='min')

# 백분위 순위
df['백분위순위'] = df.groupby('카테고리')['매출'].rank(pct=True)

# 전일 대비 변화
df['전일대비변화'] = df.groupby('카테고리')['매출'].diff()
df['전일대비변화율'] = df.groupby('카테고리')['매출'].pct_change() * 100

print("데이터 샘플:")
print(df.head(30))

print("\n카테고리별 통계:")
summary = df.groupby('카테고리').agg({
    '매출': ['count', 'mean', 'sum'],
    '누적합계': 'max',
    '7일이동평균': 'mean'
})
print(summary)
```

---

### 문제 29 정답: 다중 인덱스 (MultiIndex)

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
products = ['노트북', '스마트폰', '태블릿', '이어폰']
regions = ['서울', '부산', '대구']
months = pd.date_range('2024-01', periods=12, freq='MS')

index = pd.MultiIndex.from_product(
    [products, regions, months],
    names=['제품', '지역', '월']
)

df = pd.DataFrame({
    '판매량': np.random.randint(10, 200, len(index)),
    '매출': np.random.randint(1000000, 50000000, len(index))
}, index=index)

print("MultiIndex DataFrame:")
print(df.head(20))

# 레벨별 집계
print("\n1. 제품별 총 매출:")
product_sales = df.groupby(level='제품').sum()
print(product_sales)

print("\n2. 지역별 평균 판매량:")
region_avg = df.groupby(level='지역')['판매량'].mean()
print(region_avg)

print("\n3. 월별 총 매출:")
monthly_sales = df.groupby(level='월').sum()
print(monthly_sales)

# 다중 레벨 집계
print("\n4. 제품-지역별 총 매출:")
product_region = df.groupby(level=['제품', '지역']).sum()
print(product_region)

# 슬라이싱
print("\n5. 노트북의 모든 데이터:")
laptop_data = df.loc['노트북']
print(laptop_data)

print("\n6. 서울 지역의 2024년 3월 데이터:")
seoul_march = df.loc[(slice(None), '서울', '2024-03'), :]
print(seoul_march)
```

---

### 문제 30 정답: 복잡한 조건부 집계

```python
import pandas as pd
import numpy as np

# 직원 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    '직원ID': range(1, 201),
    '이름': [f'직원{i}' for i in range(1, 201)],
    '부서': np.random.choice(['영업', '개발', '마케팅', '인사'], 200),
    '직급': np.random.choice(['사원', '대리', '과장', '차장', '부장'], 200),
    '나이': np.random.randint(25, 60, 200),
    '연봉': np.random.randint(3000, 10000, 200) * 10000,
    '근속년수': np.random.randint(1, 20, 200),
    '평가점수': np.random.uniform(60, 100, 200)
})

print("원본 데이터 샘플:")
print(df.head(10))

# 복잡한 조건부 집계 1
agg_funcs = {
    '연봉': ['mean', 'median', 'std'],
    '나이': ['mean', 'min', 'max'],
    '근속년수': 'mean',
    '평가점수': ['mean', lambda x: x.quantile(0.75)]
}

result1 = df.groupby(['부서', '직급']).agg(agg_funcs)
print("\n1. 부서-직급별 집계:")
print(result1)

# 복잡한 조건부 집계 2
df['나이대'] = pd.cut(df['나이'], bins=[20, 30, 40, 50, 60], 
                      labels=['20대', '30대', '40대', '50대'])

result2 = df.groupby(['나이대', '부서']).agg({
    '연봉': 'mean',
    '직원ID': 'count',
    '평가점수': 'mean'
}).rename(columns={'직원ID': '인원수'})

print("\n2. 나이대-부서별 집계:")
print(result2)

# 복잡한 조건부 집계 3
high_performers = df[(df['평가점수'] >= 80) & (df['근속년수'] >= 5)]

result3 = high_performers.groupby('부서').agg({
    '직원ID': 'count',
    '연봉': 'mean',
    '평가점수': 'mean'
}).rename(columns={'직원ID': '우수인원'})

print("\n3. 부서별 우수 직원 (평가 80+ & 근속 5년+):")
print(result3)

# 복잡한 조건부 집계 4
def custom_aggregation(group):
    return pd.Series({
        '인원수': len(group),
        '평균연봉': group['연봉'].mean(),
        '최고연봉': group['연봉'].max(),
        '연봉표준편차': group['연봉'].std(),
        '평가우수비율': (group['평가점수'] >= 85).sum() / len(group) * 100
    })

result4 = df.groupby('부서').apply(custom_aggregation)
print("\n4. 부서별 커스텀 집계:")
print(result4)
```

---

### 문제 31 정답: 복잡한 데이터 검증

```python
import pandas as pd
import numpy as np

# 테스트 데이터 생성 (의도적인 문제 포함)
np.random.seed(42)
df = pd.DataFrame({
    '고객ID': list(range(1, 96)) + [10, 20, 30, 40, 50],  # 중복 포함
    '이름': [f'고객{i}' for i in range(1, 96)] + ['고객10', '고객20', '고객30', '고객40', '고객50'],
    '나이': list(np.random.randint(20, 70, 95)) + [np.nan, np.nan, -5, 150, np.nan],
    '소득': list(np.random.randint(2000, 10000, 95) * 10000) + [np.nan, 9999999999, np.nan, np.nan, 1000],
    '가입일': pd.date_range('2020-01-01', periods=100, freq='3D'),
    '이메일': [f'user{i}@example.com' for i in range(95)] + ['invalid', 'test@', np.nan, 'noat.com', '']
})

def comprehensive_data_validation(df):
    """종합적인 데이터 품질 검증"""
    
    report = {}
    
    # 1. 기본 정보
    report['기본정보'] = {
        '전체_행수': len(df),
        '전체_열수': len(df.columns),
        '메모리_사용량_MB': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # 2. 중복 검사
    report['중복'] = {
        '전체_중복행': df.duplicated().sum(),
        '키_중복': {}
    }
    
    for col in df.columns:
        duplicates = df[col].duplicated().sum()
        if duplicates > 0:
            report['중복']['키_중복'][col] = duplicates
    
    # 3. 결측치 검사
    report['결측치'] = {
        col: {
            '개수': df[col].isnull().sum(),
            '비율': df[col].isnull().sum() / len(df) * 100
        }
        for col in df.columns if df[col].isnull().sum() > 0
    }
    
    # 4. 데이터 타입 검사
    report['데이터타입'] = df.dtypes.astype(str).to_dict()
    
    # 5. 수치형 데이터 이상치 검사
    report['이상치'] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            report['이상치'][col] = {
                '개수': len(outliers),
                '하한': lower_bound,
                '상한': upper_bound,
                '이상치_값': outliers.tolist()[:10]
            }
    
    # 6. 범주형 데이터 검사
    report['범주형_분포'] = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        report['범주형_분포'][col] = {
            '고유값_수': df[col].nunique(),
            '상위5개': value_counts.head().to_dict()
        }
    
    # 7. 비즈니스 룰 검증
    report['비즈니스_룰_위반'] = []
    
    if '나이' in df.columns:
        invalid_age = df[(df['나이'] < 0) | (df['나이'] > 120)]
        if len(invalid_age) > 0:
            report['비즈니스_룰_위반'].append({
                '규칙': '나이는 0~120 사이',
                '위반_건수': len(invalid_age),
                '위반_행': invalid_age.index.tolist()
            })
    
    if '이메일' in df.columns:
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        invalid_email = df[~df['이메일'].str.match(email_pattern, na=False)]
        if len(invalid_email) > 0:
            report['비즈니스_룰_위반'].append({
                '규칙': '이메일 형식',
                '위반_건수': len(invalid_email),
                '위반_행': invalid_email.index.tolist()[:10]
            })
    
    return report

# 검증 실행
validation_report = comprehensive_data_validation(df)

print("데이터 품질 검증 보고서")
print("=" * 60)

for category, details in validation_report.items():
    print(f"\n[{category}]")
    if isinstance(details, dict):
        for key, value in details.items():
            print(f"  {key}: {value}")
    else:
        for item in details:
            print(f"  {item}")
```

---
# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 4)

## Pandas 고급 정답 계속 (문제 32-40)

### 문제 32 정답: 롤링 윈도우와 이동 통계

```python
import pandas as pd
import numpy as np

# 주식 가격 데이터 시뮬레이션
np.random.seed(42)
n_days = 500
dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

# 가격 생성 (기하 브라운 운동)
returns = np.random.randn(n_days) * 0.02
price = 100 * np.exp(returns.cumsum())

df = pd.DataFrame({
    '날짜': dates,
    '종가': price
})
df.set_index('날짜', inplace=True)

# 1. 볼린저 밴드
window = 20
df['MA20'] = df['종가'].rolling(window=window).mean()
df['STD20'] = df['종가'].rolling(window=window).std()
df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

# 2. RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['종가'])

# 3. MACD
exp1 = df['종가'].ewm(span=12, adjust=False).mean()
exp2 = df['종가'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Histogram'] = df['MACD'] - df['Signal']

# 4. 추가 이동평균
df['MA5'] = df['종가'].rolling(window=5).mean()
df['MA50'] = df['종가'].rolling(window=50).mean()
df['MA200'] = df['종가'].rolling(window=200).mean()

# 5. 변동성
df['Volatility_20'] = df['종가'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100

print("주식 기술적 지표 계산 결과:")
print(df.tail(30))

print("\n최근 지표 요약:")
recent = df.iloc[-1]
print(f"현재가: {recent['종가']:.2f}")
print(f"MA20: {recent['MA20']:.2f}")
print(f"볼린저 밴드: {recent['Lower_Band']:.2f} ~ {recent['Upper_Band']:.2f}")
print(f"RSI: {recent['RSI']:.2f}")
print(f"MACD: {recent['MACD']:.4f}")
print(f"Signal: {recent['Signal']:.4f}")
print(f"20일 변동성: {recent['Volatility_20']:.2f}%")

print("\n트레이딩 신호:")
if recent['종가'] < recent['Lower_Band']:
    print("  볼린저 밴드: 매수 신호 (과매도)")
elif recent['종가'] > recent['Upper_Band']:
    print("  볼린저 밴드: 매도 신호 (과매수)")
else:
    print("  볼린저 밴드: 중립")

if recent['RSI'] < 30:
    print("  RSI: 매수 신호 (과매도)")
elif recent['RSI'] > 70:
    print("  RSI: 매도 신호 (과매수)")
else:
    print("  RSI: 중립")
```

---

### 문제 33 정답: 데이터 샘플링과 부트스트랩

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    '고객ID': range(1, 10001),
    '나이대': np.random.choice(['20대', '30대', '40대', '50대', '60대'], 10000),
    '지역': np.random.choice(['서울', '경기', '부산', '기타'], 10000),
    '구매금액': np.random.lognormal(mean=10, sigma=1, size=10000)
})

print("원본 데이터:")
print(df.head())

# 1. 계층적 샘플링
def stratified_sample(df, strata_col, sample_size):
    """계층별로 비율을 유지하면서 샘플링"""
    strata_counts = df[strata_col].value_counts()
    strata_proportions = strata_counts / len(df)
    strata_samples = (strata_proportions * sample_size).round().astype(int)
    
    samples = []
    for stratum, n in strata_samples.items():
        stratum_data = df[df[strata_col] == stratum]
        sample = stratum_data.sample(n=min(n, len(stratum_data)), random_state=42)
        samples.append(sample)
    
    return pd.concat(samples, ignore_index=True)

sample_size = 1000
stratified_sample_df = stratified_sample(df, '나이대', sample_size)

print("\n계층적 샘플링 결과:")
print(f"샘플 크기: {len(stratified_sample_df)}")
print("\n원본 나이대 분포:")
print(df['나이대'].value_counts(normalize=True).sort_index())
print("\n샘플 나이대 분포:")
print(stratified_sample_df['나이대'].value_counts(normalize=True).sort_index())

# 2. 부트스트랩
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, confidence=0.95):
    """부트스트랩을 사용한 통계량의 신뢰구간 계산"""
    bootstrap_statistics = []
    
    for i in range(n_bootstrap):
        bootstrap_sample = data.sample(n=len(data), replace=True)
        statistic = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(statistic)
    
    bootstrap_statistics = np.array(bootstrap_statistics)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
    ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
    
    return {
        'estimate': statistic_func(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_std': bootstrap_statistics.std(),
        'bootstrap_distribution': bootstrap_statistics
    }

# 구매금액 평균의 신뢰구간
mean_result = bootstrap_confidence_interval(
    df['구매금액'], 
    lambda x: x.mean(),
    n_bootstrap=1000
)

print("\n부트스트랩 결과 (평균):")
print(f"추정값: {mean_result['estimate']:.2f}")
print(f"95% 신뢰구간: [{mean_result['ci_lower']:.2f}, {mean_result['ci_upper']:.2f}]")
print(f"표준오차: {mean_result['bootstrap_std']:.2f}")
```

---

### 문제 34 정답: 복잡한 조인과 병합

```python
import pandas as pd
import numpy as np

# 4개의 관련 테이블 생성
customers = pd.DataFrame({
    '고객ID': range(1, 101),
    '고객명': [f'고객{i}' for i in range(1, 101)],
    '가입일': pd.date_range('2020-01-01', periods=100, freq='3D')
})

np.random.seed(42)
orders = pd.DataFrame({
    '주문ID': range(1, 201),
    '고객ID': np.random.choice(range(1, 81), 200),
    '주문일': pd.date_range('2024-01-01', periods=200, freq='D')[:200],
    '총액': np.random.randint(10000, 500000, 200)
})

order_details = pd.DataFrame({
    '주문상세ID': range(1, 401),
    '주문ID': np.repeat(range(1, 201), 2),
    '제품ID': np.random.randint(1, 51, 400),
    '수량': np.random.randint(1, 10, 400),
    '단가': np.random.randint(5000, 100000, 400)
})

products = pd.DataFrame({
    '제품ID': range(1, 61),
    '제품명': [f'제품{i}' for i in range(1, 61)],
    '카테고리': np.random.choice(['전자', '의류', '식품', '도서'], 60)
})

print("테이블 크기:")
print(f"고객: {len(customers)}, 주문: {len(orders)}, 주문상세: {len(order_details)}, 제품: {len(products)}")

# 1-4. 다양한 조인
inner_result = customers.merge(orders, on='고객ID', how='inner')
left_result = customers.merge(orders, on='고객ID', how='left')
right_result = customers.merge(orders, on='고객ID', how='right')
outer_result = customers.merge(orders, on='고객ID', how='outer')

print(f"\nInner Join 결과: {len(inner_result)}행")
print(f"Left Join 결과: {len(left_result)}행")
print(f"Right Join 결과: {len(right_result)}행")
print(f"Outer Join 결과: {len(outer_result)}행")

# 5. 복잡한 다중 조인
complex_join = (customers
    .merge(orders, on='고객ID', how='left')
    .merge(order_details, on='주문ID', how='left')
    .merge(products, on='제품ID', how='left')
)

print(f"\n복잡한 다중 조인 결과: {len(complex_join)}행")
print("\n샘플 데이터:")
print(complex_join[['고객명', '주문ID', '제품명', '수량', '단가', '카테고리']].head(20))

# 6. 조인 결과 분석
analysis = pd.DataFrame({
    '조인_타입': ['Inner', 'Left', 'Right', 'Outer', '다중조인'],
    '결과_행수': [len(inner_result), len(left_result), len(right_result), 
                  len(outer_result), len(complex_join)],
    '고유_고객수': [
        inner_result['고객ID'].nunique(),
        left_result['고객ID'].nunique(),
        right_result['고객ID'].nunique(),
        outer_result['고객ID'].nunique(),
        complex_join['고객ID'].nunique()
    ]
})

print("\n조인 결과 비교:")
print(analysis)

# 7. 고객별 주문 통계
customer_stats = complex_join.groupby('고객ID').agg({
    '주문ID': 'nunique',
    '총액': 'sum',
    '제품명': 'count'
}).rename(columns={
    '주문ID': '주문_건수',
    '총액': '총_주문금액',
    '제품명': '구매_제품수'
})

customer_stats = customers.merge(customer_stats, on='고객ID', how='left').fillna(0)

print("\n고객별 통계 (상위 10명):")
print(customer_stats.nlargest(10, '총_주문금액'))
```

---

### 문제 35 정답: 시계열 포맷 변환과 주기성 분석

```python
import pandas as pd
import numpy as np

# 시간별 데이터 생성
np.random.seed(42)
hourly_dates = pd.date_range('2024-01-01', periods=8760, freq='h')

df = pd.DataFrame({
    '시간': hourly_dates,
    '값': 100 + 20 * np.sin(np.arange(8760) * 2 * np.pi / 24) +
          10 * np.sin(np.arange(8760) * 2 * np.pi / (24*7)) +
          np.random.randn(8760) * 5
})

df.set_index('시간', inplace=True)

# 시간 관련 특성 추가
df['시'] = df.index.hour
df['요일'] = df.index.dayofweek
df['월'] = df.index.month
df['주차'] = df.index.isocalendar().week

print("원본 데이터 (시간별):")
print(df.head(24))

# 1. 시간대별 평균
hourly_pattern = df.groupby('시')['값'].mean()
print("\n시간대별 평균:")
print(hourly_pattern)

# 2. 요일별 평균
weekday_pattern = df.groupby('요일')['값'].mean()
weekday_pattern.index = ['월', '화', '수', '목', '금', '토', '일']
print("\n요일별 평균:")
print(weekday_pattern)

# 3. 월별 평균
monthly_pattern = df.groupby('월')['값'].mean()
print("\n월별 평균:")
print(monthly_pattern)

# 4. 일별 리샘플링
daily = df.resample('D').agg({
    '값': ['mean', 'min', 'max', 'std']
})
daily.columns = ['평균', '최소', '최대', '표준편차']
print("\n일별 집계 (처음 10일):")
print(daily.head(10))

# 5. 주별 리샘플링
weekly = df.resample('W').agg({
    '값': ['mean', 'sum', 'count']
})
weekly.columns = ['평균', '합계', '데이터수']
print("\n주별 집계 (처음 10주):")
print(weekly.head(10))

# 6. 시간대-요일 조합
heatmap_data = df.groupby(['요일', '시'])['값'].mean().unstack()
heatmap_data.index = ['월', '화', '수', '목', '금', '토', '일']
print("\n시간대-요일 조합 평균 (0-11시):")
print(heatmap_data.iloc[:, :12])

# 7. 피크 타임
peak_hours = hourly_pattern.nlargest(5)
print("\n피크 시간대 Top 5:")
print(peak_hours)

# 8. 주기별 변동성
print("\n주기별 변동성:")
print(f"시간대별 변동계수: {(hourly_pattern.std() / hourly_pattern.mean()):.4f}")
print(f"요일별 변동계수: {(weekday_pattern.std() / weekday_pattern.mean()):.4f}")
print(f"월별 변동계수: {(monthly_pattern.std() / monthly_pattern.mean()):.4f}")
```

---

# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 5 - 완결편)

## Pandas 고급 정답 계속 (문제 36-40)

### 문제 36 정답: 복잡한 카테고리 인코딩

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    '고객ID': range(1, 1001),
    '도시': np.random.choice(['서울', '부산', '대구', '인천', '광주', '대전'], 1000, 
                             p=[0.4, 0.2, 0.15, 0.15, 0.05, 0.05]),
    '직업': np.random.choice(['회사원', '자영업', '프리랜서', '학생', '주부'], 1000),
    '학력': np.random.choice(['고졸', '대졸', '대학원졸'], 1000, p=[0.3, 0.6, 0.1]),
    '소득수준': np.random.choice(['하', '중하', '중', '중상', '상'], 1000)
})

# 타겟 변수 생성
df['구매여부'] = 0
for idx, row in df.iterrows():
    prob = 0.3
    if row['도시'] == '서울': prob += 0.2
    if row['직업'] == '회사원': prob += 0.15
    if row['학력'] == '대학원졸': prob += 0.1
    df.loc[idx, '구매여부'] = np.random.choice([0, 1], p=[1-prob, prob])

print("원본 데이터:")
print(df.head(10))

# 1. 원-핫 인코딩
onehot_df = pd.get_dummies(df, columns=['도시', '직업', '학력', '소득수준'], 
                           prefix=['도시', '직업', '학력', '소득'])
print("\n1. 원-핫 인코딩:")
print(f"   열 개수: {len(df.columns)} -> {len(onehot_df.columns)}")
print(onehot_df.head())

# 2. 레이블 인코딩
label_df = df.copy()
label_encoders = {}
for col in ['도시', '직업', '학력', '소득수준']:
    le = LabelEncoder()
    label_df[f'{col}_label'] = le.fit_transform(label_df[col])
    label_encoders[col] = le

print("\n2. 레이블 인코딩:")
print(label_df[['도시', '도시_label', '직업', '직업_label']].head(10))

# 3. 타겟 인코딩
target_df = df.copy()
for col in ['도시', '직업', '학력', '소득수준']:
    target_mean = df.groupby(col)['구매여부'].mean()
    target_df[f'{col}_target'] = target_df[col].map(target_mean)

print("\n3. 타겟 인코딩:")
print(target_df[['도시', '도시_target', '직업', '직업_target', '구매여부']].head(10))
print("\n도시별 구매율:")
print(df.groupby('도시')['구매여부'].mean().sort_values(ascending=False))

# 4. 빈도 인코딩
freq_df = df.copy()
for col in ['도시', '직업', '학력', '소득수준']:
    freq_map = df[col].value_counts()
    freq_df[f'{col}_freq'] = freq_df[col].map(freq_map)
    freq_df[f'{col}_freq_norm'] = freq_df[col].map(freq_map / len(df))

print("\n4. 빈도 인코딩:")
print(freq_df[['도시', '도시_freq', '도시_freq_norm', '직업', '직업_freq']].head(10))

# 5. 순서형 인코딩
ordinal_df = df.copy()
학력_order = {'고졸': 1, '대졸': 2, '대학원졸': 3}
소득_order = {'하': 1, '중하': 2, '중': 3, '중상': 4, '상': 5}
ordinal_df['학력_ordinal'] = ordinal_df['학력'].map(학력_order)
ordinal_df['소득수준_ordinal'] = ordinal_df['소득수준'].map(소득_order)

print("\n5. 순서형 인코딩:")
print(ordinal_df[['학력', '학력_ordinal', '소득수준', '소득수준_ordinal']].head(10))

# 6. 인코딩 방법 비교
comparison = pd.DataFrame({
    '방법': ['원본', '원-핫', '레이블', '타겟', '빈도', '순서형'],
    '열_개수': [len(df.columns), len(onehot_df.columns), len(label_df.columns),
               len(target_df.columns), len(freq_df.columns), len(ordinal_df.columns)],
    '메모리_MB': [
        df.memory_usage(deep=True).sum() / 1024**2,
        onehot_df.memory_usage(deep=True).sum() / 1024**2,
        label_df.memory_usage(deep=True).sum() / 1024**2,
        target_df.memory_usage(deep=True).sum() / 1024**2,
        freq_df.memory_usage(deep=True).sum() / 1024**2,
        ordinal_df.memory_usage(deep=True).sum() / 1024**2
    ]
})
print("\n6. 인코딩 방법 비교:")
print(comparison)
```

---

### 문제 37 정답: 복잡한 피벗과 크로스탭

```python
import pandas as pd
import numpy as np

# 판매 데이터 생성
np.random.seed(42)
n_records = 5000

df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=n_records, freq='H')[:n_records],
    '지역': np.random.choice(['서울', '부산', '대구', '인천'], n_records),
    '제품': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_records),
    '카테고리': np.random.choice(['전자', '의류', '식품'], n_records),
    '판매량': np.random.randint(1, 50, n_records),
    '금액': np.random.randint(10000, 500000, n_records)
})

df['월'] = df['날짜'].dt.month
df['요일'] = df['날짜'].dt.day_name()

print("원본 데이터:")
print(df.head(10))

# 1. 기본 피벗
pivot1 = df.pivot_table(values='금액', index='지역', columns='제품', 
                        aggfunc='sum', fill_value=0)
print("\n1. 지역별-제품별 판매금액:")
print(pivot1)

# 2. 다중 집계 함수
pivot2 = df.pivot_table(values='금액', index='지역', columns='제품',
                        aggfunc=['sum', 'mean', 'count'], fill_value=0)
print("\n2. 다중 집계:")
print(pivot2)

# 3. 다중 값 컬럼
pivot3 = df.pivot_table(values=['금액', '판매량'], index='지역', columns='제품',
                        aggfunc={'금액': 'sum', '판매량': 'sum'})
print("\n3. 금액과 판매량 동시 집계:")
print(pivot3)

# 4. 다중 인덱스 피벗
pivot4 = df.pivot_table(values='금액', index=['지역', '카테고리'], columns='제품',
                        aggfunc='sum', fill_value=0, margins=True, margins_name='합계')
print("\n4. 지역-카테고리별, 제품별 판매금액:")
print(pivot4)

# 5. 시계열 피벗
pivot5 = df.pivot_table(values='금액', index='지역', columns='월',
                        aggfunc='sum', fill_value=0)
print("\n5. 지역별 월별 판매금액:")
print(pivot5)

# 6. 크로스탭
crosstab1 = pd.crosstab(df['지역'], df['제품'], values=df['금액'],
                        aggfunc='sum', margins=True)
print("\n6. 크로스탭 - 지역별 제품별 판매금액:")
print(crosstab1)

# 7. 정규화된 크로스탭
crosstab2 = pd.crosstab(df['지역'], df['제품'], normalize='index') * 100
print("\n7. 지역별 제품 구성 비율 (%):")
print(crosstab2.round(2))

# 8. 복잡한 크로스탭
crosstab3 = pd.crosstab([df['지역'], df['카테고리']], df['제품'],
                        values=df['금액'], aggfunc='sum', margins=True)
print("\n8. 지역-카테고리별, 제품별 판매금액:")
print(crosstab3)

# 9. 요일별 패턴
pivot6 = df.pivot_table(values='금액', index='요일', columns='지역',
                        aggfunc=['sum', 'mean'])
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot6 = pivot6.reindex(day_order)
print("\n9. 요일별-지역별 판매 패턴:")
print(pivot6)

# 10. 퍼센트 변화
pivot7 = df.pivot_table(values='금액', index='지역', columns='월', aggfunc='sum')
pivot7_pct = pivot7.pct_change(axis=1) * 100
print("\n10. 지역별 월별 판매금액 전월 대비 변화율 (%):")
print(pivot7_pct.round(2))
```

---

### 문제 38 정답: 메모리 최적화

```python
import pandas as pd
import numpy as np

# 대용량 DataFrame 생성
np.random.seed(42)
n_rows = 1_000_000

df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, n_rows),
    'float_col': np.random.randn(n_rows),
    'category_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
    'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='min'),
    'bool_col': np.random.choice([True, False], n_rows),
    'string_col': np.random.choice([f'문자열{i}' for i in range(10)], n_rows)
})

def print_memory_usage(df, name="DataFrame"):
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024**2
    print(f"\n{name} 메모리 사용량: {total_mb:.2f} MB")
    for col, mem in memory_usage.items():
        if col != 'Index':
            print(f"  {col}: {mem/1024**2:.2f} MB ({df[col].dtype})")

print("원본 DataFrame")
print_memory_usage(df, "원본")

# 최적화 적용
df_optimized = df.copy()

# 1. 정수형 다운캐스팅
df_optimized['int_col'] = df_optimized['int_col'].astype('int8')

# 2. 부동소수점 다운캐스팅
df_optimized['float_col'] = df_optimized['float_col'].astype('float32')

# 3. 카테고리형 변환
df_optimized['category_col'] = df_optimized['category_col'].astype('category')
df_optimized['string_col'] = df_optimized['string_col'].astype('category')

print("\n최적화 후")
print_memory_usage(df_optimized, "최적화 후")

# 메모리 절감량
original_memory = df.memory_usage(deep=True).sum() / 1024**2
optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
reduction = original_memory - optimized_memory
reduction_pct = (reduction / original_memory) * 100

print(f"\n메모리 절감: {reduction:.2f} MB ({reduction_pct:.1f}%)")

# 자동 최적화 함수
def optimize_dataframe(df):
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type == 'int64':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
            else:
                if c_min > -128 and c_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
        
        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')
        
        elif col_type == 'object':
            num_unique = len(df_optimized[col].unique())
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

df_auto = optimize_dataframe(df)
print("\n자동 최적화")
print_memory_usage(df_auto, "자동 최적화")
```

---

### 문제 39 정답: 복잡한 문자열 파싱

```python
import pandas as pd
import numpy as np
import json
import re

# 복잡한 문자열 데이터 생성
np.random.seed(42)

df = pd.DataFrame({
    '고객ID': range(1, 101),
    '이름_전화번호': [f'홍길동|010-1234-{1000+i:04d}' for i in range(100)],
    '주소_JSON': [
        json.dumps({
            '시도': np.random.choice(['서울', '부산', '대구']),
            '구군': f'{np.random.randint(1, 10)}구',
            '우편번호': f'{np.random.randint(10000, 99999)}'
        }, ensure_ascii=False) for _ in range(100)
    ],
    '구매내역_리스트': [
        str([f'제품{np.random.randint(1, 20)}' for _ in range(np.random.randint(1, 5))])
        for _ in range(100)
    ],
    '이메일_날짜': [
        f'user{i}@example.com / 2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}'
        for i in range(1, 101)
    ],
    'HTML_태그': [
        f'<div class="product"><span id="p{i}">상품{i}</span></div>'
        for i in range(1, 101)
    ]
})

print("원본 데이터:")
print(df.head())

# 1. 구분자로 분리
df[['이름', '전화번호']] = df['이름_전화번호'].str.split('|', expand=True)
print("\n1. 이름과 전화번호 분리:")
print(df[['이름', '전화번호']].head())

# 2. JSON 파싱
df['주소_딕셔너리'] = df['주소_JSON'].apply(json.loads)
df['시도'] = df['주소_딕셔너리'].apply(lambda x: x['시도'])
df['구군'] = df['주소_딕셔너리'].apply(lambda x: x['구군'])
df['우편번호'] = df['주소_딕셔너리'].apply(lambda x: x['우편번호'])
print("\n2. JSON 주소 파싱:")
print(df[['시도', '구군', '우편번호']].head())

# 3. 리스트 파싱
df['구매내역_파싱'] = df['구매내역_리스트'].apply(eval)
df['구매개수'] = df['구매내역_파싱'].apply(len)
df['첫번째_구매'] = df['구매내역_파싱'].apply(lambda x: x[0] if x else None)
print("\n3. 구매내역 리스트 파싱:")
print(df[['구매개수', '첫번째_구매']].head(10))

# 4. 복합 패턴 추출
df[['이메일', '가입일']] = df['이메일_날짜'].str.split(' / ', expand=True)
df['가입일'] = pd.to_datetime(df['가입일'])
print("\n4. 이메일과 날짜 추출:")
print(df[['이메일', '가입일']].head())

# 5. 정규표현식으로 HTML 태그 파싱
df['상품ID'] = df['HTML_태그'].str.extract(r'id="p(\d+)"')[0]
df['상품명'] = df['HTML_태그'].str.extract(r'<span[^>]*>(.*?)</span>')[0]
print("\n5. HTML 태그 파싱:")
print(df[['상품ID', '상품명']].head())

# 6. 전화번호 형식 검증
phone_pattern = r'^\d{3}-\d{4}-\d{4}$'
df['전화번호_유효'] = df['전화번호'].str.match(phone_pattern)
df['전화번호_숫자만'] = df['전화번호'].str.replace('-', '')
print("\n6. 전화번호 형식 검증:")
print(df[['전화번호', '전화번호_유효', '전화번호_숫자만']].head())

# 7. 이메일 도메인 추출
df['이메일_도메인'] = df['이메일'].str.split('@').str[1]
print("\n7. 이메일 도메인:")
print(df['이메일_도메인'].value_counts())

# 8. 구매내역 확장
df_exploded = df[['고객ID', '이름', '구매내역_파싱']].copy()
df_exploded = df_exploded.explode('구매내역_파싱')
df_exploded = df_exploded.rename(columns={'구매내역_파싱': '구매제품'})
print("\n8. 구매내역 확장:")
print(df_exploded.head(20))

# 9. 제품별 구매 고객 수
product_customers = df_exploded.groupby('구매제품')['고객ID'].nunique().sort_values(ascending=False)
print("\n9. 제품별 구매 고객 수:")
print(product_customers.head(10))
```

---

### 문제 40 정답: 시계열 예측 준비

```python
import pandas as pd
import numpy as np

# 시계열 데이터 생성
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
n = len(dates)

trend = np.linspace(1000, 2000, n)
seasonality = 300 * np.sin(2 * np.pi * np.arange(n) / 365.25)
weekly = 100 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.randn(n) * 50

sales = trend + seasonality + weekly + noise

df = pd.DataFrame({'날짜': dates, '매출': sales})
df.set_index('날짜', inplace=True)

print("원본 시계열 데이터:")
print(df.head(20))

# 1. 라그 특성
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'매출_lag{lag}'] = df['매출'].shift(lag)

# 2. 롤링 통계
for window in [7, 14, 30]:
    df[f'매출_평균_{window}일'] = df['매출'].rolling(window=window).mean()
    df[f'매출_표준편차_{window}일'] = df['매출'].rolling(window=window).std()
    df[f'매출_최대_{window}일'] = df['매출'].rolling(window=window).max()
    df[f'매출_최소_{window}일'] = df['매출'].rolling(window=window).min()

# 3. 차분
df['매출_차분1'] = df['매출'].diff(1)
df['매출_차분7'] = df['매출'].diff(7)
df['매출_변화율'] = df['매출'].pct_change()

# 4. 지수 이동 평균
df['매출_EMA_7'] = df['매출'].ewm(span=7, adjust=False).mean()
df['매출_EMA_30'] = df['매출'].ewm(span=30, adjust=False).mean()

# 5. 시간 기반 특성
df['연도'] = df.index.year
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek
df['주차'] = df.index.isocalendar().week
df['분기'] = df.index.quarter
df['연초경과일'] = df.index.dayofyear
df['주말여부'] = (df['요일'] >= 5).astype(int)
df['월말여부'] = df.index.is_month_end.astype(int)

# 사인/코사인 인코딩
df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
df['요일_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
df['요일_cos'] = np.cos(2 * np.pi * df['요일'] / 7)

# 6. 통계적 특성
df['매출_범위_7일'] = df['매출_최대_7일'] - df['매출_최소_7일']
df['매출_CV_7일'] = df['매출_표준편차_7일'] / df['매출_평균_7일']
window_30 = df['매출'].rolling(window=30)
df['매출_zscore_30일'] = (df['매출'] - window_30.mean()) / window_30.std()

# 7. 타겟 변수
df['타겟_1일후'] = df['매출'].shift(-1)
df['타겟_7일후'] = df['매출'].shift(-7)
df['타겟_향후7일평균'] = df['매출'].shift(-1).rolling(window=7).mean()

print("\n생성된 특성:")
print(df.head(50))

# 결측치 처리
df_train = df.dropna().copy()
print(f"\n결측치 제거 후 크기: {df_train.shape}")

# 상관관계
feature_cols = [col for col in df_train.columns 
                if col not in ['매출', '타겟_1일후', '타겟_7일후', '타겟_향후7일평균']]
correlations = df_train[feature_cols + ['매출']].corr()['매출'].drop('매출')
correlations = correlations.abs().sort_values(ascending=False)

print("\n매출과의 상관관계 (상위 10개):")
print(correlations.head(10))

print(f"\n총 특성 개수: {len(df_train.columns)}")
print(f"학습 가능한 샘플 수: {len(df_train)}")
```

---

## Part 3: 통합 문제 정답 (문제 41-50)

### 문제 41 정답: 포괄적인 EDA

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
n = 10000

df = pd.DataFrame({
    '고객ID': range(1, n+1),
    '나이': np.random.randint(20, 70, n),
    '성별': np.random.choice(['남', '여'], n),
    '지역': np.random.choice(['서울', '경기', '부산', '대구', '인천', '기타'], n),
    '직업': np.random.choice(['회사원', '자영업', '프리랜서', '학생', '주부', '기타'], n),
    '연봉': np.random.lognormal(mean=8, sigma=0.5, size=n) * 1000,
    '신용점수': np.random.normal(700, 100, n).clip(300, 850),
    '가입기간_월': np.random.randint(1, 120, n),
    '월평균거래액': np.random.lognormal(mean=12, sigma=1, size=n),
    '거래빈도_월': np.random.poisson(lam=5, size=n),
    '이탈여부': np.random.choice([0, 1], n, p=[0.85, 0.15])
})

print("="*80)
print("포괄적인 탐색적 데이터 분석 (EDA)")
print("="*80)

# 1. 데이터 개요
print("\n1. 데이터 기본 정보")
print("-"*80)
print(f"행 수: {len(df):,}")
print(f"열 수: {len(df.columns)}")
print(f"\n데이터 타입:")
print(df.dtypes)
print(f"\n첫 5행:")
print(df.head())

# 2. 결측치 분석
print("\n2. 결측치 분석")
print("-"*80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("결측치 없음")

# 3. 수치형 변수 통계
print("\n3. 수치형 변수 기술통계")
print("-"*80)
print(df.describe())

# 4. 범주형 변수 분포
print("\n4. 범주형 변수 분포")
print("-"*80)
for col in ['성별', '지역', '직업', '이탈여부']:
    print(f"\n{col}:")
    value_counts = df[col].value_counts()
    value_pct = df[col].value_counts(normalize=True) * 100
    dist = pd.DataFrame({'빈도': value_counts, '비율(%)': value_pct})
    print(dist)

# 5. 상관관계 분석
print("\n5. 수치형 변수 간 상관관계")
print("-"*80)
numeric_cols = ['나이', '연봉', '신용점수', '가입기간_월', '월평균거래액', '거래빈도_월', '이탈여부']
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

# 6. 이상치 분석
print("\n6. 이상치 분석")
print("-"*80)
outliers_summary = []
for col in ['연봉', '신용점수', '월평균거래액']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outliers_summary.append({
        '변수': col,
        '하한': lower_bound,
        '상한': upper_bound,
        '이상치_개수': len(outliers),
        '이상치_비율(%)': len(outliers) / len(df) * 100
    })

outliers_df = pd.DataFrame(outliers_summary)
print(outliers_df)

# 7. 타겟 변수 분석
print("\n7. 이탈 고객 분석")
print("-"*80)
churn_rate = df['이탈여부'].mean()
print(f"전체 이탈률: {churn_rate:.2%}")

print("\n범주형 변수별 이탈률:")
for col in ['성별', '지역', '직업']:
    churn_by_cat = df.groupby(col)['이탈여부'].agg(['mean', 'count'])
    churn_by_cat.columns = ['이탈률', '고객수']
    churn_by_cat['이탈률'] = churn_by_cat['이탈률'] * 100
    print(f"\n{col}:")
    print(churn_by_cat.sort_values('이탈률', ascending=False))

# 8. 주요 인사이트
print("\n8. 주요 인사이트")
print("-"*80)
highest_churn_region = df.groupby('지역')['이탈여부'].mean().idxmax()
highest_churn_rate = df.groupby('지역')['이탈여부'].mean().max()
print(f"• 이탈률이 가장 높은 지역: {highest_churn_region} ({highest_churn_rate:.1%})")

avg_tenure = df['가입기간_월'].mean()
print(f"• 평균 가입기간: {avg_tenure:.1f}개월")

churn_customers = df[df['이탈여부'] == 1]
avg_age_churn = churn_customers['나이'].mean()
print(f"• 이탈 고객 평균 나이: {avg_age_churn:.1f}세")
```

---

# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 6 - 문제 42-50)

## 통합 문제 정답 (문제 42-50)

### 문제 42 정답: 데이터 파이프라인 구축

```python
import pandas as pd
import numpy as np

# 원시 데이터 생성
np.random.seed(42)

raw_sales = pd.DataFrame({
    'order_id': range(1, 10001),
    'customer_id': np.random.randint(1, 1001, 10000),
    'product_id': np.random.randint(1, 101, 10000),
    'order_date': pd.date_range('2023-01-01', periods=10000, freq='h'),
    'quantity': np.random.randint(1, 10, 10000),
    'unit_price': np.random.uniform(1000, 100000, 10000)
})

raw_customers = pd.DataFrame({
    'customer_id': range(1, 1001),
    'age': np.random.randint(20, 70, 1000),
    'region': np.random.choice(['서울', '경기', '부산', '기타'], 1000),
    'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D')
})

raw_products = pd.DataFrame({
    'product_id': range(1, 101),
    'category': np.random.choice(['전자', '의류', '식품', '도서'], 100),
    'price': np.random.uniform(1000, 100000, 100)
})

print("="*80)
print("데이터 파이프라인 구축")
print("="*80)

# 1단계: 데이터 로드 및 기본 검증
print("\n1단계: 데이터 로드 및 검증")
print("-"*80)
print(f"주문 데이터: {len(raw_sales):,}행")
print(f"고객 데이터: {len(raw_customers):,}행")
print(f"제품 데이터: {len(raw_products):,}행")

# 결측치 확인
print("\n결측치 확인:")
print(f"  주문: {raw_sales.isnull().sum().sum()}")
print(f"  고객: {raw_customers.isnull().sum().sum()}")
print(f"  제품: {raw_products.isnull().sum().sum()}")

# 2단계: 데이터 전처리
print("\n2단계: 데이터 전처리")
print("-"*80)

# 주문 데이터 전처리
sales_clean = raw_sales.copy()
sales_clean['order_date'] = pd.to_datetime(sales_clean['order_date'])
sales_clean['total_price'] = sales_clean['quantity'] * sales_clean['unit_price']
sales_clean['year'] = sales_clean['order_date'].dt.year
sales_clean['month'] = sales_clean['order_date'].dt.month
sales_clean['quarter'] = sales_clean['order_date'].dt.quarter

print("주문 데이터 전처리 완료")
print(f"  총 매출액 열 추가")
print(f"  날짜 파생 변수 생성 (year, month, quarter)")

# 고객 데이터 전처리
customers_clean = raw_customers.copy()
customers_clean['signup_date'] = pd.to_datetime(customers_clean['signup_date'])
customers_clean['age_group'] = pd.cut(
    customers_clean['age'], 
    bins=[0, 30, 40, 50, 100], 
    labels=['20대', '30대', '40대', '50대+']
)

print("\n고객 데이터 전처리 완료")
print(f"  나이대 그룹 생성")

# 3단계: 데이터 병합
print("\n3단계: 데이터 병합")
print("-"*80)

# 주문 + 고객
merged_data = sales_clean.merge(customers_clean, on='customer_id', how='left')
print(f"주문 + 고객 병합: {len(merged_data):,}행")

# + 제품
merged_data = merged_data.merge(raw_products, on='product_id', how='left')
print(f"+ 제품 병합: {len(merged_data):,}행")

# 4단계: 데이터 변환 및 파생 변수 생성
print("\n4단계: 데이터 변환")
print("-"*80)

# 고객별 집계
customer_metrics = merged_data.groupby('customer_id').agg({
    'order_id': 'count',
    'total_price': 'sum',
    'quantity': 'sum'
}).rename(columns={
    'order_id': '총주문수',
    'total_price': '총구매액',
    'quantity': '총구매수량'
})

customer_metrics['평균주문금액'] = (
    customer_metrics['총구매액'] / customer_metrics['총주문수']
)

print(f"고객별 집계 지표 생성:")
print(f"  총주문수, 총구매액, 총구매수량, 평균주문금액")

# 제품별 집계
product_metrics = merged_data.groupby('product_id').agg({
    'order_id': 'count',
    'quantity': 'sum',
    'total_price': 'sum'
}).rename(columns={
    'order_id': '주문건수',
    'quantity': '판매수량',
    'total_price': '총매출'
})

product_metrics['평균판매가'] = (
    product_metrics['총매출'] / product_metrics['판매수량']
)

print(f"\n제품별 집계 지표 생성:")
print(f"  주문건수, 판매수량, 총매출, 평균판매가")

# 월별 집계
monthly_metrics = merged_data.groupby(['year', 'month']).agg({
    'total_price': 'sum',
    'order_id': 'count',
    'customer_id': 'nunique'
}).rename(columns={
    'total_price': '월매출',
    'order_id': '월주문수',
    'customer_id': '활성고객수'
})

print(f"\n월별 집계 지표 생성:")
print(f"  월매출, 월주문수, 활성고객수")

# 5단계: 최종 분석 리포트 생성
print("\n5단계: 최종 분석 리포트")
print("-"*80)

# 전체 요약
total_revenue = merged_data['total_price'].sum()
total_orders = len(merged_data)
total_customers = merged_data['customer_id'].nunique()
avg_order_value = total_revenue / total_orders

print(f"\n전체 비즈니스 요약:")
print(f"  총 매출: {total_revenue:,.0f}원")
print(f"  총 주문수: {total_orders:,}건")
print(f"  총 고객수: {total_customers:,}명")
print(f"  평균 주문금액: {avg_order_value:,.0f}원")

# 카테고리별 매출
category_sales = merged_data.groupby('category')['total_price'].sum().sort_values(ascending=False)
print(f"\n카테고리별 매출:")
for cat, sales in category_sales.items():
    print(f"  {cat}: {sales:,.0f}원 ({sales/total_revenue*100:.1f}%)")

# 지역별 매출
region_sales = merged_data.groupby('region')['total_price'].sum().sort_values(ascending=False)
print(f"\n지역별 매출:")
for region, sales in region_sales.items():
    print(f"  {region}: {sales:,.0f}원 ({sales/total_revenue*100:.1f}%)")

# 상위 고객
top_customers = customer_metrics.nlargest(10, '총구매액')
print(f"\n상위 10 고객:")
print(top_customers)

# 인기 제품
top_products = product_metrics.nlargest(10, '총매출')
print(f"\n상위 10 제품:")
print(top_products)

# 월별 트렌드
print(f"\n월별 매출 트렌드:")
print(monthly_metrics)

# 6단계: 최종 데이터 저장 준비
print("\n6단계: 파이프라인 완료")
print("-"*80)
print("생성된 데이터셋:")
print(f"  1. merged_data: {merged_data.shape}")
print(f"  2. customer_metrics: {customer_metrics.shape}")
print(f"  3. product_metrics: {product_metrics.shape}")
print(f"  4. monthly_metrics: {monthly_metrics.shape}")

print("\n파이프라인 실행 완료! ✓")
```

---

### 문제 43 정답: A/B 테스트 분석

```python
import pandas as pd
import numpy as np
from scipy import stats

# 데이터 생성
np.random.seed(42)

group_a = pd.DataFrame({
    '사용자ID': range(1, 1001),
    '그룹': 'A',
    '전환': np.random.choice([0, 1], 1000, p=[0.85, 0.15]),
    '체류시간_초': np.random.normal(120, 30, 1000).clip(10, 300),
    '페이지뷰': np.random.poisson(5, 1000)
})

group_b = pd.DataFrame({
    '사용자ID': range(1001, 2001),
    '그룹': 'B',
    '전환': np.random.choice([0, 1], 1000, p=[0.80, 0.20]),
    '체류시간_초': np.random.normal(135, 30, 1000).clip(10, 300),
    '페이지뷰': np.random.poisson(6, 1000)
})

ab_test_data = pd.concat([group_a, group_b], ignore_index=True)

print("="*80)
print("A/B 테스트 분석")
print("="*80)

# 1. 기본 통계
print("\n1. 그룹별 기본 통계")
print("-"*80)

summary = ab_test_data.groupby('그룹').agg({
    '사용자ID': 'count',
    '전환': ['sum', 'mean'],
    '체류시간_초': ['mean', 'std'],
    '페이지뷰': ['mean', 'std']
})

summary.columns = ['사용자수', '전환수', '전환율', '평균체류시간', '체류시간_표준편차', 
                    '평균페이지뷰', '페이지뷰_표준편차']

print(summary)

# 2. 전환율 비교 및 검정
print("\n2. 전환율 A/B 테스트")
print("-"*80)

# 그룹별 데이터
conversions_a = group_a['전환'].sum()
total_a = len(group_a)
conversion_rate_a = conversions_a / total_a

conversions_b = group_b['전환'].sum()
total_b = len(group_b)
conversion_rate_b = conversions_b / total_b

# 카이제곱 검정
contingency_table = pd.crosstab(ab_test_data['그룹'], ab_test_data['전환'])
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)

print(f"그룹 A 전환율: {conversion_rate_a:.2%} ({conversions_a}/{total_a})")
print(f"그룹 B 전환율: {conversion_rate_b:.2%} ({conversions_b}/{total_b})")
print(f"전환율 차이: {(conversion_rate_b - conversion_rate_a):.2%}")
print(f"상대적 증가율: {((conversion_rate_b / conversion_rate_a - 1) * 100):.1f}%")

print(f"\n카이제곱 검정:")
print(f"  카이제곱 통계량: {chi2:.4f}")
print(f"  p-value: {p_value_chi2:.4f}")

if p_value_chi2 < 0.05:
    print(f"  결과: 통계적으로 유의미한 차이 있음 (p < 0.05)")
else:
    print(f"  결과: 통계적으로 유의미한 차이 없음 (p >= 0.05)")

# 3. 체류시간 비교 및 검정
print("\n3. 체류시간 A/B 테스트")
print("-"*80)

mean_time_a = group_a['체류시간_초'].mean()
mean_time_b = group_b['체류시간_초'].mean()

# t-검정
t_stat, p_value_t = stats.ttest_ind(group_a['체류시간_초'], group_b['체류시간_초'])

print(f"그룹 A 평균 체류시간: {mean_time_a:.1f}초")
print(f"그룹 B 평균 체류시간: {mean_time_b:.1f}초")
print(f"체류시간 차이: {(mean_time_b - mean_time_a):.1f}초")
print(f"상대적 증가율: {((mean_time_b / mean_time_a - 1) * 100):.1f}%")

print(f"\nt-검정:")
print(f"  t-통계량: {t_stat:.4f}")
print(f"  p-value: {p_value_t:.4f}")

if p_value_t < 0.05:
    print(f"  결과: 통계적으로 유의미한 차이 있음 (p < 0.05)")
else:
    print(f"  결과: 통계적으로 유의미한 차이 없음 (p >= 0.05)")

# 4. 페이지뷰 비교 및 검정
print("\n4. 페이지뷰 A/B 테스트")
print("-"*80)

mean_pv_a = group_a['페이지뷰'].mean()
mean_pv_b = group_b['페이지뷰'].mean()

# Mann-Whitney U 검정 (비모수 검정)
u_stat, p_value_u = stats.mannwhitneyu(group_a['페이지뷰'], group_b['페이지뷰'])

print(f"그룹 A 평균 페이지뷰: {mean_pv_a:.2f}")
print(f"그룹 B 평균 페이지뷰: {mean_pv_b:.2f}")
print(f"페이지뷰 차이: {(mean_pv_b - mean_pv_a):.2f}")
print(f"상대적 증가율: {((mean_pv_b / mean_pv_a - 1) * 100):.1f}%")

print(f"\nMann-Whitney U 검정:")
print(f"  U-통계량: {u_stat:.4f}")
print(f"  p-value: {p_value_u:.4f}")

if p_value_u < 0.05:
    print(f"  결과: 통계적으로 유의미한 차이 있음 (p < 0.05)")
else:
    print(f"  결과: 통계적으로 유의미한 차이 없음 (p >= 0.05)")

# 5. 신뢰구간 계산
print("\n5. 전환율 95% 신뢰구간")
print("-"*80)

# 그룹 A 신뢰구간
se_a = np.sqrt(conversion_rate_a * (1 - conversion_rate_a) / total_a)
ci_a_lower = conversion_rate_a - 1.96 * se_a
ci_a_upper = conversion_rate_a + 1.96 * se_a

# 그룹 B 신뢰구간
se_b = np.sqrt(conversion_rate_b * (1 - conversion_rate_b) / total_b)
ci_b_lower = conversion_rate_b - 1.96 * se_b
ci_b_upper = conversion_rate_b + 1.96 * se_b

print(f"그룹 A: {conversion_rate_a:.2%} [{ci_a_lower:.2%}, {ci_a_upper:.2%}]")
print(f"그룹 B: {conversion_rate_b:.2%} [{ci_b_lower:.2%}, {ci_b_upper:.2%}]")

# 6. 최종 권장사항
print("\n6. 최종 권장사항")
print("-"*80)

metrics_improved = 0
if conversion_rate_b > conversion_rate_a and p_value_chi2 < 0.05:
    print("✓ 전환율: B가 통계적으로 유의미하게 높음")
    metrics_improved += 1
else:
    print("○ 전환율: 유의미한 차이 없음")

if mean_time_b > mean_time_a and p_value_t < 0.05:
    print("✓ 체류시간: B가 통계적으로 유의미하게 높음")
    metrics_improved += 1
else:
    print("○ 체류시간: 유의미한 차이 없음")

if mean_pv_b > mean_pv_a and p_value_u < 0.05:
    print("✓ 페이지뷰: B가 통계적으로 유의미하게 높음")
    metrics_improved += 1
else:
    print("○ 페이지뷰: 유의미한 차이 없음")

print(f"\n개선된 지표 수: {metrics_improved}/3")

if metrics_improved >= 2:
    print("\n🎯 권장사항: B 버전 채택 권장")
elif metrics_improved == 1:
    print("\n⚠️  권장사항: 추가 테스트 필요")
else:
    print("\n❌ 권장사항: A 버전 유지")
```

---

### 문제 44 정답: 고객 세그멘테이션

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)

df = pd.DataFrame({
    '고객ID': range(1, 1001),
    '연령': np.random.randint(20, 70, 1000),
    '연간구매액': np.random.lognormal(mean=10, sigma=1, size=1000),
    '구매빈도': np.random.poisson(lam=10, size=1000),
    '최근구매일': np.random.randint(1, 365, 1000),
    '평균구매금액': np.random.lognormal(mean=11, sigma=0.5, size=1000)
})

print("="*80)
print("고객 세그멘테이션 (K-means Clustering)")
print("="*80)

# 1. 데이터 준비 및 탐색
print("\n1. 데이터 기본 통계")
print("-"*80)
print(df.describe())

# 2. 특성 선택 및 스케일링
print("\n2. 데이터 전처리")
print("-"*80)

features = ['연령', '연간구매액', '구매빈도', '최근구매일', '평균구매금액']
X = df[features].copy()

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"선택된 특성: {features}")
print(f"스케일링 완료: {X_scaled.shape}")

# 3. 최적 클러스터 수 찾기 (Elbow Method)
print("\n3. 최적 클러스터 수 탐색 (Elbow Method)")
print("-"*80)

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

print("K값별 Inertia:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.2f}")

# Elbow 찾기 (간단한 방법)
inertia_diff = np.diff(inertias)
optimal_k = K_range[np.argmin(inertia_diff)] + 1
print(f"\n권장 클러스터 수: {optimal_k}")

# 4. K-means 클러스터링 수행
print("\n4. K-means 클러스터링 수행")
print("-"*80)

n_clusters = 4  # 4개로 설정
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['클러스터'] = kmeans.fit_predict(X_scaled)

print(f"클러스터링 완료: {n_clusters}개 세그먼트")
print(f"\n클러스터별 고객 수:")
print(df['클러스터'].value_counts().sort_index())

# 5. 클러스터별 특성 분석
print("\n5. 클러스터별 특성 분석")
print("-"*80)

cluster_summary = df.groupby('클러스터').agg({
    '고객ID': 'count',
    '연령': 'mean',
    '연간구매액': 'mean',
    '구매빈도': 'mean',
    '최근구매일': 'mean',
    '평균구매금액': 'mean'
}).round(2)

cluster_summary.columns = ['고객수', '평균연령', '평균연간구매액', '평균구매빈도',
                           '평균최근구매일', '평균구매금액']

print(cluster_summary)

# 6. 클러스터 명명 및 해석
print("\n6. 클러스터 프로파일링")
print("-"*80)

for cluster_id in range(n_clusters):
    cluster_data = df[df['클러스터'] == cluster_id]
    
    avg_age = cluster_data['연령'].mean()
    avg_purchase = cluster_data['연간구매액'].mean()
    avg_freq = cluster_data['구매빈도'].mean()
    avg_recency = cluster_data['최근구매일'].mean()
    avg_amount = cluster_data['평균구매금액'].mean()
    
    print(f"\n클러스터 {cluster_id} ({len(cluster_data)}명):")
    
    # 특성 기반 명명
    if avg_purchase > df['연간구매액'].quantile(0.75):
        if avg_freq > df['구매빈도'].quantile(0.75):
            segment_name = "💎 VIP 고객"
        else:
            segment_name = "👑 고가치 고객"
    elif avg_recency < df['최근구매일'].quantile(0.25):
        segment_name = "🔥 활성 고객"
    elif avg_recency > df['최근구매일'].quantile(0.75):
        segment_name = "😴 휴면 고객"
    else:
        segment_name = "📊 일반 고객"
    
    print(f"  세그먼트명: {segment_name}")
    print(f"  평균 연령: {avg_age:.1f}세")
    print(f"  연간 구매액: {avg_purchase:,.0f}원")
    print(f"  구매 빈도: {avg_freq:.1f}회")
    print(f"  최근 구매: {avg_recency:.0f}일 전")
    print(f"  평균 구매금액: {avg_amount:,.0f}원")

# 7. 클러스터 중심점 분석
print("\n7. 클러스터 중심점 (원본 스케일)")
print("-"*80)

# 스케일링 역변환
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(
    cluster_centers_original,
    columns=features,
    index=[f'클러스터 {i}' for i in range(n_clusters)]
).round(2)

print(centers_df)

# 8. 비즈니스 인사이트
print("\n8. 비즈니스 인사이트 및 권장 액션")
print("-"*80)

# 각 클러스터별 권장 액션
for cluster_id in range(n_clusters):
    cluster_data = df[df['클러스터'] == cluster_id]
    
    avg_purchase = cluster_data['연간구매액'].mean()
    avg_freq = cluster_data['구매빈도'].mean()
    avg_recency = cluster_data['최근구매일'].mean()
    
    print(f"\n클러스터 {cluster_id}:")
    
    if avg_purchase > df['연간구매액'].quantile(0.75):
        print("  • VIP 프로그램 제공")
        print("  • 전담 상담사 배정")
        print("  • 프리미엄 혜택 제공")
    
    if avg_recency > df['최근구매일'].quantile(0.75):
        print("  • 재활성화 캠페인 필요")
        print("  • 할인 쿠폰 발송")
        print("  • 개인화된 상품 추천")
    
    if avg_freq < df['구매빈도'].quantile(0.25):
        print("  • 구매 빈도 증대 캠페인")
        print("  • 정기 구매 혜택 안내")
        print("  • 리마인더 발송")

# 9. 요약
print("\n9. 세그멘테이션 요약")
print("-"*80)
print(f"총 고객 수: {len(df):,}명")
print(f"세그먼트 수: {n_clusters}개")
print(f"가장 큰 세그먼트: 클러스터 {df['클러스터'].value_counts().idxmax()} ({df['클러스터'].value_counts().max()}명)")
print(f"가장 작은 세그먼트: 클러스터 {df['클러스터'].value_counts().idxmin()} ({df['클러스터'].value_counts().min()}명)")

# 상위 세그먼트 (연간구매액 기준)
top_cluster = cluster_summary['평균연간구매액'].idxmax()
print(f"최고 가치 세그먼트: 클러스터 {top_cluster} (평균 {cluster_summary.loc[top_cluster, '평균연간구매액']:,.0f}원)")
```

---

### 문제 45 정답: 시계열 이상 탐지

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000, freq='h')

normal_data = 100 + 20 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000) * 5

# 이상치 추가
anomaly_indices = np.random.choice(range(100, 900), 20, replace=False)
normal_data[anomaly_indices] += np.random.choice([-50, 50], 20) + np.random.randn(20) * 10

df = pd.DataFrame({'시간': dates, '값': normal_data})
df.set_index('시간', inplace=True)

print("="*80)
print("시계열 이상 탐지")
print("="*80)

# 1. 기본 통계
print("\n1. 데이터 기본 통계")
print("-"*80)
print(df.describe())

# 2. 방법 1: Z-Score 기반 이상 탐지
print("\n2. Z-Score 기반 이상 탐지")
print("-"*80)

df['평균'] = df['값'].mean()
df['표준편차'] = df['값'].std()
df['Z_Score'] = (df['값'] - df['평균']) / df['표준편차']

# Z-Score 임계값 설정 (±3)
threshold_z = 3
df['이상치_Z'] = (np.abs(df['Z_Score']) > threshold_z).astype(int)

anomalies_z = df[df['이상치_Z'] == 1]
print(f"Z-Score 방법으로 탐지된 이상치: {len(anomalies_z)}개")
print(f"\n이상치 샘플:")
print(anomalies_z[['값', 'Z_Score']].head(10))

# 3. 방법 2: IQR 기반 이상 탐지
print("\n3. IQR 기반 이상 탐지")
print("-"*80)

Q1 = df['값'].quantile(0.25)
Q3 = df['값'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['이상치_IQR'] = ((df['값'] < lower_bound) | (df['값'] > upper_bound)).astype(int)

anomalies_iqr = df[df['이상치_IQR'] == 1]
print(f"IQR: {IQR:.2f}")
print(f"하한: {lower_bound:.2f}, 상한: {upper_bound:.2f}")
print(f"IQR 방법으로 탐지된 이상치: {len(anomalies_iqr)}개")

# 4. 방법 3: 이동 평균 기반 이상 탐지
print("\n4. 이동 평균 기반 이상 탐지")
print("-"*80)

window_size = 24  # 24시간
df['이동평균'] = df['값'].rolling(window=window_size, center=True).mean()
df['이동표준편차'] = df['값'].rolling(window=window_size, center=True).std()

# 이동 평균으로부터의 편차
df['편차'] = df['값'] - df['이동평균']
df['편차_표준화'] = df['편차'] / df['이동표준편차']

# 임계값 설정
threshold_ma = 3
df['이상치_MA'] = (np.abs(df['편차_표준화']) > threshold_ma).astype(int)

anomalies_ma = df[df['이상치_MA'] == 1]
print(f"이동 평균 방법으로 탐지된 이상치: {len(anomalies_ma)}개 (처음/끝 {window_size//2}개 제외)")

# 5. 방법 4: 계절성 분해 기반
print("\n5. 계절성 분해 기반 이상 탐지")
print("-"*80)

# 시간대별 평균 (일일 패턴)
df['시간대'] = df.index.hour
hourly_avg = df.groupby('시간대')['값'].transform('mean')
hourly_std = df.groupby('시간대')['값'].transform('std')

df['기대값'] = hourly_avg
df['잔차'] = df['값'] - df['기대값']
df['잔차_표준화'] = df['잔차'] / hourly_std

threshold_seasonal = 3
df['이상치_계절성'] = (np.abs(df['잔차_표준화']) > threshold_seasonal).astype(int)

anomalies_seasonal = df[df['이상치_계절성'] == 1]
print(f"계절성 분해 방법으로 탐지된 이상치: {len(anomalies_seasonal)}개")

# 6. 통합 이상치 점수
print("\n6. 통합 이상치 분석")
print("-"*80)

# 여러 방법으로 탐지된 횟수
df['이상치_점수'] = (df['이상치_Z'] + df['이상치_IQR'] + 
                       df['이상치_MA'] + df['이상치_계절성'])

print("이상치 점수 분포:")
print(df['이상치_점수'].value_counts().sort_index())

# 2개 이상 방법에서 탐지된 경우를 최종 이상치로 판단
df['최종_이상치'] = (df['이상치_점수'] >= 2).astype(int)
final_anomalies = df[df['최종_이상치'] == 1]

print(f"\n최종 이상치 ({len(final_anomalies)}개):")
print(final_anomalies[['값', '이상치_점수', 'Z_Score', '편차_표준화']].head(20))

# 7. 이상치 통계
print("\n7. 이상치 상세 분석")
print("-"*80)

print(f"전체 데이터 포인트: {len(df)}")
print(f"최종 이상치 개수: {len(final_anomalies)}")
print(f"이상치 비율: {len(final_anomalies)/len(df)*100:.2f}%")

print(f"\n이상치 값 범위:")
print(f"  최소: {final_anomalies['값'].min():.2f}")
print(f"  최대: {final_anomalies['값'].max():.2f}")
print(f"  평균: {final_anomalies['값'].mean():.2f}")

print(f"\n정상 데이터 값 범위:")
normal_data_df = df[df['최종_이상치'] == 0]
print(f"  최소: {normal_data_df['값'].min():.2f}")
print(f"  최대: {normal_data_df['값'].max():.2f}")
print(f"  평균: {normal_data_df['값'].mean():.2f}")

# 8. 시간대별 이상치 분포
print("\n8. 시간대별 이상치 발생 빈도")
print("-"*80)

hourly_anomalies = final_anomalies.groupby(final_anomalies.index.hour).size()
print(hourly_anomalies.head(10))

# 9. 연속된 이상치 탐지
print("\n9. 연속된 이상치 패턴")
print("-"*80)

# 연속된 이상치 찾기
df['이상치_변화'] = df['최종_이상치'].diff()
anomaly_starts = df[df['이상치_변화'] == 1].index
anomaly_ends = df[df['이상치_변화'] == -1].index

print(f"이상치 구간 수: {len(anomaly_starts)}")

if len(anomaly_starts) > 0:
    print(f"\n처음 5개 이상치 구간:")
    for i, start in enumerate(anomaly_starts[:5]):
        if i < len(anomaly_ends):
            duration = (anomaly_ends[i] - start).total_seconds() / 3600
            print(f"  {start} ~ {anomaly_ends[i]} (지속시간: {duration:.1f}시간)")
        else:
            print(f"  {start} ~ (진행 중)")

# 10. 권장 조치
print("\n10. 권장 조치 사항")
print("-"*80)

if len(final_anomalies) > len(df) * 0.05:
    print("⚠️  이상치 비율이 5%를 초과합니다.")
    print("   → 데이터 수집 프로세스 점검 필요")

if len(final_anomalies) > 0:
    # 가장 큰 이상치
    max_anomaly_idx = final_anomalies['값'].abs().idxmax()
    max_anomaly_val = final_anomalies.loc[max_anomaly_idx, '값']
    
    print(f"\n가장 큰 이상치: {max_anomaly_val:.2f} (시간: {max_anomaly_idx})")
    print("   → 해당 시점의 시스템 로그 확인 필요")

print("\n분석 완료 ✓")
```

---
# NumPy와 Pandas 고급 문제 50선 - 정답 (Part 7 최종편 - 문제 46-50)

## 통합 문제 정답 계속 (문제 46-50)

### 문제 46 정답: 복잡한 데이터 집계 리포트

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({
    '날짜': dates,
    '매출': np.random.lognormal(mean=12, sigma=0.5, size=len(dates)) * (1 + np.arange(len(dates)) * 0.0001),
    '지역': np.random.choice(['서울', '부산', '대구'], len(dates)),
    '제품군': np.random.choice(['A', 'B', 'C'], len(dates))
})

print("="*80)
print("복잡한 데이터 집계 리포트")
print("="*80)

# 날짜 파생 변수 추가
df['연도'] = df['날짜'].dt.year
df['월'] = df['날짜'].dt.month
df['분기'] = df['날짜'].dt.quarter
df['년월'] = df['날짜'].dt.to_period('M')

# 1. 월별 매출 리포트
print("\n1. 월별 매출 리포트")
print("-"*80)

monthly_sales = df.groupby(['연도', '월']).agg({
    '매출': ['sum', 'mean', 'count']
}).round(0)

monthly_sales.columns = ['총매출', '평균매출', '거래건수']
print(monthly_sales.tail(24))

# 2. 분기별 매출 리포트
print("\n2. 분기별 매출 리포트")
print("-"*80)

quarterly_sales = df.groupby(['연도', '분기']).agg({
    '매출': ['sum', 'mean']
}).round(0)

quarterly_sales.columns = ['총매출', '평균매출']
print(quarterly_sales)

# 3. 연도별 매출 리포트
print("\n3. 연도별 매출 리포트")
print("-"*80)

yearly_sales = df.groupby('연도').agg({
    '매출': ['sum', 'mean', 'count']
}).round(0)

yearly_sales.columns = ['총매출', '평균매출', '거래건수']
print(yearly_sales)

# 4. 전년 동기 대비 성장률 (YoY)
print("\n4. 전년 동기 대비 성장률 (월별)")
print("-"*80)

# 월별 매출을 피벗
monthly_pivot = df.groupby(['연도', '월'])['매출'].sum().unstack(fill_value=0)

# YoY 성장률 계산
yoy_growth = monthly_pivot.pct_change() * 100

print("월별 YoY 성장률 (%):")
print(yoy_growth.round(2))

# 5. 전월 대비 성장률 (MoM)
print("\n5. 전월 대비 성장률")
print("-"*80)

monthly_series = df.groupby('년월')['매출'].sum()
mom_growth = monthly_series.pct_change() * 100

print("최근 12개월 MoM 성장률 (%):")
print(mom_growth.tail(12).round(2))

# 6. 지역별 성과
print("\n6. 지역별 매출 성과")
print("-"*80)

region_sales = df.groupby(['지역', '연도'])['매출'].sum().unstack()
region_sales['합계'] = region_sales.sum(axis=1)
region_sales.loc['합계'] = region_sales.sum()

print(region_sales.round(0))

# 지역별 성장률
region_growth = region_sales.iloc[:-1, :-1].pct_change(axis=1) * 100
print("\n지역별 전년 대비 성장률 (%):")
print(region_growth.round(2))

# 7. 제품군별 성과
print("\n7. 제품군별 매출 성과")
print("-"*80)

product_sales = df.groupby(['제품군', '연도'])['매출'].sum().unstack()
product_sales['합계'] = product_sales.sum(axis=1)
product_sales.loc['합계'] = product_sales.sum()

print(product_sales.round(0))

# 제품군별 비중
product_share = df.groupby(['연도', '제품군'])['매출'].sum().unstack()
product_share_pct = product_share.div(product_share.sum(axis=1), axis=0) * 100

print("\n제품군별 매출 비중 (%):")
print(product_share_pct.round(2))

# 8. 종합 대시보드 지표
print("\n8. 종합 대시보드 지표")
print("-"*80)

# 최근 12개월 데이터
last_12m = df[df['날짜'] >= df['날짜'].max() - pd.DateOffset(months=12)]
prev_12m = df[(df['날짜'] >= df['날짜'].max() - pd.DateOffset(months=24)) & 
               (df['날짜'] < df['날짜'].max() - pd.DateOffset(months=12))]

current_revenue = last_12m['매출'].sum()
previous_revenue = prev_12m['매출'].sum()
revenue_growth = ((current_revenue / previous_revenue) - 1) * 100

print(f"최근 12개월 총 매출: {current_revenue:,.0f}원")
print(f"이전 12개월 총 매출: {previous_revenue:,.0f}원")
print(f"YoY 성장률: {revenue_growth:.2f}%")

print(f"\n최근 12개월 평균 일매출: {last_12m['매출'].mean():,.0f}원")
print(f"최근 12개월 최고 일매출: {last_12m['매출'].max():,.0f}원")
print(f"최근 12개월 최저 일매출: {last_12m['매출'].min():,.0f}원")

# 9. 분기별 상세 분석
print("\n9. 분기별 상세 분석 (최근 2년)")
print("-"*80)

recent_quarters = df[df['연도'] >= df['연도'].max() - 1].groupby(['연도', '분기']).agg({
    '매출': ['sum', 'mean', 'std', 'count']
}).round(0)

recent_quarters.columns = ['총매출', '평균매출', '표준편차', '거래건수']
print(recent_quarters)

# 분기별 성장률
quarterly_series = df.groupby(['연도', '분기'])['매출'].sum()
quarterly_growth = quarterly_series.pct_change() * 100

print("\n분기별 성장률 (QoQ, %):")
print(quarterly_growth.tail(8).round(2))

# 10. 트렌드 분석
print("\n10. 트렌드 분석")
print("-"*80)

# 이동 평균 (90일)
df_sorted = df.sort_values('날짜')
df_sorted['매출_90일이동평균'] = df_sorted['매출'].rolling(window=90, min_periods=1).mean()

# 최근 30일 vs 이전 30일
last_30d = df[df['날짜'] >= df['날짜'].max() - pd.DateOffset(days=30)]['매출'].mean()
prev_30d = df[(df['날짜'] >= df['날짜'].max() - pd.DateOffset(days=60)) & 
              (df['날짜'] < df['날짜'].max() - pd.DateOffset(days=30))]['매출'].mean()

trend = ((last_30d / prev_30d) - 1) * 100

print(f"최근 30일 평균 일매출: {last_30d:,.0f}원")
print(f"이전 30일 평균 일매출: {prev_30d:,.0f}원")
print(f"단기 트렌드: {trend:+.2f}%")

if trend > 5:
    print("📈 매출이 상승 추세입니다")
elif trend < -5:
    print("📉 매출이 하락 추세입니다")
else:
    print("➡️  매출이 안정적입니다")

# 11. 계절성 분석
print("\n11. 계절성 분석 (월별 패턴)")
print("-"*80)

monthly_pattern = df.groupby('월')['매출'].mean()
monthly_pattern_norm = (monthly_pattern / monthly_pattern.mean() - 1) * 100

print("월별 계절성 지수 (평균 대비 %):")
for month, value in monthly_pattern_norm.items():
    print(f"  {month}월: {value:+.2f}%")

# 12. 최종 요약 리포트
print("\n12. 경영진 요약 리포트")
print("="*80)

total_revenue = df['매출'].sum()
total_transactions = len(df)
avg_transaction = df['매출'].mean()

top_region = df.groupby('지역')['매출'].sum().idxmax()
top_product = df.groupby('제품군')['매출'].sum().idxmax()

best_year = df.groupby('연도')['매출'].sum().idxmax()
best_year_revenue = df.groupby('연도')['매출'].sum().max()

print(f"📊 전체 기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
print(f"💰 총 누적 매출: {total_revenue:,.0f}원")
print(f"📝 총 거래 건수: {total_transactions:,}건")
print(f"💵 평균 거래액: {avg_transaction:,.0f}원")
print(f"\n🏆 최고 실적 지역: {top_region}")
print(f"🏆 최고 실적 제품군: {top_product}")
print(f"🏆 최고 실적 연도: {best_year}년 ({best_year_revenue:,.0f}원)")
print(f"\n📈 최근 YoY 성장률: {revenue_growth:+.2f}%")
print(f"📊 최근 30일 트렌드: {trend:+.2f}%")
```

---

### 문제 47 정답: RFM 분석

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)

transactions = pd.DataFrame({
    '거래ID': range(1, 50001),
    '고객ID': np.random.randint(1, 1001, 50000),
    '거래일': pd.date_range('2023-01-01', periods=50000, freq='10min'),
    '거래금액': np.random.lognormal(mean=10, sigma=1, size=50000)
})

analysis_date = pd.Timestamp('2024-12-31')

print("="*80)
print("RFM 분석 (Recency, Frequency, Monetary)")
print("="*80)

# 1. RFM 지표 계산
print("\n1. RFM 지표 계산")
print("-"*80)

# Recency: 최근 구매일로부터 경과 일수
recency = transactions.groupby('고객ID')['거래일'].max().reset_index()
recency['Recency'] = (analysis_date - recency['거래일']).dt.days

# Frequency: 구매 빈도
frequency = transactions.groupby('고객ID')['거래ID'].count().reset_index()
frequency.columns = ['고객ID', 'Frequency']

# Monetary: 총 구매 금액
monetary = transactions.groupby('고객ID')['거래금액'].sum().reset_index()
monetary.columns = ['고객ID', 'Monetary']

# RFM 데이터프레임 생성
rfm = recency[['고객ID', 'Recency']].merge(frequency, on='고객ID').merge(monetary, on='고객ID')

print(f"RFM 분석 대상 고객 수: {len(rfm):,}명")
print(f"\nRFM 지표 기술통계:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

print(f"\n샘플 데이터:")
print(rfm.head(10))

# 2. RFM 점수 부여
print("\n2. RFM 점수 부여 (1-5점)")
print("-"*80)

# 각 지표를 5개 구간으로 분할하여 점수 부여
# Recency: 낮을수록 좋음 (역순)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Frequency: 높을수록 좋음
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Monetary: 높을수록 좋음
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# RFM 점수를 문자열로 결합
rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                    rfm['F_Score'].astype(str) + 
                    rfm['M_Score'].astype(str))

# 총점 계산
rfm['RFM_Total'] = (rfm['R_Score'].astype(int) + 
                    rfm['F_Score'].astype(int) + 
                    rfm['M_Score'].astype(int))

print("RFM 점수 분포:")
print(rfm[['R_Score', 'F_Score', 'M_Score', 'RFM_Total']].describe())

print(f"\n상위 10명 고객:")
print(rfm.nlargest(10, 'RFM_Total')[['고객ID', 'Recency', 'Frequency', 'Monetary', 
                                      'R_Score', 'F_Score', 'M_Score', 'RFM_Total']])

# 3. 고객 세그먼트 분류
print("\n3. 고객 세그먼트 분류")
print("-"*80)

def assign_segment(row):
    """RFM 점수에 따른 세그먼트 할당"""
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    
    # Champions: 최고 고객
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    
    # Loyal Customers: 충성 고객
    elif r >= 3 and f >= 4:
        return 'Loyal Customers'
    
    # Potential Loyalist: 잠재 충성 고객
    elif r >= 4 and f >= 2 and f <= 3:
        return 'Potential Loyalist'
    
    # New Customers: 신규 고객
    elif r >= 4 and f <= 2:
        return 'New Customers'
    
    # Promising: 유망 고객
    elif r >= 3 and f <= 2:
        return 'Promising'
    
    # Need Attention: 관심 필요
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Need Attention'
    
    # About to Sleep: 이탈 위험
    elif r >= 2 and r <= 3:
        return 'About to Sleep'
    
    # At Risk: 위험 고객
    elif r <= 2 and f >= 4:
        return 'At Risk'
    
    # Cannot Lose Them: 놓쳐서는 안될 고객
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cannot Lose Them'
    
    # Hibernating: 휴면 고객
    elif r <= 2 and f <= 2:
        return 'Hibernating'
    
    # Lost: 이탈 고객
    else:
        return 'Lost'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

# 세그먼트별 고객 수
segment_counts = rfm['Segment'].value_counts()
print("세그먼트별 고객 분포:")
for segment, count in segment_counts.items():
    pct = count / len(rfm) * 100
    print(f"  {segment}: {count:,}명 ({pct:.1f}%)")

# 4. 세그먼트별 RFM 평균
print("\n4. 세그먼트별 RFM 지표 평균")
print("-"*80)

segment_stats = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    '고객ID': 'count'
}).round(2)

segment_stats.columns = ['평균_Recency', '평균_Frequency', '평균_Monetary', '고객수']
segment_stats = segment_stats.sort_values('평균_Monetary', ascending=False)

print(segment_stats)

# 5. 세그먼트별 매출 기여도
print("\n5. 세그먼트별 매출 기여도")
print("-"*80)

segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
segment_revenue_pct = (segment_revenue / segment_revenue.sum() * 100).round(2)

revenue_contribution = pd.DataFrame({
    '총매출': segment_revenue,
    '매출비중(%)': segment_revenue_pct
})

print(revenue_contribution)

# 6. 액션 플랜
print("\n6. 세그먼트별 권장 액션 플랜")
print("-"*80)

action_plans = {
    'Champions': '🏆 VIP 프로그램, 신제품 우선 제공, 리뷰 요청',
    'Loyal Customers': '💎 감사 이벤트, 멤버십 혜택 강화, 크로스셀링',
    'Potential Loyalist': '🌟 로열티 프로그램 가입 유도, 추천 제품 제공',
    'New Customers': '🎁 온보딩 프로그램, 첫 구매 후 팔로업',
    'Promising': '📈 개인화된 추천, 구매 빈도 증대 캠페인',
    'Need Attention': '⚠️  특별 할인, 재구매 유도 캠페인',
    'About to Sleep': '😴 리액티베이션 이메일, 한정 쿠폰',
    'At Risk': '🚨 개인화된 재참여 캠페인, 고객 서비스 컨택',
    'Cannot Lose Them': '💰 VIP 전용 혜택, 개인 상담, 특별 관리',
    'Hibernating': '😪 윈백 캠페인, 대폭 할인, 설문조사',
    'Lost': '❌ 마지막 재참여 시도, ROI 분석 후 포기 여부 결정'
}

for segment in segment_counts.index:
    print(f"\n{segment} ({segment_counts[segment]:,}명):")
    print(f"  {action_plans.get(segment, '일반 마케팅')}")

# 7. 주요 인사이트
print("\n7. 주요 인사이트")
print("="*80)

top_segment = segment_revenue.idxmax()
top_segment_revenue = segment_revenue.max()
top_segment_pct = segment_revenue_pct.max()

print(f"💰 최고 매출 세그먼트: {top_segment}")
print(f"   매출: {top_segment_revenue:,.0f}원 ({top_segment_pct:.1f}%)")

champions_count = segment_counts.get('Champions', 0)
champions_pct = champions_count / len(rfm) * 100
print(f"\n🏆 Champions 비율: {champions_pct:.1f}% ({champions_count:,}명)")

at_risk_count = segment_counts.get('At Risk', 0) + segment_counts.get('Cannot Lose Them', 0)
at_risk_pct = at_risk_count / len(rfm) * 100
print(f"🚨 위험 고객 비율: {at_risk_pct:.1f}% ({at_risk_count:,}명)")

hibernating_count = segment_counts.get('Hibernating', 0) + segment_counts.get('Lost', 0)
hibernating_pct = hibernating_count / len(rfm) * 100
print(f"😴 휴면/이탈 고객 비율: {hibernating_pct:.1f}% ({hibernating_count:,}명)")

# 8. 상위 고객 리스트
print("\n8. Champions 고객 리스트 (상위 20명)")
print("-"*80)

champions = rfm[rfm['Segment'] == 'Champions'].nlargest(20, 'Monetary')
print(champions[['고객ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']])
```

---

### 문제 48 정답: 코호트 분석

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)

customers = pd.DataFrame({
    '고객ID': range(1, 10001),
    '가입일': pd.date_range('2023-01-01', periods=10000, freq='h')
})

activities = []
for customer_id in range(1, 10001):
    signup_date = customers[customers['고객ID'] == customer_id]['가입일'].iloc[0]
    n_activities = np.random.randint(0, 12)
    if n_activities > 0:
        for j in range(n_activities):
            days_after_signup = np.random.randint(1, 365)
            activity_date = signup_date + pd.DateOffset(days=days_after_signup)
            activities.append({'고객ID': customer_id, '활동일': activity_date})

activity_df = pd.DataFrame(activities)

print("="*80)
print("코호트 분석 (Cohort Analysis)")
print("="*80)

# 1. 데이터 준비
print("\n1. 데이터 개요")
print("-"*80)
print(f"총 고객 수: {len(customers):,}명")
print(f"총 활동 기록: {len(activity_df):,}건")
print(f"기간: {customers['가입일'].min().date()} ~ {customers['가입일'].max().date()}")

# 2. 가입 월 코호트 생성
print("\n2. 가입 월 코호트 생성")
print("-"*80)

customers['가입월'] = customers['가입일'].dt.to_period('M')
activity_df['활동월'] = activity_df['활동일'].dt.to_period('M')

# 활동 데이터와 고객 데이터 병합
cohort_data = activity_df.merge(customers[['고객ID', '가입월']], on='고객ID')

# 가입 후 경과 월 수 계산
cohort_data['경과월'] = (cohort_data['활동월'] - cohort_data['가입월']).apply(lambda x: x.n)

# 코호트별 고객 수
cohort_sizes = customers.groupby('가입월')['고객ID'].nunique()
print("코호트별 가입 고객 수:")
print(cohort_sizes.head(12))

# 3. 코호트별 활성 사용자 수 계산
print("\n3. 코호트별 월별 활성 사용자 수")
print("-"*80)

# 각 코호트의 경과월별 활성 고객 수
cohort_active = cohort_data.groupby(['가입월', '경과월'])['고객ID'].nunique().reset_index()
cohort_active.columns = ['가입월', '경과월', '활성고객수']

# 피벗 테이블로 변환
cohort_pivot = cohort_active.pivot(index='가입월', columns='경과월', values='활성고객수')

print("코호트 활성 사용자 테이블 (처음 12개월):")
print(cohort_pivot.iloc[:6, :12].fillna(0).astype(int))

# 4. 리텐션율 계산
print("\n4. 리텐션율 계산 (%)")
print("-"*80)

# 각 코호트의 초기 고객 수로 나누어 리텐션율 계산
cohort_retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

print("코호트 리텐션율 테이블 (처음 12개월):")
print(cohort_retention.iloc[:6, :12].fillna(0).round(1))

# 5. 평균 리텐션율
print("\n5. 월별 평균 리텐션율")
print("-"*80)

avg_retention = cohort_retention.mean(axis=0)
print("경과월별 평균 리텐션율 (%):")
for month, retention in avg_retention.iloc[:12].items():
    print(f"  {month}개월차: {retention:.1f}%")

# 6. 코호트별 리텐션 곡선
print("\n6. 주요 코호트 리텐션 비교")
print("-"*80)

# 초기 3개 코호트 비교
early_cohorts = cohort_retention.iloc[:3, :12]
print(early_cohorts.round(1))

# 7. 리텐션 지표
print("\n7. 주요 리텐션 지표")
print("-"*80)

# Day 1 Retention (1개월 리텐션)
month_1_retention = cohort_retention[0].mean()
print(f"평균 1개월 리텐션율: {month_1_retention:.1f}%")

# Day 30 Retention (3개월 리텐션)
if 2 in cohort_retention.columns:
    month_3_retention = cohort_retention[2].mean()
    print(f"평균 3개월 리텐션율: {month_3_retention:.1f}%")

# Day 180 Retention (6개월 리텐션)
if 5 in cohort_retention.columns:
    month_6_retention = cohort_retention[5].mean()
    print(f"평균 6개월 리텐션율: {month_6_retention:.1f}%")

# Day 365 Retention (12개월 리텐션)
if 11 in cohort_retention.columns:
    month_12_retention = cohort_retention[11].mean()
    print(f"평균 12개월 리텐션율: {month_12_retention:.1f}%")

# 8. 리텐션 개선 트렌드
print("\n8. 리텐션 개선 트렌드 분석")
print("-"*80)

# 초기 vs 최근 코호트 비교
if len(cohort_retention) >= 6:
    early_cohorts_avg = cohort_retention.iloc[:3, :6].mean(axis=0)
    recent_cohorts_avg = cohort_retention.iloc[-3:, :6].mean(axis=0)
    
    print("초기 코호트 vs 최근 코호트 (6개월 리텐션):")
    comparison = pd.DataFrame({
        '초기_코호트': early_cohorts_avg,
        '최근_코호트': recent_cohorts_avg,
        '차이(%)': recent_cohorts_avg - early_cohorts_avg
    }).round(1)
    print(comparison)

# 9. 이탈 분석
print("\n9. 이탈 분석")
print("-"*80)

# 경과월별 이탈률 (리텐션율의 감소)
churn_rate = 100 - cohort_retention.mean(axis=0)
print("경과월별 평균 이탈률 (%):")
for month, churn in churn_rate.iloc[:12].items():
    print(f"  {month}개월차: {churn:.1f}%")

# 10. 액션 아이템
print("\n10. 권장 액션 아이템")
print("="*80)

if month_1_retention < 50:
    print("🚨 1개월 리텐션율이 낮습니다 (< 50%)")
    print("   → 온보딩 프로세스 개선 필요")
    print("   → 첫 주 사용자 경험 최적화")

if len(comparison) > 0 and comparison['차이(%)'].iloc[2] < 0:
    print("\n📉 최근 코호트의 3개월 리텐션이 하락했습니다")
    print("   → 제품 변경사항 검토 필요")
    print("   → 사용자 피드백 분석")

best_cohort = cohort_retention[0].idxmax()
best_retention = cohort_retention.loc[best_cohort, 0]
print(f"\n🏆 최고 리텐션 코호트: {best_cohort} ({best_retention:.1f}%)")
print("   → 해당 기간의 마케팅/제품 전략 분석")

# 11. 코호트 요약
print("\n11. 코호트 분석 요약")
print("-"*80)

total_cohorts = len(cohort_sizes)
avg_cohort_size = cohort_sizes.mean()

print(f"총 코호트 수: {total_cohorts}개")
print(f"평균 코호트 크기: {avg_cohort_size:.0f}명")
print(f"전체 기간 평균 1개월 리텐션: {month_1_retention:.1f}%")

if 11 in cohort_retention.columns:
    print(f"전체 기간 평균 12개월 리텐션: {month_12_retention:.1f}%")
    
# 리텐션 등급
if month_1_retention >= 60:
    grade = "A (우수)"
elif month_1_retention >= 40:
    grade = "B (양호)"
elif month_1_retention >= 20:
    grade = "C (보통)"
else:
    grade = "D (개선 필요)"

print(f"\n리텐션 등급: {grade}")
```

---

### 문제 49 정답: 판매 예측 데이터 준비

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
n = len(dates)

trend = np.linspace(1000, 2000, n)
yearly_seasonality = 300 * np.sin(2 * np.pi * np.arange(n) / 365.25)
weekly_seasonality = 100 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.randn(n) * 50

sales = trend + yearly_seasonality + weekly_seasonality + noise

df = pd.DataFrame({
    '날짜': dates,
    '판매량': sales.clip(0)
})

print("="*80)
print("판매 예측 데이터 준비")
print("="*80)

# 1. 기본 정보
print("\n1. 데이터 기본 정보")
print("-"*80)
print(f"데이터 기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
print(f"총 일수: {len(df):,}일")
print(f"\n판매량 기술통계:")
print(df['판매량'].describe())

df.set_index('날짜', inplace=True)

# 2. 시간 기반 특성
print("\n2. 시간 기반 특성 생성")
print("-"*80)

df['연도'] = df.index.year
df['월'] = df.index.month
df['일'] = df.index.day
df['요일'] = df.index.dayofweek
df['주차'] = df.index.isocalendar().week
df['분기'] = df.index.quarter
df['연초경과일'] = df.index.dayofyear
df['월말여부'] = df.index.is_month_end.astype(int)
df['주말여부'] = (df['요일'] >= 5).astype(int)

# 사인/코사인 인코딩 (주기성 표현)
df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
df['요일_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
df['요일_cos'] = np.cos(2 * np.pi * df['요일'] / 7)
df['연초경과일_sin'] = np.sin(2 * np.pi * df['연초경과일'] / 365.25)
df['연초경과일_cos'] = np.cos(2 * np.pi * df['연초경과일'] / 365.25)

print("시간 특성 생성 완료")
print(f"  기본 시간 특성: 9개")
print(f"  주기성 특성: 6개")

# 3. 라그 특성
print("\n3. 라그 특성 생성")
print("-"*80)

lag_days = [1, 2, 3, 7, 14, 30, 60, 90]
for lag in lag_days:
    df[f'판매량_lag{lag}'] = df['판매량'].shift(lag)

print(f"라그 특성 생성: {len(lag_days)}개")
print(f"  라그 일수: {lag_days}")

# 4. 롤링 통계 특성
print("\n4. 롤링 통계 특성 생성")
print("-"*80)

windows = [7, 14, 30, 60, 90]
for window in windows:
    df[f'판매량_평균_{window}일'] = df['판매량'].rolling(window=window).mean()
    df[f'판매량_표준편차_{window}일'] = df['판매량'].rolling(window=window).std()
    df[f'판매량_최대_{window}일'] = df['판매량'].rolling(window=window).max()
    df[f'판매량_최소_{window}일'] = df['판매량'].rolling(window=window).min()
    df[f'판매량_중앙값_{window}일'] = df['판매량'].rolling(window=window).median()

print(f"롤링 통계 특성 생성: {len(windows) * 5}개")
print(f"  윈도우 크기: {windows}")
print(f"  통계량: 평균, 표준편차, 최대, 최소, 중앙값")

# 5. 변화율 특성
print("\n5. 변화율 특성 생성")
print("-"*80)

df['판매량_차분1'] = df['판매량'].diff(1)
df['판매량_차분7'] = df['판매량'].diff(7)
df['판매량_변화율1'] = df['판매량'].pct_change(1)
df['판매량_변화율7'] = df['판매량'].pct_change(7)

# 7일 평균 대비 변화율
df['판매량_7일평균대비'] = (df['판매량'] - df['판매량_평균_7일']) / df['판매량_평균_7일']

print("변화율 특성 생성: 5개")

# 6. 지수 이동 평균
print("\n6. 지수 이동 평균 생성")
print("-"*80)

spans = [7, 14, 30]
for span in spans:
    df[f'판매량_EMA_{span}'] = df['판매량'].ewm(span=span, adjust=False).mean()

print(f"EMA 특성 생성: {len(spans)}개")

# 7. 상호작용 특성
print("\n7. 상호작용 특성 생성")
print("-"*80)

df['주말_x_월'] = df['주말여부'] * df['월']
df['주말_x_판매량_평균7일'] = df['주말여부'] * df['판매량_평균_7일']
df['월_x_판매량_평균30일'] = df['월'] * df['판매량_평균_30일']

print("상호작용 특성 생성: 3개")

# 8. 타겟 변수 (예측 대상)
print("\n8. 타겟 변수 생성")
print("-"*80)

df['타겟_1일후'] = df['판매량'].shift(-1)
df['타겟_7일후'] = df['판매량'].shift(-7)
df['타겟_향후7일평균'] = df['판매량'].shift(-1).rolling(window=7).mean()

print("타겟 변수 생성: 3개")
print("  - 1일 후 판매량")
print("  - 7일 후 판매량")
print("  - 향후 7일 평균 판매량")

# 9. 결측치 처리
print("\n9. 결측치 처리")
print("-"*80)

print("결측치 현황:")
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
print(f"  결측치가 있는 열: {len(missing_counts)}개")
print(f"  최대 결측치 개수: {missing_counts.max()}개")

# 타겟 변수의 결측치가 있는 행 제거
df_clean = df.dropna(subset=['타겟_1일후'])

print(f"\n타겟 결측치 제거 후:")
print(f"  원본: {len(df)}행")
print(f"  정제: {len(df_clean)}행")
print(f"  제거: {len(df) - len(df_clean)}행")

# 나머지 결측치는 forward fill
df_clean = df_clean.fillna(method='ffill')

print(f"\n최종 결측치: {df_clean.isnull().sum().sum()}개")

# 10. Train/Validation/Test 분할
print("\n10. 데이터 분할 (Train/Validation/Test)")
print("-"*80)

# 시계열 데이터이므로 시간 순서 유지
train_size = int(len(df_clean) * 0.7)
val_size = int(len(df_clean) * 0.15)

train_df = df_clean.iloc[:train_size]
val_df = df_clean.iloc[train_size:train_size+val_size]
test_df = df_clean.iloc[train_size+val_size:]

print(f"Train set:")
print(f"  크기: {len(train_df):,}행")
print(f"  기간: {train_df.index.min().date()} ~ {train_df.index.max().date()}")
print(f"  비율: {len(train_df)/len(df_clean)*100:.1f}%")

print(f"\nValidation set:")
print(f"  크기: {len(val_df):,}행")
print(f"  기간: {val_df.index.min().date()} ~ {val_df.index.max().date()}")
print(f"  비율: {len(val_df)/len(df_clean)*100:.1f}%")

print(f"\nTest set:")
print(f"  크기: {len(test_df):,}행")
print(f"  기간: {test_df.index.min().date()} ~ {test_df.index.max().date()}")
print(f"  비율: {len(test_df)/len(df_clean)*100:.1f}%")

# 11. 특성 요약
print("\n11. 생성된 특성 요약")
print("-"*80)

feature_groups = {
    '원본': ['판매량'],
    '시간 특성': [col for col in df_clean.columns if any(x in col for x in ['연도', '월', '일', '요일', '주차', '분기', 'sin', 'cos'])],
    '라그 특성': [col for col in df_clean.columns if 'lag' in col],
    '롤링 특성': [col for col in df_clean.columns if any(x in col for x in ['평균', '표준편차', '최대', '최소', '중앙값']) and 'EMA' not in col],
    '변화율 특성': [col for col in df_clean.columns if any(x in col for x in ['차분', '변화율', '대비'])],
    'EMA 특성': [col for col in df_clean.columns if 'EMA' in col],
    '상호작용 특성': [col for col in df_clean.columns if '_x_' in col],
    '타겟 변수': [col for col in df_clean.columns if '타겟' in col]
}

total_features = 0
for group, features in feature_groups.items():
    print(f"{group}: {len(features)}개")
    total_features += len(features)

print(f"\n총 특성 개수: {total_features}개")

# 12. 상관관계 분석
print("\n12. 타겟 변수와의 상관관계 (상위 10개)")
print("-"*80)

# 타겟을 제외한 특성들
feature_cols = [col for col in df_clean.columns if '타겟' not in col and col != '판매량']

correlations = df_clean[feature_cols + ['타겟_1일후']].corr()['타겟_1일후'].drop('타겟_1일후')
correlations = correlations.abs().sort_values(ascending=False)

print(correlations.head(10))

# 13. 최종 요약
print("\n13. 데이터 준비 완료")
print("="*80)
print(f"✓ 총 {total_features}개 특성 생성")
print(f"✓ Train/Val/Test 분할 완료")
print(f"✓ 결측치 처리 완료")
print(f"✓ 예측 모델 학습 준비 완료")

print(f"\n다음 단계:")
print("  1. 특성 스케일링 (StandardScaler, MinMaxScaler)")
print("  2. 특성 선택 (상관관계, 중요도 기반)")
print("  3. 모델 학습 (ARIMA, Prophet, LSTM, XGBoost)")
print("  4. 하이퍼파라미터 튜닝")
print("  5. 모델 평가 (MAE, RMSE, MAPE)")
```

---

### 문제 50 정답: 종합 비즈니스 대시보드 데이터

```python
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)

sales = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '매출': np.random.lognormal(mean=14, sigma=0.3, size=365),
    '주문수': np.random.poisson(lam=100, size=365)
})

customers = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '신규고객': np.random.poisson(lam=20, size=365),
    '활성고객': np.random.poisson(lam=500, size=365),
    '이탈고객': np.random.poisson(lam=10, size=365)
})

marketing = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '광고비': np.random.uniform(100000, 500000, 365),
    '방문자수': np.random.poisson(lam=1000, size=365),
    '전환수': np.random.poisson(lam=50, size=365)
})

print("="*80)
print("종합 비즈니스 대시보드 데이터")
print("="*80)

# 1. 데이터 통합
print("\n1. 데이터 통합")
print("-"*80)

dashboard = sales.merge(customers, on='날짜').merge(marketing, on='날짜')

print(f"통합 데이터 shape: {dashboard.shape}")
print(f"기간: {dashboard['날짜'].min().date()} ~ {dashboard['날짜'].max().date()}")
print(f"\n컬럼:")
for col in dashboard.columns:
    print(f"  - {col}")

# 날짜 파생 변수
dashboard['연도'] = dashboard['날짜'].dt.year
dashboard['월'] = dashboard['날짜'].dt.month
dashboard['주'] = dashboard['날짜'].dt.isocalendar().week
dashboard['요일'] = dashboard['날짜'].dt.dayofweek
dashboard['주말여부'] = (dashboard['요일'] >= 5).astype(int)

# 2. 핵심 KPI 계산
print("\n2. 핵심 KPI 계산")
print("-"*80)

# 매출 KPI
dashboard['평균주문금액'] = dashboard['매출'] / dashboard['주문수']
dashboard['일일매출목표'] = 3000000  # 예시
dashboard['매출달성률'] = (dashboard['매출'] / dashboard['일일매출목표']) * 100

# 고객 KPI
dashboard['순고객증가'] = dashboard['신규고객'] - dashboard['이탈고객']
dashboard['고객이탈률'] = (dashboard['이탈고객'] / dashboard['활성고객']) * 100

# 마케팅 KPI
dashboard['전환율'] = (dashboard['전환수'] / dashboard['방문자수']) * 100
dashboard['CPA'] = dashboard['광고비'] / dashboard['전환수']  # Cost Per Acquisition
dashboard['ROAS'] = (dashboard['매출'] / dashboard['광고비']) * 100  # Return On Ad Spend

print("KPI 생성 완료:")
print("  매출 KPI: 평균주문금액, 매출달성률")
print("  고객 KPI: 순고객증가, 고객이탈률")
print("  마케팅 KPI: 전환율, CPA, ROAS")

# 3. 이동 평균 지표
print("\n3. 이동 평균 지표")
print("-"*80)

windows = [7, 30]
for window in windows:
    dashboard[f'매출_MA{window}'] = dashboard['매출'].rolling(window=window).mean()
    dashboard[f'주문수_MA{window}'] = dashboard['주문수'].rolling(window=window).mean()
    dashboard[f'신규고객_MA{window}'] = dashboard['신규고객'].rolling(window=window).mean()
    dashboard[f'전환율_MA{window}'] = dashboard['전환율'].rolling(window=window).mean()

print(f"이동 평균 생성: {len(windows)}개 윈도우 (7일, 30일)")

# 4. 주요 대시보드 지표
print("\n4. 주요 대시보드 지표")
print("="*80)

# 전체 기간 요약
total_revenue = dashboard['매출'].sum()
total_orders = dashboard['주문수'].sum()
total_customers_new = dashboard['신규고객'].sum()
total_ad_spend = dashboard['광고비'].sum()

avg_daily_revenue = dashboard['매출'].mean()
avg_order_value = total_revenue / total_orders
overall_conversion = (dashboard['전환수'].sum() / dashboard['방문자수'].sum()) * 100
overall_roas = (total_revenue / total_ad_spend) * 100

print(f"📊 전체 기간 요약 (365일)")
print("-"*80)
print(f"총 매출: {total_revenue:,.0f}원")
print(f"총 주문수: {total_orders:,}건")
print(f"평균 일매출: {avg_daily_revenue:,.0f}원")
print(f"평균 주문금액: {avg_order_value:,.0f}원")
print(f"\n총 신규고객: {total_customers_new:,}명")
print(f"전체 전환율: {overall_conversion:.2f}%")
print(f"\n총 광고비: {total_ad_spend:,.0f}원")
print(f"전체 ROAS: {overall_roas:.1f}%")
print(f"평균 CPA: {total_ad_spend/dashboard['전환수'].sum():,.0f}원")

# 5. 월별 대시보드
print("\n5. 월별 성과 대시보드")
print("-"*80)

monthly_dashboard = dashboard.groupby('월').agg({
    '매출': 'sum',
    '주문수': 'sum',
    '평균주문금액': 'mean',
    '신규고객': 'sum',
    '이탈고객': 'sum',
    '활성고객': 'mean',
    '광고비': 'sum',
    '방문자수': 'sum',
    '전환수': 'sum',
    '전환율': 'mean',
    'ROAS': 'mean'
}).round(0)

print(monthly_dashboard)

# 6. 주간 트렌드
print("\n6. 주간 트렌드 분석")
print("-"*80)

# 최근 4주
recent_4_weeks = dashboard[dashboard['날짜'] >= dashboard['날짜'].max() - pd.DateOffset(weeks=4)]
weekly_trend = recent_4_weeks.groupby('주').agg({
    '매출': 'sum',
    '주문수': 'sum',
    '신규고객': 'sum',
    '전환율': 'mean'
}).round(0)

print("최근 4주 트렌드:")
print(weekly_trend)

# 7. 요일별 패턴
print("\n7. 요일별 패턴 분석")
print("-"*80)

weekday_pattern = dashboard.groupby('요일').agg({
    '매출': 'mean',
    '주문수': 'mean',
    '방문자수': 'mean',
    '전환율': 'mean'
}).round(0)

weekday_pattern.index = ['월', '화', '수', '목', '금', '토', '일']
print(weekday_pattern)

# 8. 성과 지표 달성률
print("\n8. 목표 달성률")
print("-"*80)

# 월별 목표 설정
monthly_target = 100_000_000  # 월 1억원
monthly_actual = dashboard.groupby('월')['매출'].sum()
monthly_achievement = (monthly_actual / monthly_target * 100).round(1)

print("월별 매출 목표 달성률:")
for month, achievement in monthly_achievement.items():
    status = "✓" if achievement >= 100 else "✗"
    print(f"  {month}월: {achievement}% {status}")

# 9. 고객 생애 가치 추정
print("\n9. 고객 생애 가치 (LTV) 추정")
print("-"*80)

avg_customer_lifespan = 365  # 1년
avg_purchase_frequency = total_orders / total_customers_new
avg_customer_value = avg_order_value * avg_purchase_frequency

print(f"평균 구매 빈도: {avg_purchase_frequency:.2f}회/고객")
print(f"평균 주문금액: {avg_order_value:,.0f}원")
print(f"추정 고객 생애 가치: {avg_customer_value:,.0f}원")

# CAC vs LTV
avg_cac = total_ad_spend / total_customers_new
ltv_cac_ratio = avg_customer_value / avg_cac

print(f"\n고객 획득 비용 (CAC): {avg_cac:,.0f}원")
print(f"LTV / CAC 비율: {ltv_cac_ratio:.2f}")

if ltv_cac_ratio >= 3:
    print("✓ 건강한 비즈니스 모델 (LTV/CAC >= 3)")
else:
    print("⚠️ 마케팅 효율 개선 필요 (LTV/CAC < 3)")

# 10. 알림 및 인사이트
print("\n10. 자동 알림 및 인사이트")
print("="*80)

# 최근 7일 데이터
recent_7d = dashboard.tail(7)
prev_7d = dashboard.iloc[-14:-7]

# 매출 변화
revenue_change = ((recent_7d['매출'].mean() / prev_7d['매출'].mean()) - 1) * 100

if revenue_change < -10:
    print("🚨 경고: 최근 7일 매출이 10% 이상 감소했습니다")
    print(f"   변화율: {revenue_change:.1f}%")
elif revenue_change > 10:
    print("🎉 축하: 최근 7일 매출이 10% 이상 증가했습니다")
    print(f"   변화율: {revenue_change:.1f}%")

# 전환율 체크
recent_conversion = recent_7d['전환율'].mean()
if recent_conversion < 3:
    print("\n⚠️ 주의: 전환율이 낮습니다 (< 3%)")
    print(f"   현재 전환율: {recent_conversion:.2f}%")

# ROAS 체크
recent_roas = recent_7d['ROAS'].mean()
if recent_roas < 100:
    print("\n⚠️ 주의: ROAS가 100% 미만입니다 (광고비 > 매출)")
    print(f"   현재 ROAS: {recent_roas:.1f}%")

# 최고/최저 성과일
best_day = dashboard.loc[dashboard['매출'].idxmax()]
worst_day = dashboard.loc[dashboard['매출'].idxmin()]

print(f"\n📈 최고 매출일: {best_day['날짜'].date()}")
print(f"   매출: {best_day['매출']:,.0f}원")

print(f"\n📉 최저 매출일: {worst_day['날짜'].date()}")
print(f"   매출: {worst_day['매출']:,.0f}원")

# 11. 대시보드 출력 준비
print("\n11. 대시보드 데이터 준비 완료")
print("-"*80)
print(f"✓ 총 {len(dashboard)}일 데이터")
print(f"✓ {len(dashboard.columns)}개 지표")
print(f"✓ 통합 대시보드 생성 완료")

print("\n생성된 주요 데이터프레임:")
print("  1. dashboard: 일별 상세 데이터")
print("  2. monthly_dashboard: 월별 집계")
print("  3. weekly_trend: 주간 트렌드")
print("  4. weekday_pattern: 요일별 패턴")

print("\n대시보드 시각화 권장사항:")
print("  📊 매출 추이 라인 차트 (일별, 월별)")
print("  📈 전환율 추이")
print("  💰 ROAS 및 CPA 추이")
print("  👥 고객 증감 추이")
print("  📅 요일별/월별 히트맵")
```

---

## 🎉 50개 문제 완성!

모든 문제의 정답을 완성했습니다!

**학습 완료 체크리스트:**
- ✅ NumPy 고급 (문제 1-20)
- ✅ Pandas 고급 (문제 21-40)
- ✅ 통합 실전 (문제 41-50)

**다음 단계:**
1. 각 문제를 직접 실행해보기
2. 코드 수정해보며 실험하기
3. 실무 데이터에 적용해보기
4. 포트폴리오 프로젝트 만들기

축하합니다! 🎊

(계속해서 작성하겠습니다)
**학습 완료! 🎉**

이제 NumPy와 Pandas의 고급 기능을 마스터했습니다!

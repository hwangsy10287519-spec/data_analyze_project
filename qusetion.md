# NumPy와 Pandas 고급 문제 50선 (문제편)

## Part 1: NumPy 고급 (문제 1-20)

### 문제 1: 복잡한 배열 인덱싱
**문제:** 100x100 크기의 랜덤 정수 배열(0~255)을 생성하고, 체스판 패턴(짝수 행의 짝수 열, 홀수 행의 홀수 열)에 해당하는 요소들의 평균을 구하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randint(0, 256, size=(100, 100))
```

---

### 문제 2: 이동 평균 (Moving Average)
**문제:** 길이 1000의 1차원 랜덤 배열을 생성하고, 윈도우 크기가 50인 이동 평균을 계산하세요. (for 루프 없이 벡터화 연산으로)

**데이터 생성:**
```python
import numpy as np
arr = np.random.randn(1000)
window_size = 50
```

---

### 문제 3: 행렬 분해와 재구성
**문제:** 5x5 랜덤 행렬을 생성하고 SVD(특이값 분해)를 수행한 후, 상위 3개의 특이값만 사용하여 행렬을 근사 재구성하고 원본과의 Frobenius norm 오차를 계산하세요.

**데이터 생성:**
```python
import numpy as np
A = np.random.randn(5, 5)
```

---

### 문제 4: 복잡한 브로드캐스팅
**문제:** shape이 (100, 1)인 배열과 (1, 100)인 배열을 생성하여 외적(outer product)을 계산하고, 그 결과에서 대각선 방향으로 5칸 떨어진 모든 대각선들의 합을 구하세요.

**데이터 생성:**
```python
import numpy as np
a = np.arange(1, 101).reshape(100, 1)
b = np.arange(1, 101).reshape(1, 100)
```

---

### 문제 5: 커스텀 정렬
**문제:** 10x10 랜덤 정수 배열(1~100)을 생성하고, 각 행을 그 행의 중앙값을 기준으로 정렬하되, 중앙값보다 작은 값들은 내림차순, 큰 값들은 오름차순으로 정렬하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randint(1, 101, size=(10, 10))
```

---

### 문제 6: 다차원 히스토그램
**문제:** 두 개의 1000개 요소를 가진 정규분포 배열을 생성하고, 20x20 빈을 사용한 2D 히스토그램을 계산한 후, 가장 빈도가 높은 빈의 좌표와 빈도수를 찾으세요.

**데이터 생성:**
```python
import numpy as np
x = np.random.randn(1000)
y = np.random.randn(1000)
```

---

### 문제 7: 행렬의 고유값 문제
**문제:** 대칭 행렬 A(7x7)를 랜덤하게 생성하고, 모든 고유값이 양수가 되도록 만든 후, 가장 큰 고유값에 해당하는 고유벡터를 구하고 정규화하세요.

**데이터 생성:**
```python
import numpy as np
A = np.random.randn(7, 7)
A = (A + A.T) / 2  # 대칭 행렬로 만들기
```

---

### 문제 8: 복잡한 마스킹
**문제:** 50x50 배열을 생성하고, 중심점 (25, 25)로부터 유클리드 거리가 10 이상 20 이하인 원형 고리(annulus) 영역의 값들만 추출하여 평균을 구하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randn(50, 50)
```

---

### 문제 9: 텐서 연산
**문제:** 3차원 배열(10x20x30)을 생성하고, 각 2D 슬라이스(10x20)에 대해 행렬식(determinant)이 계산 가능한 정사각 부분행렬(10x10)을 추출한 후, 30개의 행렬식 값 중 상위 5개의 인덱스를 찾으세요.

**데이터 생성:**
```python
import numpy as np
tensor = np.random.randn(10, 20, 30)
```

---

### 문제 10: 조건부 배열 생성
**문제:** 1부터 10000까지의 숫자 중에서, 각 자릿수의 제곱의 합이 소수(prime number)인 숫자들만 추출하여 배열로 만드세요.

**데이터 생성:**
```python
import numpy as np
numbers = np.arange(1, 10001)
```

---

### 문제 11: 복잡한 reshape와 전치
**문제:** (2, 3, 4, 5) shape의 4차원 배열을 생성하고, 이를 (3, 5, 2, 4)로 재배열한 후, 축 순서를 (2, 0, 3, 1)로 변경하여 최종 shape를 출력하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.arange(2*3*4*5).reshape(2, 3, 4, 5)
```

---

### 문제 12: 고급 인덱싱과 팬시 인덱싱
**문제:** 100x100 배열에서 대각선 요소들을 기준으로 상삼각 행렬과 하삼각 행렬의 요소들을 각각 별도의 1차원 배열로 추출하고, 두 배열의 평균 차이를 계산하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randn(100, 100)
```

---

### 문제 13: 벡터화된 거리 계산
**문제:** 1000개의 2D 포인트(x, y)를 랜덤하게 생성하고, 모든 포인트 쌍 사이의 유클리드 거리를 계산한 1000x1000 거리 행렬을 만든 후, 각 포인트에서 가장 가까운 5개 이웃의 평균 거리를 구하세요. (for 루프 없이)

**데이터 생성:**
```python
import numpy as np
points = np.random.randn(1000, 2)
```

---

### 문제 14: 복잡한 집계 연산
**문제:** (100, 50) shape의 배열을 생성하고, 각 열에 대해 "연속된 양수 구간의 최대 길이"를 계산하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randn(100, 50)
```

---

### 문제 15: 다항식 피팅
**문제:** x 값이 0부터 10까지 100개의 등간격 점에서, y = 2x³ - 3x² + 5x - 1 + noise에 대해 5차 다항식으로 피팅하고, 원래 계수와 피팅된 계수의 차이를 분석하세요.

**데이터 생성:**
```python
import numpy as np
x = np.linspace(0, 10, 100)
y_true = 2*x**3 - 3*x**2 + 5*x - 1
noise = np.random.randn(100) * 10
y_noisy = y_true + noise
```

---

### 문제 16: 메모리 효율적인 연산
**문제:** 대용량 배열을 시뮬레이션하고, 메모리를 절약하면서 평균과 표준편차를 계산하는 방법을 구현하세요. (청크 단위 처리)

**데이터 생성:**
```python
import numpy as np
total_size = 1_000_000
chunk_size = 100_000
# 청크별로 데이터 생성
```

---

### 문제 17: 복잡한 조건부 연산
**문제:** (100, 100) 배열에서 각 3x3 윈도우의 중심값이 그 윈도우 내 중앙값보다 크면 유지하고, 작으면 0으로 만드는 필터를 구현하세요.

**데이터 생성:**
```python
import numpy as np
arr = np.random.randn(100, 100)
```

---

### 문제 18: 고급 선형대수
**문제:** (100, 80) 행렬 A와 (80, 60) 행렬 B의 곱 AB를 계산하고, AB의 특이값 분해를 수행한 후, 조건수(condition number)를 계산하세요.

**데이터 생성:**
```python
import numpy as np
A = np.random.randn(100, 80)
B = np.random.randn(80, 60)
```

---

### 문제 19: 복잡한 배열 병합
**문제:** 3개의 서로 다른 shape의 배열 (100, 3), (100, 5), (100, 2)를 열 방향으로 연결하고, 결과 배열에서 각 행의 분산이 상위 20%에 해당하는 행들만 추출하세요.

**데이터 생성:**
```python
import numpy as np
arr1 = np.random.randn(100, 3)
arr2 = np.random.randn(100, 5)
arr3 = np.random.randn(100, 2)
```

---

### 문제 20: 고급 통계 연산
**문제:** 정규분포를 따르는 1000개 샘플을 생성하고, 부트스트랩(bootstrap) 방법을 사용하여 평균의 95% 신뢰구간을 계산하세요. (1000회 리샘플링)

**데이터 생성:**
```python
import numpy as np
np.random.seed(42)
data = np.random.randn(1000)
```

---

## Part 2: Pandas 고급 (문제 21-40)

### 문제 21: 복잡한 GroupBy 연산
**문제:** 100,000행의 DataFrame을 생성하고 (제품 10종, 지역 5곳, 날짜 2000일), 각 제품-지역 조합에 대해 30일 이동 평균 매출을 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)
n_rows = 100_000

df = pd.DataFrame({
    '날짜': pd.date_range('2020-01-01', periods=n_rows//500, freq='D').repeat(500),
    '제품': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n_rows),
    '지역': np.random.choice(['서울', '부산', '대구', '인천', '광주'], n_rows),
    '매출': np.random.randint(1000, 100000, n_rows)
})
```

---

### 문제 22: 복잡한 Merge 연산
**문제:** 3개의 DataFrame(고객, 주문, 제품)을 만들고, 고객당 총 주문금액과 가장 많이 구매한 제품을 포함하는 최종 DataFrame을 생성하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np

customers = pd.DataFrame({
    '고객ID': range(1, 101),
    '이름': [f'고객{i}' for i in range(1, 101)],
    '지역': np.random.choice(['서울', '부산', '대구'], 100)
})

orders = pd.DataFrame({
    '주문ID': range(1, 501),
    '고객ID': np.random.randint(1, 101, 500),
    '제품ID': np.random.randint(1, 21, 500),
    '수량': np.random.randint(1, 10, 500)
})

products = pd.DataFrame({
    '제품ID': range(1, 21),
    '제품명': [f'제품{i}' for i in range(1, 21)],
    '가격': np.random.randint(10000, 100000, 20)
})
```

---

### 문제 23: 시계열 리샘플링과 집계
**문제:** 시간별 센서 데이터(1년치, 매 시간)를 생성하고, 일별, 주별, 월별로 리샘플링하여 평균, 최대, 최소값을 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np

date_range = pd.date_range('2024-01-01', periods=8760, freq='h')
df = pd.DataFrame({
    '시간': date_range,
    '온도': 20 + 10 * np.sin(np.arange(8760) * 2 * np.pi / 24) + np.random.randn(8760) * 2,
    '습도': 50 + 20 * np.sin(np.arange(8760) * 2 * np.pi / 24 + np.pi/2) + np.random.randn(8760) * 5,
    '압력': 1013 + np.random.randn(8760) * 5
})
df.set_index('시간', inplace=True)
```

---

### 문제 24: 피봇과 멜트
**문제:** 학생별, 과목별 점수 데이터를 wide format으로 생성한 후, long format으로 변환하고, 다시 학생별 평균을 추가한 wide format으로 재구성하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

students = [f'학생{i}' for i in range(1, 51)]
df = pd.DataFrame({
    '학생': students,
    '수학': np.random.randint(60, 100, 50),
    '영어': np.random.randint(60, 100, 50),
    '과학': np.random.randint(60, 100, 50),
    '국어': np.random.randint(60, 100, 50),
    '사회': np.random.randint(60, 100, 50)
})
```

---

### 문제 25: 복잡한 문자열 처리
**문제:** 이메일 주소를 포함한 DataFrame을 생성하고, 도메인별 사용자 수, 가장 긴 이메일 주소, 특정 패턴(숫자 포함)의 이메일 비율을 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
import random
import string

def generate_email():
    username_length = random.randint(5, 15)
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
    domain = random.choice(['gmail.com', 'naver.com', 'daum.net', 'yahoo.com', 'outlook.com'])
    return f"{username}@{domain}"

df = pd.DataFrame({
    '사용자ID': range(1, 1001),
    '이메일': [generate_email() for _ in range(1000)]
})
```

---

### 문제 26: 결측치 고급 처리
**문제:** 다양한 패턴의 결측치를 포함한 DataFrame에서, 열별로 최적의 결측치 처리 방법(평균, 중앙값, 최빈값, 선형 보간)을 자동으로 선택하여 적용하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
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
```

---

### 문제 27: 복잡한 날짜/시간 연산
**문제:** 거래 데이터에서 각 고객의 첫 구매일, 마지막 구매일, 평균 구매 간격, 구매 요일 분포를 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)
n_transactions = 5000

df = pd.DataFrame({
    '거래ID': range(1, n_transactions + 1),
    '고객ID': np.random.randint(1, 201, n_transactions),
    '거래일시': pd.date_range('2023-01-01', periods=n_transactions, freq='3h'),
    '금액': np.random.randint(10000, 500000, n_transactions)
})
df['거래일시'] = pd.to_datetime(np.random.choice(df['거래일시'], size=n_transactions, replace=False))
```

---

### 문제 28: 윈도우 함수와 순위
**문제:** 각 카테고리 내에서 누적 합계, 이동 평균, 그리고 rank를 동시에 계산하는 복잡한 윈도우 연산을 수행하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365),
    '카테고리': np.random.choice(['전자', '의류', '식품', '도서'], 365),
    '매출': np.random.randint(100000, 1000000, 365)
})
```

---

### 문제 29: 다중 인덱스 (MultiIndex)
**문제:** 제품, 지역, 월별 판매 데이터를 MultiIndex로 구성하고, 다양한 레벨에서 집계 및 슬라이싱을 수행하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
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
```

---

### 문제 30: 복잡한 조건부 집계
**문제:** 여러 조건을 조합하여 복잡한 집계를 수행하고, 조건에 따라 다른 집계 함수를 적용하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
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
```

---

### 문제 31: 복잡한 데이터 검증
**문제:** DataFrame의 데이터 품질을 체크하는 종합적인 검증 함수를 작성하세요. (중복, 결측, 이상치, 데이터 타입 등)

**데이터 생성:** (답안 참조)

---

### 문제 32: 롤링 윈도우와 이동 통계
**문제:** 주식 가격 데이터를 시뮬레이션하고, 볼린저 밴드, RSI, MACD 같은 기술적 지표를 계산하세요.

**데이터 생성:** (답안 참조)

---

### 문제 33: 데이터 샘플링과 부트스트랩
**문제:** 계층적 샘플링(stratified sampling)을 구현하고, 부트스트랩을 통해 통계량의 신뢰구간을 계산하세요.

**데이터 생성:** (답안 참조)

---

### 문제 34: 복잡한 조인과 병합
**문제:** 4개의 서로 관련된 테이블을 다양한 조인 방법으로 병합하고, 조인 결과의 차이를 분석하세요.

**데이터 생성:** (답안 참조)

---

### 문제 35: 시계열 포맷 변환과 주기성 분석
**문제:** 시간, 일, 주, 월별 다양한 주기의 데이터를 생성하고, 각 주기별 패턴을 분석하세요.

**데이터 생성:** (답안 참조)

---

### 문제 36: 복잡한 카테고리 인코딩
**문제:** 여러 범주형 변수를 다양한 방법(원-핫, 레이블, 타겟, 빈도 인코딩)으로 인코딩하고 비교하세요.

**데이터 생성:** (답안 참조)

---

### 문제 37: 복잡한 피벗과 크로스탭
**문제:** 다차원 데이터를 여러 방식으로 피벗하고, 교차표를 생성하여 관계를 분석하세요.

**데이터 생성:** (답안 참조)

---

### 문제 38: 메모리 최적화
**문제:** 대용량 DataFrame의 메모리 사용량을 분석하고, 다양한 최적화 기법을 적용하여 메모리를 절감하세요.

**데이터 생성:** (답안 참조)

---

### 문제 39: 복잡한 문자열 파싱
**문제:** 다양한 형식의 문자열 데이터(JSON, 리스트, 중첩 구조)를 파싱하고 정규화하세요.

**데이터 생성:** (답안 참조)

---

### 문제 40: 시계열 예측 준비
**문제:** 시계열 데이터에서 라그 특성, 롤링 특성, 시간 기반 특성을 생성하여 예측 모델을 위한 데이터셋을 구축하세요.

**데이터 생성:** (답안 참조)

---

## Part 3: 통합 문제 (문제 41-50)

### 문제 41: 포괄적인 EDA (탐색적 데이터 분석)
**문제:** 복잡한 데이터셋에 대해 완전한 EDA를 수행하고, 인사이트를 도출하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
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
```

---

### 문제 42: 데이터 파이프라인 구축
**문제:** 원시 데이터를 로드하고, 전처리, 변환, 집계를 거쳐 최종 분석 결과를 도출하는 완전한 파이프라인을 구축하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# 원시 데이터 (여러 소스 시뮬레이션)
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
```

---

### 문제 43: A/B 테스트 분석
**문제:** 두 그룹(A/B)의 실험 데이터를 생성하고, 통계적 유의성을 검정하여 어느 그룹이 더 나은지 분석하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# A 그룹: 기존 버전
group_a = pd.DataFrame({
    '사용자ID': range(1, 1001),
    '그룹': 'A',
    '전환': np.random.choice([0, 1], 1000, p=[0.85, 0.15]),
    '체류시간_초': np.random.normal(120, 30, 1000).clip(10, 300),
    '페이지뷰': np.random.poisson(5, 1000)
})

# B 그룹: 새 버전 (약간 더 나은 성능)
group_b = pd.DataFrame({
    '사용자ID': range(1001, 2001),
    '그룹': 'B',
    '전환': np.random.choice([0, 1], 1000, p=[0.80, 0.20]),
    '체류시간_초': np.random.normal(135, 30, 1000).clip(10, 300),
    '페이지뷰': np.random.poisson(6, 1000)
})

ab_test_data = pd.concat([group_a, group_b], ignore_index=True)
```

---

### 문제 44: 고객 세그멘테이션
**문제:** K-means 클러스터링을 사용하여 고객을 세그먼트화하고, 각 세그먼트의 특성을 분석하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

df = pd.DataFrame({
    '고객ID': range(1, 1001),
    '연령': np.random.randint(20, 70, 1000),
    '연간구매액': np.random.lognormal(mean=10, sigma=1, size=1000),
    '구매빈도': np.random.poisson(lam=10, size=1000),
    '최근구매일': np.random.randint(1, 365, 1000),
    '평균구매금액': np.random.lognormal(mean=11, sigma=0.5, size=1000)
})
```

---

### 문제 45: 시계열 이상 탐지
**문제:** 시계열 데이터에서 통계적 방법을 사용하여 이상치를 탐지하고 시각화하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

dates = pd.date_range('2024-01-01', periods=1000, freq='h')
# 정상 패턴 + 이상치
normal_data = 100 + 20 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000) * 5

# 이상치 추가
anomaly_indices = np.random.choice(range(100, 900), 20, replace=False)
normal_data[anomaly_indices] += np.random.choice([-50, 50], 20) + np.random.randn(20) * 10

df = pd.DataFrame({
    '시간': dates,
    '값': normal_data
})
df.set_index('시간', inplace=True)
```

---

### 문제 46: 복잡한 데이터 집계 리포트
**문제:** 월별, 분기별, 연도별 매출 리포트를 생성하고, 전년 동기 대비 성장률을 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({
    '날짜': dates,
    '매출': np.random.lognormal(mean=12, sigma=0.5, size=len(dates)) * (1 + np.arange(len(dates)) * 0.0001),
    '지역': np.random.choice(['서울', '부산', '대구'], len(dates)),
    '제품군': np.random.choice(['A', 'B', 'C'], len(dates))
})
```

---

### 문제 47: RFM 분석
**문제:** 고객의 Recency, Frequency, Monetary 값을 계산하고, RFM 스코어를 기반으로 고객을 등급화하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# 거래 데이터
transactions = pd.DataFrame({
    '거래ID': range(1, 50001),
    '고객ID': np.random.randint(1, 1001, 50000),
    '거래일': pd.date_range('2023-01-01', periods=50000, freq='10min'),
    '거래금액': np.random.lognormal(mean=10, sigma=1, size=50000)
})

# 분석 기준일
analysis_date = pd.Timestamp('2024-12-31')
```

---

### 문제 48: 코호트 분석
**문제:** 가입 월별 코호트를 생성하고, 각 코호트의 월별 리텐션율을 계산하여 분석하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# 고객 가입 정보
customers = pd.DataFrame({
    '고객ID': range(1, 10001),
    '가입일': pd.date_range('2023-01-01', periods=10000, freq='h')
})

# 활동 로그
activities = []
for customer_id in range(1, 10001):
    signup_date = customers[customers['고객ID'] == customer_id]['가입일'].iloc[0]
    # 각 고객이 랜덤하게 활동
    n_activities = np.random.randint(0, 12)
    if n_activities > 0:
        activity_dates = pd.date_range(
            start=signup_date,
            periods=n_activities,
            freq=f'{np.random.randint(1, 60)}D'
        )
        for date in activity_dates:
            activities.append({'고객ID': customer_id, '활동일': date})

activity_df = pd.DataFrame(activities)
```

---

### 문제 49: 판매 예측 데이터 준비
**문제:** 시계열 판매 데이터에 다양한 특성을 추가하고, train/validation/test 세트로 분할하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# 3년치 일별 판매 데이터
dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
n = len(dates)

# 트렌드 + 계절성 + 노이즈
trend = np.linspace(1000, 2000, n)
yearly_seasonality = 300 * np.sin(2 * np.pi * np.arange(n) / 365.25)
weekly_seasonality = 100 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.randn(n) * 50

sales = trend + yearly_seasonality + weekly_seasonality + noise

df = pd.DataFrame({
    '날짜': dates,
    '판매량': sales.clip(0)
})
```

---

### 문제 50: 종합 비즈니스 대시보드 데이터
**문제:** 여러 소스의 데이터를 통합하여 비즈니스 대시보드에 필요한 모든 지표를 계산하세요.

**데이터 생성:**
```python
import pandas as pd
import numpy as np
np.random.seed(42)

# 매출 데이터
sales = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '매출': np.random.lognormal(mean=14, sigma=0.3, size=365),
    '주문수': np.random.poisson(lam=100, size=365)
})

# 고객 데이터
customers = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '신규고객': np.random.poisson(lam=20, size=365),
    '활성고객': np.random.poisson(lam=500, size=365),
    '이탈고객': np.random.poisson(lam=10, size=365)
})

# 마케팅 데이터
marketing = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=365, freq='D'),
    '광고비': np.random.uniform(100000, 500000, 365),
    '방문자수': np.random.poisson(lam=1000, size=365),
    '전환수': np.random.poisson(lam=50, size=365)
})
```

**학습 팁:**
- 각 문제의 데이터 생성 코드를 먼저 실행하세요
- 문제를 단계별로 나누어 해결하세요
- 여러 방법으로 시도해보세요
- 결과를 검증하는 습관을 들이세요

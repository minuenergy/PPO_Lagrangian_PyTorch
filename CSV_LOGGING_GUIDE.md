# CSV Logging 가이드

PPO 학습 메트릭을 **CSV 형식**으로 저장하도록 변경되었습니다.

---

## 🔄 변경 사항

### 이전 (Tab-Separated Values)
```
Epoch	EpRet	EpCost	KL	Entropy
0	234.5	23.0	0.009	1.234
1	345.6	22.0	0.010	1.123
```
- 파일명: `progress.txt`
- 형식: 탭으로 구분 (`.tsv`)
- 문제: 엑셀/판다스에서 보기 어려움

### 현재 (CSV)
```csv
Epoch,EpRet,EpCost,KL,Entropy
0,234.5,23.0,0.009,1.234
1,345.6,22.0,0.010,1.123
```
- 파일명: `progress.csv`
- 형식: 쉼표로 구분 (`.csv`)
- 장점: 엑셀, 판다스, 구글 시트에서 바로 열림

---

## 📂 파일 위치

학습 실행 시 자동으로 생성됩니다:

```
data/
└── ppo/
    └── ppo_s0/
        ├── progress.csv       ← 메트릭 데이터 (CSV)
        ├── config.json        ← 하이퍼파라미터
        └── pyt_save/          ← 모델 체크포인트
```

---

## 📊 CSV 파일 사용법

### 1. **엑셀에서 열기**
```
1. progress.csv 파일 더블클릭
2. 자동으로 엑셀에서 열림
3. 필터/정렬/차트 작성 가능
```

### 2. **Python pandas로 분석**
```python
import pandas as pd
import matplotlib.pyplot as plt

# CSV 읽기
df = pd.read_csv('data/ppo/ppo_s0/progress.csv')

# 데이터 확인
print(df.head())
print(df.describe())

# 특정 메트릭 조회
print(df[['Epoch', 'EpRet', 'EpCost']])

# 학습 곡선 그리기
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['Epoch'], df['EpRet'])
plt.xlabel('Epoch')
plt.ylabel('Episode Return')
plt.title('Training Progress')

plt.subplot(1, 3, 2)
plt.plot(df['Epoch'], df['EpCost'])
plt.axhline(y=25, color='r', linestyle='--', label='Cost Limit')
plt.xlabel('Epoch')
plt.ylabel('Episode Cost')
plt.legend()
plt.title('Safety Constraint')

plt.subplot(1, 3, 3)
plt.plot(df['Epoch'], df['KL'])
plt.axhline(y=0.01, color='r', linestyle='--', label='Target KL')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()
plt.title('Policy Update Quality')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

# 통계 분석
print("\n=== Training Statistics ===")
print(f"Final EpRet: {df['EpRet'].iloc[-1]:.2f}")
print(f"Max EpRet: {df['EpRet'].max():.2f}")
print(f"Final EpCost: {df['EpCost'].iloc[-1]:.2f}")
print(f"Avg EpCost (last 50): {df['EpCost'].tail(50).mean():.2f}")
print(f"Constraint violations: {(df['EpCost'] > 25).sum()} / {len(df)}")
```

### 3. **여러 실험 비교**
```python
import pandas as pd
import matplotlib.pyplot as plt

# 여러 실험 로드
experiments = {
    'PPO': 'data/ppo/ppo_s0/progress.csv',
    'PPO-Lagrangian': 'data/ppo_lagrangian/ppo_s0/progress.csv',
    'RC-PPO': 'data/rcppo/rcppo_s0/progress.csv'
}

plt.figure(figsize=(10, 5))

for name, path in experiments.items():
    df = pd.read_csv(path)
    plt.plot(df['Epoch'], df['EpRet'], label=name)

plt.xlabel('Epoch')
plt.ylabel('Episode Return')
plt.title('Algorithm Comparison')
plt.legend()
plt.grid(True)
plt.savefig('algorithm_comparison.png')
plt.show()
```

### 4. **구글 시트에서 열기**
```
1. Google Drive 접속
2. 파일 업로드
3. "연결 앱" → "Google Sheets"로 열기
4. 자동으로 스프레드시트로 변환
```

---

## 🔧 기술적 변경 사항

### `utils/logx.py` 수정 내용

1. **CSV 모듈 추가**
   ```python
   import csv  # Line 16
   ```

2. **기본 파일명 변경**
   ```python
   def __init__(self, output_dir=None, output_fname='progress.csv', exp_name=None):
   ```

3. **CSV 형식으로 저장**
   ```python
   def dump_tabular(self):
       if self.output_file is not None:
           csv_writer = csv.writer(self.output_file)
           if self.first_row:
               csv_writer.writerow(self.log_headers)  # 헤더
           csv_writer.writerow(vals)  # 데이터
           self.output_file.flush()
   ```

### 장점

1. **표준 CSV 형식**: RFC 4180 준수
2. **값에 쉼표 포함 가능**: 자동으로 따옴표 처리
3. **호환성**: 엑셀, 구글 시트, 판다스, R 등에서 바로 사용
4. **간편한 분석**: 추가 전처리 없이 바로 분석 가능

---

## 📈 메트릭 설명

CSV 파일에 포함되는 컬럼들:

| 컬럼 | 설명 | 단위 |
|------|------|------|
| `Epoch` | 에폭 번호 | - |
| `AverageEpRet` | 평균 에피소드 보상 | - |
| `StdEpRet` | 보상 표준편차 | - |
| `MaxEpRet` | 최대 보상 | - |
| `MinEpRet` | 최소 보상 | - |
| `AverageEpCost` | 평균 에피소드 비용 | - |
| `StdEpCost` | 비용 표준편차 | - |
| `MaxEpCost` | 최대 비용 | - |
| `MinEpCost` | 최소 비용 | - |
| `EpLen` | 평균 에피소드 길이 | 스텝 |
| `AverageVVals` | 평균 가치 추정 | - |
| `StdVVals` | 가치 표준편차 | - |
| `MaxVVals` | 최대 가치 | - |
| `MinVVals` | 최소 가치 | - |
| `TotalEnvInteracts` | 총 환경 상호작용 | 스텝 |
| `LossPi` | 정책 손실 | - |
| `LossV` | 가치 손실 | - |
| `DeltaLossPi` | 정책 손실 변화 | - |
| `DeltaLossV` | 가치 손실 변화 | - |
| `Entropy` | 정책 엔트로피 | nats |
| `KL` | KL 발산 | - |
| `ClipFrac` | 클리핑 비율 | 0~1 |
| `StopIter` | 조기 종료 횟수 | - |
| `Time` | 경과 시간 | 초 |

---

## 🎯 빠른 분석 스크립트

학습 결과를 빠르게 확인하는 스크립트:

```python
#!/usr/bin/env python
"""Quick analysis of PPO training results"""
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python analyze.py <path_to_progress.csv>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

print("=" * 60)
print(f"Training Results: {csv_path}")
print("=" * 60)

print("\n📊 Overall Statistics:")
print(f"  Total Epochs: {len(df)}")
print(f"  Total Steps: {df['TotalEnvInteracts'].iloc[-1]:,.0f}")

print("\n🎯 Performance:")
print(f"  Initial EpRet: {df['AverageEpRet'].iloc[0]:.2f}")
print(f"  Final EpRet: {df['AverageEpRet'].iloc[-1]:.2f}")
print(f"  Max EpRet: {df['MaxEpRet'].max():.2f}")
print(f"  Improvement: {df['AverageEpRet'].iloc[-1] - df['AverageEpRet'].iloc[0]:.2f}")

print("\n🛡️ Safety:")
print(f"  Final EpCost: {df['AverageEpCost'].iloc[-1]:.2f}")
print(f"  Avg Cost (last 50): {df['AverageEpCost'].tail(50).mean():.2f}")
print(f"  Cost Limit: 25.0")
violations = (df['AverageEpCost'] > 25).sum()
print(f"  Violations: {violations} / {len(df)} ({100*violations/len(df):.1f}%)")

print("\n🔧 Learning Quality:")
print(f"  Final KL: {df['KL'].iloc[-1]:.4f}")
print(f"  Final Entropy: {df['Entropy'].iloc[-1]:.3f}")
print(f"  Avg StopIter: {df['StopIter'].mean():.1f}")

print("\n⏱️ Training Time:")
print(f"  Total Time: {df['Time'].iloc[-1]:.1f} seconds ({df['Time'].iloc[-1]/3600:.2f} hours)")
print(f"  Time per Epoch: {df['Time'].iloc[-1] / len(df):.1f} seconds")

print("=" * 60)
```

저장: `analyze_training.py`

사용법:
```bash
python analyze_training.py data/ppo/ppo_s0/progress.csv
```

---

## ✅ 호환성

- ✅ Microsoft Excel 2010+
- ✅ Google Sheets
- ✅ Python pandas
- ✅ R read.csv()
- ✅ MATLAB readtable()
- ✅ Julia CSV.jl
- ✅ 모든 텍스트 에디터

---

## 📝 참고 사항

- **기존 코드와 호환**: 코드 변경 없이 사용 가능
- **터미널 출력 유지**: 화면 출력은 이전과 동일
- **자동 생성**: 학습 시작 시 자동으로 CSV 파일 생성
- **실시간 업데이트**: 매 에폭마다 CSV 파일에 추가

---

## 🔄 이전 데이터 변환

기존 `progress.txt` 파일을 CSV로 변환:

```bash
# Tab을 쉼표로 변경
sed 's/\t/,/g' progress.txt > progress.csv
```

또는 Python으로:

```python
import pandas as pd

# TSV 읽기
df = pd.read_csv('progress.txt', sep='\t')

# CSV로 저장
df.to_csv('progress.csv', index=False)
```

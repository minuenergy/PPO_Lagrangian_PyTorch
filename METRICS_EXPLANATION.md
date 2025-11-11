# PPO 학습 지표 상세 설명

PPO 학습 중 출력되는 각 지표의 의미를 설명합니다.

---

## 📊 에피소드 성능 지표

### 1. **Epoch**
```
현재 학습 에폭 번호
```
- **의미**: 몇 번째 업데이트 사이클인지
- **범위**: 0 ~ 332 (총 333 에폭)
- **코드**: `ppo.py:444`

---

### 2. **EpRet (Episode Return)**
```
AverageEpRet: 평균 에피소드 누적 보상
StdEpRet: 표준편차
MaxEpRet: 최대값
MinEpRet: 최소값
```

**의미**:
- 에피소드가 끝날 때까지 받은 **총 보상의 합**
- 정책이 얼마나 잘 수행하는지 측정하는 핵심 지표

**수식**:
```
EpRet = Σ r_t (에피소드 시작부터 끝까지)
```

**코드**:
```python
# ppo.py:404
ep_ret += r

# ppo.py:432 (에피소드 종료 시)
logger.store(EpRet=ep_ret, ...)
```

**해석**:
- **높을수록 좋음** ✅
- 증가 추세 → 정책이 개선되고 있음
- 예: `EpRet = 500` → 한 에피소드에서 총 500의 보상 획득

---

### 3. **EpCost (Episode Cost)**
```
AverageEpCost: 평균 에피소드 누적 비용
StdEpCost: 표준편차
MaxEpCost: 최대값
MinEpCost: 최소값
```

**의미**:
- 에피소드 동안 발생한 **총 비용** (안전 위반 횟수)
- PPO-Lagrangian에서 **제약 조건 만족 여부** 측정

**수식**:
```
EpCost = Σ c_t (에피소드 동안의 비용 합)
c_t = info['cost']  # 충돌/위험 진입 시 1.0
```

**코드**:
```python
# ppo.py:403
c = info['cost']
ep_cret += c

# ppo.py:432
logger.store(EpCost=ep_cret)
```

**제약 조건**:
```python
# ppo.py:260
cost_limit = 25  # 목표: EpCost ≤ 25
```

**해석**:
- **낮을수록 좋음** ⬇️
- `EpCost = 25` → 제약 조건 만족
- `EpCost > 25` → 너무 위험한 정책 (페널티 증가)
- `EpCost < 25` → 안전 여유 있음 (페널티 감소)

---

### 4. **EpLen (Episode Length)**
```
Average: 평균 에피소드 길이
```

**의미**:
- 에피소드가 **몇 스텝 동안 지속되었는지**
- 목표 도달 속도 또는 생존 시간

**코드**:
```python
# ppo.py:406
ep_len += 1

# ppo.py:432
logger.store(EpLen=ep_len)
```

**해석**:
- **상황에 따라 다름**:
  - 목표 도달 태스크: 짧을수록 좋음 (빠른 도달)
  - 생존 태스크: 길수록 좋음 (오래 생존)
- Safety Gym: 보통 1000 스텝이 max (`max_ep_len=1000`)
- 예: `EpLen = 850` → 평균 850 스텝 후 에피소드 종료

---

## 🧠 가치 함수 지표

### 5. **VVals (Value Estimates)**
```
AverageVVals: 평균 가치 추정값
StdVVals: 표준편차
MaxVVals: 최대값
MinVVals: 최소값
```

**의미**:
- Critic 네트워크가 예측한 **상태 가치 V(s)**
- "이 상태에서 미래에 받을 보상의 기댓값"

**수식**:
```
V(s) = E[Σ γ^t * r_t | s_0 = s]
```

**코드**:
```python
# ppo.py:400
a, v, vc, logp = ac.step(...)

# ppo.py:410
logger.store(VVals=v)
```

**해석**:
- **EpRet과 비슷한 범위**여야 함 (잘 학습된 경우)
- `VVals ≈ EpRet` → 가치 함수가 정확하게 학습됨 ✅
- `VVals << EpRet` → 가치 함수가 과소평가 중
- `VVals >> EpRet` → 가치 함수가 과대평가 중

---

## 📈 학습 진행 지표

### 6. **TotalEnvInteracts**
```
총 환경 상호작용 횟수
```

**의미**:
- 학습 시작부터 **지금까지 실행한 총 스텝 수**

**수식**:
```
TotalEnvInteracts = (epoch + 1) × steps_per_epoch
                  = (epoch + 1) × 30,000
```

**코드**:
```python
# ppo.py:449
logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
```

**해석**:
- 샘플 효율성 측정에 사용
- 예: `Epoch 10 → 10 × 30,000 = 300,000 스텝`
- 논문에서는 보통 **1천만 스텝** 학습 (`num_steps = 1e7`)

---

## 🎯 정책 손실 지표

### 7. **LossPi (Policy Loss)**
```
Average: 평균 정책 손실
```

**의미**:
- PPO 정책 업데이트의 **목적 함수 값**
- Lagrangian 결합: 보상 최대화 - 페널티 × 비용

**수식**:
```python
# ppo.py:276-278
p = softplus(penalty_param)
pi_objective = loss_rpi - p * loss_cpi
pi_objective = pi_objective / (1 + p)
loss_pi = -pi_objective
```

**코드**:
```python
# ppo.py:386
logger.store(LossPi=pi_l_old, ...)
```

**해석**:
- **음수 값**이 나옴 (목적 함수를 최소화하기 때문)
- 절댓값이 클수록 → 정책이 큰 변화를 시도 중
- 학습이 진행되면서 **안정화**되어야 함

---

### 8. **LossV (Value Loss)**
```
Average: 평균 가치 함수 손실
```

**의미**:
- 가치 함수 V(s)의 **MSE 손실**
- 예측값과 실제 return의 차이

**수식**:
```python
# ppo.py:300
loss_v = ((ac.v(obs) - ret)**2).mean()
```

**코드**:
```python
# ppo.py:386
logger.store(LossV=v_l_old, ...)
```

**해석**:
- **낮을수록 좋음** ⬇️
- 학습이 진행되면서 **감소**해야 함
- 예: `LossV = 0.5` → 평균 제곱 오차 0.5

---

### 9. **DeltaLossPi / DeltaLossV**
```
업데이트 전후 손실 변화량
```

**의미**:
- **한 번의 업데이트**로 손실이 얼마나 변했는지

**수식**:
```
DeltaLossPi = LossPi(after) - LossPi(before)
DeltaLossV = LossV(after) - LossV(before)
```

**코드**:
```python
# ppo.py:388-389
logger.store(
    DeltaLossPi=(loss_pi.item() - pi_l_old),
    DeltaLossV=(loss_v.item() - v_l_old)
)
```

**해석**:
- **음수**: 손실 감소 (개선) ✅
- **양수**: 손실 증가 (악화) ⚠️
- 이상적으로는 **음수**여야 함

---

## 🔍 정책 업데이트 품질 지표

### 10. **Entropy (정책 엔트로피)**
```
Average: 평균 엔트로피
```

**의미**:
- 정책의 **불확실성/무작위성** 정도
- 탐험(exploration) 수준 측정

**수식**:
```
H(π) = -Σ π(a|s) log π(a|s)
```
또는 Gaussian의 경우:
```
H = 0.5 × log(2πe × σ²)
```

**코드**:
```python
# ppo.py:290
ent = pi.entropy().mean().item()

# ppo.py:385
logger.store(Entropy=ent, ...)
```

**해석**:
- **높음**: 정책이 무작위적 → 탐험 많이 함 🔍
- **낮음**: 정책이 결정론적 → 확신 있는 행동 🎯
- 학습 초기: 높음 (탐험)
- 학습 후기: 감소 (수렴)
- 예:
  - `Entropy = 1.5` → 높은 탐험
  - `Entropy = 0.3` → 거의 결정론적

---

### 11. **KL (KL Divergence)**
```
Average: 평균 KL 발산
```

**의미**:
- **이전 정책**과 **현재 정책**의 차이
- 정책이 얼마나 크게 변했는지 측정

**수식**:
```
KL(π_old || π_new) ≈ E[log π_old(a|s) - log π_new(a|s)]
```

**코드**:
```python
# ppo.py:289
approx_kl = (logp_old - logp).mean().item()

# ppo.py:357 (조기 종료)
if kl > 1.2 * target_kl:  # target_kl = 0.01
    break
```

**해석**:
- **낮음**: 안정적인 업데이트 ✅
- **높음**: 급격한 정책 변화 ⚠️
- `target_kl = 0.01` → 목표: KL ≈ 0.01
- `KL > 0.012` → 조기 종료 (너무 큰 변화 방지)
- PPO의 핵심: **Trust Region** 유지

---

### 12. **ClipFrac (Clipping Fraction)**
```
Average: 클리핑된 비율
```

**의미**:
- PPO의 **클리핑이 적용된 샘플의 비율**
- 정책 변화가 제한된 정도

**수식**:
```python
# ppo.py:265-266
ratio = exp(logp - logp_old)
clip_adv = clamp(ratio, 1-0.2, 1+0.2) * adv
clipped = (ratio > 1.2) or (ratio < 0.8)
```

**코드**:
```python
# ppo.py:291-292
clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
```

**해석**:
- **0.0**: 클리핑 없음 (작은 업데이트) ✅
- **1.0**: 모두 클리핑됨 (큰 업데이트) ⚠️
- 이상적: **0.1 ~ 0.3** (일부 클리핑)
- 예:
  - `ClipFrac = 0.23` → 23%의 샘플이 클리핑됨
  - 너무 높으면 → learning rate 낮춰야 함

---

### 13. **StopIter (조기 종료 반복 횟수)**
```
Average: 실제 업데이트 횟수
```

**의미**:
- 정책 업데이트를 **몇 번 반복했는지**
- KL divergence로 인한 조기 종료 여부

**설정**:
```python
train_pi_iters = 80  # 최대 80회
```

**코드**:
```python
# ppo.py:352-359
for i in range(train_pi_iters):  # 최대 80회
    ...
    kl = mpi_avg(pi_info['kl'])
    if kl > 1.2 * target_kl:
        logger.log('Early stopping at step %d due to reaching max kl.'%i)
        break

# ppo.py:365
logger.store(StopIter=i)
```

**해석**:
- **80**: 조기 종료 없음 (KL이 작음) ✅
- **< 80**: 조기 종료됨 (KL이 커짐) ⚠️
- 예:
  - `StopIter = 15` → 15번만 업데이트하고 중단
  - `StopIter = 80` → 전부 실행

---

### 14. **Time**
```
학습 시작부터 경과 시간 (초)
```

**코드**:
```python
# ppo.py:394
start_time = time.time()

# ppo.py:458
logger.log_tabular('Time', time.time()-start_time)
```

**해석**:
- Epoch 0: 45초
- Epoch 10: 450초 (7.5분)
- Epoch 100: 4500초 (75분)

---

## 📋 지표 요약표

| 지표 | 의미 | 좋은 방향 | 정상 범위 |
|------|------|-----------|-----------|
| **EpRet** | 에피소드 보상 | ⬆️ 높을수록 | 환경마다 다름 |
| **EpCost** | 에피소드 비용 | ⬇️ 낮을수록 | ≤ 25 |
| **EpLen** | 에피소드 길이 | 상황에 따라 | ~ 1000 |
| **VVals** | 가치 추정 | ≈ EpRet | EpRet 근처 |
| **LossPi** | 정책 손실 | 안정화 | 음수, 안정 |
| **LossV** | 가치 손실 | ⬇️ 감소 | 0에 수렴 |
| **DeltaLossPi** | 정책 손실 변화 | ⬇️ 음수 | < 0 |
| **DeltaLossV** | 가치 손실 변화 | ⬇️ 음수 | < 0 |
| **Entropy** | 엔트로피 | 점진적 감소 | 1.5 → 0.5 |
| **KL** | KL 발산 | ⬇️ 낮을수록 | ≈ 0.01 |
| **ClipFrac** | 클리핑 비율 | 적당히 | 0.1 ~ 0.3 |
| **StopIter** | 업데이트 횟수 | 80 유지 | 15 ~ 80 |

---

## 🎯 학습 상태 진단

### ✅ 정상 학습
```
Epoch: 50
EpRet: 450 ⬆️ (증가 중)
EpCost: 22 ⬇️ (제약 만족)
KL: 0.009 (안정)
ClipFrac: 0.18 (적당)
StopIter: 80 (조기 종료 없음)
Entropy: 0.8 (점진적 감소)
```

### ⚠️ 문제 징후

**1. 정책이 너무 빠르게 변함**
```
KL: 0.025 (너무 큼!)
ClipFrac: 0.65 (너무 많이 클리핑)
StopIter: 8 (너무 빨리 종료)

→ 해결: pi_lr 낮추기 (3e-4 → 1e-4)
```

**2. 학습이 안됨**
```
EpRet: 100 (변화 없음)
DeltaLossPi: 0.0 (업데이트 없음)
KL: 0.0001 (너무 작음)

→ 해결: pi_lr 높이기, clip_ratio 높이기
```

**3. 가치 함수가 발산**
```
VVals: 1000 (EpRet=200인데)
LossV: 500 (너무 큼)

→ 해결: vf_lr 낮추기 (1e-3 → 5e-4)
```

**4. 비용 제약 위반**
```
EpCost: 45 (> 25, 제약 위반)

→ 결과: penalty_param 자동 증가
```

---

## 💡 실전 팁

### 학습 초기 (Epoch 0-50)
- `EpRet`: 낮음 → 정상
- `Entropy`: 높음 (1.0~1.5) → 탐험 중
- `KL`: 다소 높을 수 있음 (0.01~0.02)

### 학습 중기 (Epoch 50-150)
- `EpRet`: 증가 추세 ⬆️
- `EpCost`: 감소 또는 25 근처 안정화
- `Entropy`: 감소 (0.5~1.0)

### 학습 후기 (Epoch 150-333)
- `EpRet`: 높은 수준 유지
- `EpCost`: ≤ 25 안정적
- `Entropy`: 낮음 (0.3~0.6)
- `KL`: 매우 안정 (< 0.01)

### 하이퍼파라미터 조정 가이드
```python
# KL이 너무 크면 (> 0.02)
pi_lr = 1e-4  # 기본: 3e-4

# 학습이 너무 느리면
clip_ratio = 0.3  # 기본: 0.2
target_kl = 0.015  # 기본: 0.01

# 가치 함수가 불안정하면
vf_lr = 5e-4  # 기본: 1e-3
train_v_iters = 120  # 기본: 80
```

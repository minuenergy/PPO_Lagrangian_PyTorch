# RC-PPO Phase 1 vs Phase 2 상세 설명

## 🔄 전체 실행 흐름

```
시작
 │
 ├─ Phase 1: z-conditioned policy 학습 (rcppo 함수)
 │   └─ 결과: 여러 z값에 대해 일반화된 정책 π(a|x,y,z)
 │
 └─ Phase 2: Optimal z* 찾기 (옵션: --phase2 플래그)
     ├─ 2-1: Fine-tune value function (finetune_value_function)
     └─ 2-2: Bisection으로 z* 계산 (find_optimal_z)
```

---

## 📍 Phase 1: z-conditioned Policy 학습

### 위치
`rcppo()` 함수 (`rcppo.py:78-258`)

### 실행 순서

```python
# 1. 초기화 (lines 85-107)
ac = RCPPOActorCritic(...)  # Augmented state (x,y,z)를 받는 네트워크
buf = RCPPOBuffer(...)
C_const = 10.0

# 2. 메인 학습 루프 (lines 193-241)
for epoch in range(333):
    for t in range(30000):
        # === 핵심: 매 스텝마다 z를 랜덤 샘플링 ===
        z0 = np.random.uniform(z_min, z_max)  # Line 195
        # z0 ∈ [-100, 1000] 범위에서 랜덤

        # Augmented state 구성
        y = -1.0 if avoid_fn(o) <= 0 else 1.0  # Line 196
        z = z0
        obs_aug = np.concatenate([o, [y], [z]])  # Line 199

        # z-conditioned policy로 행동 선택
        a, v, logp = ac.step(obs_aug)  # Line 200
        # 입력: (x, y, z) → 출력: 행동 a

        # 환경 실행
        next_o, r, d, info = env.step(a)
        cost = cost_fn(o, a)

        # Augmented dynamics 업데이트
        y_next = update_y(y, next_o, avoid_fn)  # Line 208
        z_next = z - cost  # Line 209

        # g_hat 계산
        g_hat = compute_g_hat(o, y, z, goal_fn, avoid_fn, C_const)
        # g_hat = max{g(x), C*y, -z}

        buf.store(o, a, y, z, g_hat, v, logp, r)

    # === Policy와 Value Function 업데이트 ===
    update()  # Line 228
    # - Policy: π(a|x,y,z) 학습
    # - Value: Ṽ(x,y,z) 학습
```

### Phase 1이 학습하는 것

1. **다양한 z값에 대한 정책**
   - z = 50일 때 어떻게 행동?
   - z = 500일 때 어떻게 행동?
   - z가 클수록 → 더 많은 비용 사용 가능 → 빠르게 목표 도달
   - z가 작을수록 → 비용 절약 필요 → 천천히, 효율적으로 이동

2. **z-conditioned Value Function**
   ```
   Ṽ(x, y, z) = 이 상태에서 z 비용으로 목표에 도달 가능한가?

   Ṽ(x, y, z) ≤ 0 → 도달 가능!
   Ṽ(x, y, z) > 0  → 도달 불가능
   ```

### Phase 1 손실 함수

```python
# Policy Loss (lines 118-136)
def compute_loss_pi(data):
    obs_aug = torch.cat([obs, y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
    pi, logp = ac.pi(obs_aug, act)
    ratio = torch.exp(logp - logp_old)

    # PPO clipping
    clip_adv = torch.clamp(ratio, 1-0.2, 1+0.2) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    return loss_pi

# Value Loss (lines 138-141)
def compute_loss_v(data):
    obs_aug = torch.cat([obs, y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
    v_pred = ac.v_reach(obs_aug)

    # Min-based Bellman target
    # target = (1-γ)g_hat + γ*min{g_hat, Ṽ_next}
    return (v_pred - ret)**2
```

---

## 📍 Phase 2: Optimal z* 찾기

### 위치
1. `finetune_value_function()` (`rcppo.py:261-318`)
2. `find_optimal_z()` (`rcppo.py:321-345`)

### 실행 조건
```bash
python rcppo.py --phase2  # 이 플래그가 있어야 실행됨!
```

### Phase 2-1: Value Function Fine-tuning

**목적**: Stochastic policy → Deterministic policy 전환 후 value function 재학습

```python
def finetune_value_function(ac, env_fn, ...):
    """Line 261-318"""

    for rollout in range(100):  # 100개 rollout
        z0 = np.random.uniform(-100, 1000)
        o = env.reset()
        y = -1.0 if avoid_fn(o) <= 0 else 1.0
        z = z0

        traj_obs, traj_y, traj_z, traj_g_hat, traj_vals = [], [], [], [], []

        for t in range(max_ep_len):
            obs_aug = np.concatenate([o, [y], [z]])

            # === Deterministic policy 사용 (차이점!) ===
            a = ac.act_deterministic(obs_aug)  # Line 277
            # Stochastic: a ~ π(·|x,y,z)
            # Deterministic: a = mean(π(·|x,y,z))

            v = ac.v_reach(obs_aug)
            g_hat = compute_g_hat(o, y, z, goal_fn, avoid_fn, C_const)

            # 궤적 저장
            traj_obs.append(o)
            traj_y.append(y)
            traj_z.append(z)
            traj_g_hat.append(g_hat)
            traj_vals.append(v)

            next_o, _, d, _ = env.step(a)
            cost = cost_fn(o, a)

            y = update_y(y, next_o, avoid_fn)
            z = z - cost
            o = next_o

            if d or goal_fn(o) <= 0:
                break

        # === Value function만 업데이트 ===
        for _ in range(50):  # Line 300
            for i in range(len(traj_obs)):
                v_pred = ac.v_reach(obs_aug)

                # Min-based Bellman target
                if i < len(traj_obs) - 1:
                    target = (1-0.99)*g_hat[i] + 0.99*min(g_hat[i], vals[i+1])
                else:
                    target = g_hat[i]

                loss += (v_pred - target)**2

            loss.backward()
            vf_optimizer.step()

    return ac
```

**왜 Fine-tune이 필요한가?**
- Phase 1: Stochastic policy로 학습 → 탐험 위해 필요
- Phase 2: Deterministic policy로 실행 → 실제 배포 시 사용
- Deterministic policy에 맞춰 value function 재조정 필요!

### Phase 2-2: Optimal z* 찾기 (Bisection)

**목적**: 주어진 초기 상태 x0에서 목표 도달을 보장하는 최소 비용 z* 계산

```python
def find_optimal_z(ac, x0, goal_fn, avoid_fn, z_min=-100, z_max=1000, ...):
    """Line 321-345"""

    y0 = -1.0 if avoid_fn(x0) <= 0 else 1.0

    def eval_value(z):
        obs_aug = np.concatenate([x0, [y0], [z]])
        v = ac.v_reach(obs_aug)  # Ṽ(x0, y0, z)
        return v

    # === Bisection 알고리즘 ===
    z_low, z_high = z_min, z_max

    for iteration in range(50):
        z_mid = (z_low + z_high) / 2
        v_mid = eval_value(z_mid)

        # 목표: Ṽ(x0, y0, z*) ≤ 0을 만족하는 최소 z* 찾기

        if v_mid > 0:
            # z가 너무 작음 → 목표 도달 불가
            z_low = z_mid  # z를 늘림
        else:
            # z가 충분함 → z를 줄여봄
            z_high = z_mid

        if abs(v_mid) < 0.1 or (z_high - z_low) < 0.1:
            return z_mid  # 수렴

    return (z_low + z_high) / 2
```

**Bisection 예시**:
```
초기 상태 x0 = [1.0, 2.0, ...]

[Iteration 1]
z_mid = 450, Ṽ(x0, y0, 450) = 0.5 > 0 → z 부족, z_low = 450

[Iteration 2]
z_mid = 725, Ṽ(x0, y0, 725) = -0.3 < 0 → z 충분, z_high = 725

[Iteration 3]
z_mid = 587.5, Ṽ(x0, y0, 587.5) = 0.1 > 0 → z_low = 587.5

...

[수렴]
z* = 623.4 → 최소 비용!
```

### Phase 2 실행 (main에서)

```python
# Line 243-258
if run_phase2 and proc_id() == 0:
    logger.log('===== Phase 2 시작 =====')

    # 2-1: Fine-tune
    ac = finetune_value_function(ac, env_fn, goal_fn, avoid_fn, cost_fn, C_const)

    # 2-2: 테스트
    test_env = env_fn()
    for i in range(5):
        x0 = test_env.reset()
        z_star = find_optimal_z(ac, x0, goal_fn, avoid_fn, z_min, z_max)
        print(f'State {i}: Optimal z* = {z_star:.2f}')
```

---

## 📊 Phase 1 vs Phase 2 비교

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| **목적** | z-conditioned policy 학습 | 최적 z* 찾기 |
| **Policy** | Stochastic π(a\|x,y,z) | Deterministic mean(π) |
| **Value Function** | 학습 | Fine-tune |
| **z 샘플링** | 매 스텝 랜덤 | 고정 후 탐색 |
| **출력** | 일반화된 정책 | 특정 상태의 z* |
| **실행 시간** | 333 epochs × 30000 steps | 100 rollouts |
| **위치** | `rcppo()` 함수 | `finetune_value_function()` + `find_optimal_z()` |

---

## 🚀 사용 예시

### Phase 1만 실행
```bash
python rcppo.py --env Safexp-PointGoal1-v0 --epochs 333
```
→ z-conditioned policy 학습만

### Phase 1 + Phase 2 실행
```bash
python rcppo.py --env Safexp-PointGoal1-v0 --epochs 333 --phase2
```
→ 학습 후 optimal z* 계산까지

### 학습된 모델로 실행
```python
import torch
import numpy as np
import core

# 모델 로드
ac = core.RCPPOActorCritic(...)
ac.load_state_dict(torch.load('model.pth'))

# Phase 2: z* 찾기
x0 = env.reset()
z_star = find_optimal_z(ac, x0, goal_fn, avoid_fn)

# 실행
y = -1.0
z = z_star
for t in range(1000):
    obs_aug = np.concatenate([x0, [y], [z]])
    a = ac.act_deterministic(torch.as_tensor(obs_aug, dtype=torch.float32))

    next_o, _, d, _ = env.step(a)
    cost = cost_fn(x0, a)

    y = update_y(y, next_o, avoid_fn)
    z = z - cost
    x0 = next_o

    if d:
        break

print(f'Total cost: {z_star - z:.2f}')
```

---

## 🔑 핵심 정리

1. **Phase 1 = 학습 단계**
   - 여러 z값에 대해 정책 학습
   - "z가 100일 때는?", "z가 500일 때는?" 모두 학습

2. **Phase 2 = 최적화 단계**
   - Deterministic policy로 전환
   - 특정 초기 상태에 대한 최소 비용 z* 계산
   - Bisection으로 효율적 탐색

3. **왜 2단계로 나누나?**
   - Phase 1에서 z*를 직접 학습하기 어려움 (탐험 부족)
   - 다양한 z로 학습 → 이후 최적 z 선택이 효율적

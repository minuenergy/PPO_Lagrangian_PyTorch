# RCPPOActorCritic 'pi' AttributeError 해결 가이드

## 에러 메시지
```
AttributeError: 'RCPPOActorCritic' object has no attribute 'pi'
```

## 가능한 원인 및 해결책

### 1. 모델 로드 문제

**문제**: 이전 버전의 모델을 로드하려고 시도
```python
# 잘못된 방법
ac = torch.load('pyt_save/model.pt')
```

**해결**:
```python
import core
import gym

# 환경 생성
env = gym.make('Safexp-PointGoal1-v0')

# 새로 초기화
ac = core.RCPPOActorCritic(
    env.observation_space,
    env.action_space,
    hidden_sizes=[256, 256]
)

# state_dict만 로드
ac.load_state_dict(torch.load('pyt_save/model.pt'))
```

---

### 2. 환경 타입 문제

**문제**: action_space가 Box도 Discrete도 아님

**확인**:
```python
import gym
env = gym.make('Safexp-PointGoal1-v0')
print(f"Action space type: {type(env.action_space)}")
print(f"Action space: {env.action_space}")

from gym.spaces import Box, Discrete
print(f"Is Box? {isinstance(env.action_space, Box)}")
print(f"Is Discrete? {isinstance(env.action_space, Discrete)}")
```

**해결**: Safety Gym은 Box를 사용해야 합니다. 다른 환경을 사용 중이라면 확인 필요.

---

### 3. Import 캐시 문제

**문제**: `core.py`를 수정했지만 Python이 이전 버전을 사용

**해결**:
```bash
# Python 재시작
exit()
python

# 또는 강제 재로드
import importlib
import core
importlib.reload(core)
```

---

### 4. 초기화 중 에러 발생

**문제**: `__init__` 실행 중 다른 에러로 중단

**디버깅**:
```python
import traceback
import core
import gym

env = gym.make('Safexp-PointGoal1-v0')

try:
    ac = core.RCPPOActorCritic(
        env.observation_space,
        env.action_space,
        hidden_sizes=[256, 256]
    )
    print("✅ Initialization successful")
    print(f"Has pi: {hasattr(ac, 'pi')}")
    print(f"Has v_reach: {hasattr(ac, 'v_reach')}")
except Exception as e:
    print("❌ Initialization failed:")
    traceback.print_exc()
```

---

### 5. 잘못된 저장/로드 방법

**문제**: 전체 객체를 pickle로 저장
```python
# 잘못됨
torch.save(ac, 'model.pt')  # 전체 객체 저장
```

**올바른 방법**:
```python
# 저장 (state_dict만)
torch.save(ac.state_dict(), 'model.pt')

# 로드
ac = core.RCPPOActorCritic(env.observation_space, env.action_space)
ac.load_state_dict(torch.load('model.pt'))
```

---

## 빠른 테스트

```python
#!/usr/bin/env python
import gym
import safety_gym
import core
import torch
import numpy as np

print("=== RCPPOActorCritic Test ===")

# 1. 환경 생성
env = gym.make('Safexp-PointGoal1-v0')
print(f"✓ Environment created: {env}")
print(f"  Obs space: {env.observation_space.shape}")
print(f"  Act space: {env.action_space}")

# 2. Actor-Critic 생성
ac = core.RCPPOActorCritic(
    env.observation_space,
    env.action_space,
    hidden_sizes=[256, 256]
)
print(f"✓ RCPPOActorCritic created")
print(f"  Has pi: {hasattr(ac, 'pi')}")
print(f"  Has v_reach: {hasattr(ac, 'v_reach')}")

# 3. Step 테스트
obs = env.reset()
obs_aug = np.concatenate([obs, [-1.0], [100.0]])
a, v, logp = ac.step(torch.as_tensor(obs_aug, dtype=torch.float32))
print(f"✓ Step test passed")
print(f"  Action shape: {a.shape}")
print(f"  Value: {v}")

print("\n=== All tests passed! ===")
```

---

## 최신 core.py 확인

`core.py`의 `RCPPOActorCritic.__init__`이 다음과 같은지 확인:

```python
def __init__(self, observation_space, action_space,
             hidden_sizes=(64,64), activation=nn.Tanh):
    super().__init__()

    obs_dim = observation_space.shape[0]
    aug_obs_dim = obs_dim + 2  # x, y, z

    if isinstance(action_space, Box):
        self.pi = MLPGaussianActor(aug_obs_dim, action_space.shape[0],
                                   hidden_sizes, activation)
    elif isinstance(action_space, Discrete):
        self.pi = MLPCategoricalActor(aug_obs_dim, action_space.n,
                                      hidden_sizes, activation)
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")

    self.v_reach = MLPCritic(aug_obs_dim, hidden_sizes, activation)
```

**중요**: `else` 절이 추가되어 지원되지 않는 action space에 대해 명확한 에러를 발생시킵니다.

---

## 여전히 문제가 있다면

다음 정보를 공유해주세요:

1. **전체 traceback**:
   ```python
   import traceback
   try:
       # 에러 발생 코드
   except Exception as e:
       traceback.print_exc()
   ```

2. **환경 정보**:
   ```python
   print(f"Environment: {env}")
   print(f"Action space: {env.action_space}")
   print(f"Action space type: {type(env.action_space)}")
   ```

3. **실행 중인 코드**:
   - 어떤 스크립트? (`rcppo.py`, 커스텀 스크립트?)
   - 모델 로드 중? 초기화 중?

---

## 커밋 후 Git 상태 확인

```bash
# 최신 코드 확인
git status
git log --oneline -5

# core.py 최신 버전 확인
git diff HEAD core.py

# 필요시 최신 버전으로 리셋
git checkout HEAD -- core.py
```

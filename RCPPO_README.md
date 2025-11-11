# RC-PPO Implementation

Implementation of **Reach Constrained Proximal Policy Optimization (RC-PPO)** from the paper:
> "Solving Minimum-Cost Reach Avoid using Reinforcement Learning" (NeurIPS 2024)
> [arXiv:2410.22600](https://arxiv.org/abs/2410.22600)

## Overview

RC-PPO solves the **minimum-cost reach-avoid problem**: reaching a goal while avoiding unsafe states and minimizing cumulative cost.

### Key Features

- **Augmented State Space**: (x, y, z) where:
  - `x`: Original state
  - `y`: Safety flag (-1: safe, +1: violated)
  - `z`: Remaining cost budget

- **Min-based Bellman Equation**: Uses reachability analysis
  ```
  Ṽ(x̂_t) = (1-γ)ĝ(x̂_t) + γE[min{ĝ(x̂_t), Ṽ(x̂_{t+1})}]
  ```

- **Two-Phase Training**:
  - **Phase 1**: Learn z-conditioned policy and value function
  - **Phase 2**: Find optimal z* via bisection

## Files

- `core.py`: Added `RCPPOActorCritic` class for augmented state
- `rcppo.py`: Main RC-PPO algorithm implementation

## Usage

### Phase 1 Training

```bash
python rcppo.py --env Safexp-PointGoal1-v0 --epochs 333 --cpu 4
```

### Arguments

- `--env`: Environment name (default: `Safexp-PointGoal1-v0`)
- `--hid`: Hidden layer size (default: 256)
- `--l`: Number of hidden layers (default: 2)
- `--gamma`: Discount factor (default: 0.99)
- `--gamma_reach`: Reachability discount (default: 0.99)
- `--z_min`: Minimum cost bound (default: -100.0)
- `--z_max`: Maximum cost bound (default: 1000.0)
- `--epochs`: Number of training epochs (default: 333)
- `--steps`: Steps per epoch (default: 30000)
- `--cpu`: Number of CPUs for MPI (default: 4)

### Custom Goal/Avoid Functions

You can define custom functions:

```python
def goal_fn(x):
    """Return <= 0 if in goal region"""
    return np.linalg.norm(x[:2]) - 0.5

def avoid_fn(x):
    """Return > 0 if in unsafe region"""
    return -1.0  # No unsafe region

def cost_fn(x, u):
    """Compute step cost"""
    return np.sum(u**2)

rcppo(env_fn, goal_fn=goal_fn, avoid_fn=avoid_fn, cost_fn=cost_fn)
```

## Algorithm Details

### Augmented Goal Function

```python
ĝ(x, y, z) = max{g(x), C·y, -z}
```

Where:
- `g(x)`: Original goal function
- `C`: Penalty constant (default: 10.0)
- `y`: Safety flag
- `z`: Cost budget

### Augmented Dynamics

```python
x_{t+1} = f(x_t, u_t)
y_{t+1} = max{I_{x in F}, y_t}
z_{t+1} = z_t - c(x_t, u_t)
```

### Phase 2: Finding Optimal z*

After Phase 1 training:

```python
# Fine-tune value function
ac_finetuned = finetune_value_function(ac, env_fn, goal_fn, avoid_fn, cost_fn, C_const)

# Find optimal z* for initial state x0
z_star = find_optimal_z(ac_finetuned, x0, goal_fn, avoid_fn, z_min=-100, z_max=1000)

# Execute with optimal policy
obs_aug = np.concatenate([x0, [y0], [z_star]])
action = ac_finetuned.act_deterministic(torch.as_tensor(obs_aug, dtype=torch.float32))
```

## Comparison with PPO-Lagrangian

| Feature | PPO-Lagrangian | RC-PPO |
|---------|---------------|--------|
| Objective | Maximize reward with cost constraint | Minimize cost with reach-avoid constraint |
| State | x | (x, y, z) |
| Value Function | V(x), Vc(x) | Ṽ(x, y, z) |
| Bellman | Standard | Min-based |
| Training | Single phase | Two phases |
| Cost Handling | Lagrange multiplier | Cost budget z |

## Key Differences from Paper

- Simplified goal/avoid functions for ease of use
- Default Safety Gym environment support
- MPI parallelization for faster training
- Modular design for custom environments

## Example Results

RC-PPO achieves:
- **Comparable reach rates** to baseline methods
- **Up to 57% lower cumulative costs** (as reported in paper)
- **Better safety constraint satisfaction**

## Citation

```bibtex
@inproceedings{rcppo2024,
  title={Solving Minimum-Cost Reach Avoid using Reinforcement Learning},
  booktitle={NeurIPS},
  year={2024}
}
```

## Notes

- No try-except blocks used (as requested)
- Leverages existing PPO codebase structure
- Concise implementation focusing on core algorithm
- Compatible with Safety Gym environments

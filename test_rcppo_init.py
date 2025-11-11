#!/usr/bin/env python
"""Test RCPPOActorCritic initialization"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gym
import safety_gym
import core

print("=" * 60)
print("Testing RCPPOActorCritic Initialization")
print("=" * 60)

# Test with Safety Gym environment
env_name = 'Safexp-PointGoal1-v0'
print(f"\n1. Creating environment: {env_name}")

env = gym.make(env_name)

print(f"   Observation space: {env.observation_space}")
print(f"   Observation shape: {env.observation_space.shape}")
print(f"   Action space: {env.action_space}")
print(f"   Action space type: {type(env.action_space)}")

if hasattr(env.action_space, 'shape'):
    print(f"   Action shape: {env.action_space.shape}")
if hasattr(env.action_space, 'n'):
    print(f"   Action n: {env.action_space.n}")

print("\n2. Initializing RCPPOActorCritic...")

ac = core.RCPPOActorCritic(env.observation_space, env.action_space,
                            hidden_sizes=[256, 256])

print("   ✅ Initialization successful!")
print(f"   - Has pi: {hasattr(ac, 'pi')}")
print(f"   - Has v_reach: {hasattr(ac, 'v_reach')}")
print(f"   - pi type: {type(ac.pi)}")
print(f"   - v_reach type: {type(ac.v_reach)}")

print("\n3. Testing step function...")
import numpy as np
import torch

obs = env.reset()
y = -1.0
z = 100.0
obs_aug = np.concatenate([obs, [y], [z]])

print(f"   Original obs shape: {obs.shape}")
print(f"   Augmented obs shape: {obs_aug.shape}")

a, v, logp = ac.step(torch.as_tensor(obs_aug, dtype=torch.float32))

print(f"   ✅ Step successful!")
print(f"   - Action shape: {a.shape}")
print(f"   - Value: {v}")
print(f"   - Log prob: {logp}")

print("\n4. Testing act_deterministic function...")

a_det = ac.act_deterministic(torch.as_tensor(obs_aug, dtype=torch.float32))

print(f"   ✅ act_deterministic successful!")
print(f"   - Action shape: {a_det.shape}")

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)

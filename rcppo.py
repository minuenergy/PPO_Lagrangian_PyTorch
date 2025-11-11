import numpy as np
import torch
from torch.optim import Adam
import gym
import safety_gym
import time
import core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class RCPPOBuffer:
    """Buffer for RC-PPO with augmented state (x, y, z)"""

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97, gamma_reach=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.y_buf = np.zeros(size, dtype=np.float32)
        self.z_buf = np.zeros(size, dtype=np.float32)
        self.g_hat_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam, self.gamma_reach = gamma, lam, gamma_reach
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, y, z, g_hat, val, logp, rew):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.y_buf[self.ptr] = y
        self.z_buf[self.ptr] = z
        self.g_hat_buf[self.ptr] = g_hat
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        vals = np.append(self.val_buf[path_slice], last_val)
        g_hats = self.g_hat_buf[path_slice]

        deltas = np.zeros(len(vals) - 1, dtype=np.float32)
        for t in range(len(deltas)):
            deltas[t] = (1 - self.gamma_reach) * g_hats[t] + self.gamma_reach * np.minimum(g_hats[t], vals[t+1]) - vals[t]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]
        self.path_start_idx = self.ptr

    def get(self):
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(obs=self.obs_buf, act=self.act_buf, y=self.y_buf, z=self.z_buf,
                    ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def compute_g_hat(x, y, z, goal_fn, avoid_fn, C_const):
    """Compute augmented goal function: g_hat(x,y,z) = max{g(x), C*y, -z}"""
    g_val = goal_fn(x)
    h_val = avoid_fn(x)
    return np.maximum(np.maximum(g_val, C_const * y), -z)


def update_y(y_prev, x, avoid_fn):
    """Update safety flag: y = max{I_{x in F}, y_prev}"""
    in_avoid = 1.0 if avoid_fn(x) > 0 else -1.0
    return max(in_avoid, y_prev)


def rcppo(env_fn, actor_critic=core.RCPPOActorCritic, ac_kwargs=dict(), seed=0,
          steps_per_epoch=30000, epochs=333, gamma=0.99, gamma_reach=0.99, clip_ratio=0.2,
          pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
          max_ep_len=1000, target_kl=0.01, z_min=-100, z_max=1000,
          logger_kwargs=dict(), save_freq=10, goal_fn=None, avoid_fn=None, cost_fn=None):

    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    if goal_fn is None:
        goal_fn = lambda x: np.linalg.norm(x[:2]) - 0.5
    if avoid_fn is None:
        avoid_fn = lambda x: -1.0
    if cost_fn is None:
        cost_fn = lambda x, u: np.sum(u**2)

    C_const = 10.0

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = RCPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, gamma_reach)

    def compute_loss_pi(data):
        obs, act, y, z, adv, logp_old = data['obs'], data['act'], data['y'], data['z'], data['adv'], data['logp']
        obs_aug = torch.cat([obs, y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

        pi, logp = ac.pi(obs_aug, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, y, z, ret = data['obs'], data['y'], data['z'], data['ret']
        obs_aug = torch.cat([obs, y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
        return ((ac.v_reach(obs_aug) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v_reach.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v_reach)
            vf_optimizer.step()

        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            z0 = np.random.uniform(z_min, z_max)
            y = -1.0 if avoid_fn(o) <= 0 else 1.0
            z = z0

            obs_aug = np.concatenate([o, [y], [z]])
            a, v, logp = ac.step(torch.as_tensor(obs_aug, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            cost = cost_fn(o, a)
            ep_ret += r
            ep_cost += cost
            ep_len += 1

            y_next = update_y(y, next_o, avoid_fn)
            z_next = z - cost
            g_hat = compute_g_hat(o, y, z, goal_fn, avoid_fn, C_const)

            buf.store(o, a, y, z, g_hat, v, logp, r)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)

                if timeout or epoch_ended:
                    y_final = -1.0 if avoid_fn(o) <= 0 else 1.0
                    z_final = z_next
                    obs_aug_final = np.concatenate([o, [y_final], [z_final]])
                    _, v, _ = ac.step(torch.as_tensor(obs_aug_final, dtype=torch.float32))
                else:
                    v = 0

                buf.finish_path(v)

                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)

                o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


def finetune_value_function(ac, env_fn, goal_fn, avoid_fn, cost_fn, C_const,
                            num_rollouts=100, max_ep_len=1000, vf_lr=1e-3, train_v_iters=50):
    """Phase 2: Fine-tune value function with deterministic policy"""
    env = env_fn()
    vf_optimizer = Adam(ac.v_reach.parameters(), lr=vf_lr)

    for rollout in range(num_rollouts):
        z0 = np.random.uniform(-100, 1000)
        o = env.reset()
        y = -1.0 if avoid_fn(o) <= 0 else 1.0
        z = z0

        traj_obs, traj_y, traj_z, traj_g_hat, traj_vals = [], [], [], [], []

        for t in range(max_ep_len):
            obs_aug = np.concatenate([o, [y], [z]])
            a = ac.act_deterministic(torch.as_tensor(obs_aug, dtype=torch.float32))

            with torch.no_grad():
                v = ac.v_reach(torch.as_tensor(obs_aug, dtype=torch.float32)).numpy()

            g_hat = compute_g_hat(o, y, z, goal_fn, avoid_fn, C_const)

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

        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            total_loss = 0
            for i in range(len(traj_obs)):
                obs_aug = torch.as_tensor(np.concatenate([traj_obs[i], [traj_y[i]], [traj_z[i]]]), dtype=torch.float32)
                v_pred = ac.v_reach(obs_aug)

                if i < len(traj_obs) - 1:
                    target = (1 - 0.99) * traj_g_hat[i] + 0.99 * min(traj_g_hat[i], traj_vals[i+1])
                else:
                    target = traj_g_hat[i]

                total_loss += (v_pred - target) ** 2

            loss = total_loss / len(traj_obs)
            loss.backward()
            vf_optimizer.step()

    return ac


def find_optimal_z(ac, x0, goal_fn, avoid_fn, z_min=-100, z_max=1000, tol=0.1, max_iter=50):
    """Phase 2: Find optimal z* using bisection such that V_g_hat(x0, y0, z*) <= 0"""
    y0 = -1.0 if avoid_fn(x0) <= 0 else 1.0

    def eval_value(z):
        obs_aug = torch.as_tensor(np.concatenate([x0, [y0], [z]]), dtype=torch.float32)
        with torch.no_grad():
            v = ac.v_reach(obs_aug).numpy()
        return v

    z_low, z_high = z_min, z_max

    for iteration in range(max_iter):
        z_mid = (z_low + z_high) / 2
        v_mid = eval_value(z_mid)

        if abs(v_mid) < tol or (z_high - z_low) / 2 < tol:
            return z_mid

        if v_mid > 0:
            z_low = z_mid
        else:
            z_high = z_mid

    return (z_low + z_high) / 2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_reach', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=333)
    parser.add_argument('--exp_name', type=str, default='rcppo')
    parser.add_argument('--z_min', type=float, default=-100.0)
    parser.add_argument('--z_max', type=float, default=1000.0)
    parser.add_argument('--phase2', action='store_true', help='Run Phase 2 fine-tuning after Phase 1')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    rcppo(lambda: gym.make(args.env), actor_critic=core.RCPPOActorCritic,
          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
          gamma_reach=args.gamma_reach, seed=args.seed, steps_per_epoch=args.steps,
          epochs=args.epochs, z_min=args.z_min, z_max=args.z_max,
          logger_kwargs=logger_kwargs)

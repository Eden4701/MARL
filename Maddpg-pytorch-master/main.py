import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()

#平行环境
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            #print("init_env")
            env = make_env("simple_spread",env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000) # Sets the seed for this env's random number generator(s)
            np.random.seed(seed + rank * 1000)
            #print("env",env)
            return env
        #print("init_env",init_env)
        return init_env
    if n_rollout_threads == 1:

        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    # 路径
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed) # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    np.random.seed(config.seed) #得到相同的或部分形同的随机数。
    # 使用CPU
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads) # cpu多线程并行计算
    # 生成环境
    print("生成环境")
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    print("初始化MADDPG 和经验池")
    # 从多代理环境中实例化此类MADDPG　的实例
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)

    #　经验池
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()# 返回agent动作（数组）的矩阵
        # [[[ 0.          0.          0.80675904  0.14735897 -0.03487484
        #    -0.43281945  0.01031126  0.09936126 -1.77511656  0.71151549
        #    -1.80101839  0.08693085 -1.15346924 -0.09324277  0.
        #     0.          0.          0.        ]
        #   [ 0.          0.         -0.99425935  0.23428983  1.76614354
        #    -0.51975031  1.81132965  0.0124304   0.02590183  0.62458464
        #     1.80101839 -0.08693085  0.64754915 -0.18017362  0.
        #     0.          0.          0.        ]
        #   [ 0.          0.         -0.3467102   0.0541162   1.1185944
        #    -0.33957668  1.1637805   0.19260403 -0.62164732  0.80475826
        #     1.15346924  0.09324277 -0.64754915  0.18017362  0.
        #     0.          0.          0.        ]]]
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        # noise()
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            #print("obs",obs)
            # 从矩阵变成张量
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # print("torch_obs",torch_obs)
            # [tensor([[ 0.0000,  0.0000, -0.1660,  0.4406, -0.4615, -0.7495, -0.0405, -0.3630,
            #           0.0043, -0.0702, -0.8338, -0.8360, -0.5405, -1.2560,  0.0000,  0.0000,
            #           0.0000,  0.0000]]), tensor([[ 0.0000,  0.0000, -0.9998, -0.3953,  0.3723,  0.0865,  0.7933,  0.4730,
            #           0.8382,  0.7658,  0.8338,  0.8360,  0.2933, -0.4200,  0.0000,  0.0000,
            #           0.0000,  0.0000]]), tensor([[ 0.0000,  0.0000, -0.7065, -0.8153,  0.0790,  0.5064,  0.5000,  0.8930,
            #           0.5449,  1.1858,  0.5405,  1.2560, -0.2933,  0.4200,  0.0000,  0.0000,
            #           0.0000,  0.0000]])]

            # get actions as torch Variables ，各智能体使用本地观测和策略网络选择动作
            # Actor根据当前的state选择一个action
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # print("torch_agent_actions",torch_agent_actions)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]#数组的列表
            #print("agent_actions",agent_actions) #

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)] #[数组的列表]
            #print("actions",actions) [[array([0., 0., 1., 0., 0.], dtype=float32), array([0., 1., 0., 0., 0.], dtype=float32), array([1., 0., 0., 0., 0.], dtype=float32)]]

            next_obs, rewards, dones, infos = env.step(actions) #
            #print("next_obs",next_obs)
            #print("next_obs",next_obs) #所有智能体obs

            #各自存储各自经验
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            #更新观测
            obs = next_obs

            t += config.n_rollout_threads

            #更新model
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')

                # 每个agent本地buffer
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread",help="Name of environment")
    parser.add_argument("--model_name",default="./model",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=300, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)

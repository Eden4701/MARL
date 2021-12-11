"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    print("worker")
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    print("SubprocVecEnv(VecEnv)")
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        print("step_wait")
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos # np.stack默认以第0维（即整个数组）堆叠

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    print("DummyVecEnv(VecEnv)")
    def __init__(self, env_fns):
        #print("env_fns",len(env_fns)) #1

        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        #print("env",env)
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
    # 用这个
    def step_wait(self):
        #print("DummyVecEnv_step_wait")
        # 在这个action上step,
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)] #environment.step
        # print("results",results)#
        # results [([array([ 0.5       ,  0.        , -0.11595599,  0.44064899, -0.51152359,
        #        -0.74952753, -0.09050906, -0.36301552, -0.04565498, -0.07020999,
        #        -0.83381526, -0.83598384, -0.54053223, -1.2559718 ,  0.        ,
        #         0.        ,  0.        ,  0.        ]), array([ 5.00000000e-01,  5.40418852e-95, -9.49771250e-01, -3.95334855e-01,
        #         3.22291673e-01,  8.64563088e-02,  7.43306199e-01,  4.72968323e-01,
        #         7.88160279e-01,  7.65773856e-01,  8.33815260e-01,  8.35983842e-01,
        #         2.93283032e-01, -4.19987956e-01,  0.00000000e+00,  0.00000000e+00,
        #         0.00000000e+00,  0.00000000e+00]), array([ 5.00000000e-01, -5.40418852e-95, -6.56488218e-01, -8.15322810e-01,
        #         2.90086411e-02,  5.06444265e-01,  4.50023167e-01,  8.92956278e-01,
        #         4.94877247e-01,  1.18576181e+00,  5.40532228e-01,  1.25597180e+00,
        #        -2.93283032e-01,  4.19987956e-01,  0.00000000e+00,  0.00000000e+00,
        #         0.00000000e+00,  0.00000000e+00])], [-5.374690446912888, -5.374690446912888, -5.374690446912888], [False, False, False], {'n': [(-1.791563482304296, 1, 0.7915634823042961, 1), (-1.791563482304296, 1, 0.7915634823042961, 1), (-1.791563482304296, 1, 0.7915634823042961, 1)]})]

        # map根据提供的函数对指定序列做映射
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        # print("obs1",obs[0])
        # obs1 [[ 5.00000000e-01  0.00000000e+00 -1.15955991e-01  4.40648987e-01
        #   -5.11523587e-01 -7.49527533e-01 -9.05090609e-02 -3.63015519e-01
        #   -4.56549806e-02 -7.02099861e-02 -8.33815260e-01 -8.35983842e-01
        #   -5.40532228e-01 -1.25597180e+00  0.00000000e+00  0.00000000e+00
        #    0.00000000e+00  0.00000000e+00]
        #  [ 5.00000000e-01  5.40418852e-95 -9.49771250e-01 -3.95334855e-01
        #    3.22291673e-01  8.64563088e-02  7.43306199e-01  4.72968323e-01
        #    7.88160279e-01  7.65773856e-01  8.33815260e-01  8.35983842e-01
        #    2.93283032e-01 -4.19987956e-01  0.00000000e+00  0.00000000e+00
        #    0.00000000e+00  0.00000000e+00]
        #  [ 5.00000000e-01 -5.40418852e-95 -6.56488218e-01 -8.15322810e-01
        #    2.90086411e-02  5.06444265e-01  4.50023167e-01  8.92956278e-01
        #    4.94877247e-01  1.18576181e+00  5.40532228e-01  1.25597180e+00
        #   -2.93283032e-01  4.19987956e-01  0.00000000e+00  0.00000000e+00
        #    0.00000000e+00  0.00000000e+00]]
        for (i, done) in enumerate(dones):
            if all(done):
                #print("all")
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0

        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs] # environment中的reset，返回agent动作（数组）的列表
        #print("result",results)
        # [[array([ 0.        ,  0.        , -0.72144731,  0.61478258,  1.22307151,
        #        -0.16278661,  1.48805949, -0.36743816,  1.22333217, -0.91698589,
        #         0.51680098, -1.28407418,  1.57646447, -0.91925086,  0.        ,
        #         0.        ,  0.        ,  0.        ]), array([ 0.        ,  0.        , -0.20464633, -0.66929161,  0.70627053,
        #         1.12128758,  0.97125851,  0.91663602,  0.70653119,  0.36708829,
        #        -0.51680098,  1.28407418,  1.05966349,  0.36482333,  0.        ,
        #         0.        ,  0.        ,  0.        ]), array([ 0.        ,  0.        ,  0.85501716, -0.30446828, -0.35339295,
        #         0.75646425, -0.08840498,  0.55181269, -0.35313229,  0.00226496,
        #        -1.57646447,  0.91925086, -1.05966349, -0.36482333,  0.        ,
        #         0.        ,  0.        ,  0.        ])]]
        return np.array(results)# 将数据转化为矩阵

    def close(self):
        return
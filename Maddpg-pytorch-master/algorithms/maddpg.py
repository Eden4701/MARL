import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete

from enc.he import convolution_enc, fl_enc
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, soft_update2
from utils.agents import DDPGAgent
from enc import  he
import  tenseal as ts
import numpy  as np
MSELoss = torch.nn.MSELoss()#均方损失函数

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    # 使用actor网络
    def step(self, observations, explore=False):
        #print("step")
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        # ake a step forward in environment for a minibatch of observations
        w=[a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]
        return w

    # 基于来自重放缓冲区的样本更新代理模型的参数
    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        # Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        #curr_agent.critic_optimizer.zero_grad()

        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
                # all_trgt_acs，每个agent onehot_from_logits（target_policies（next_obs））的列表，ENC
                # print("all_trgt_acs",all_trgt_acs)
                # [tensor([[1., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 1.],
                #         [1., 0., 0., 0., 0.],
                #         ...,
                #         [1., 0., 0., 0., 0.],
                #         [1., 0., 0., 0., 0.],
                #         [1., 0., 0., 0., 0.]]), tensor([[0., 0., 0., 0., 1.],
                #         [0., 1., 0., 0., 0.],
                #print("next_obs",next_obs)
                # [tensor([[ 0.3102, -0.2290, -0.8500,  ...,  0.0000,  0.0000,  0.0000],
                #         [-0.0879, -0.5371,  0.4501,  ...,  0.0000,  0.0000,  0.0000],
                #         [-0.5014,  0.2442,  0.0802,  ...,  0.0000,  0.0000,  0.0000],
                #         ...,
                #         [ 0.2812,  0.1250, -0.0503,  ...,  0.0000,  0.0000,  0.0000],
                #         [ 0.3750,  0.9922,  0.2096,  ...,  0.0000,  0.0000,  0.0000],
                #         [ 0.6568,  0.6939, -0.1404,  ...,  0.0000,  0.0000,  0.0000]]), tensor([[-0.0148, -0.2244,  1.2587,  ...,  0.0000,  0.0000,  0.0000],
                #         [ 0.4659, -0.1590,  0.6370,  ...,  0.0000,  0.0000,  0.0000],
                #next_obs=he.enc1(next_obs)
                #print("next_obs",next_obs)

            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            # 将两个张量（tensor）拼接在一起,1维（列）
            #print(" *next_obs",*next_obs)
            #all_trgt_acs=he.enc(all_trgt_acs)



            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

            print("trgt_vf_in",trgt_vf_in.shape,trgt_vf_in)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)

        # critic 网络forward
        "加密"
        aa=convolution_enc(trgt_vf_in)
        #aa = fl_enc(trgt_vf_in)
        #print("typea",len(aa),type(aa[0]))
        #out = ts.pack_vectors(aa)
        # for i in range(0,len(aa)):
        #     a=curr_agent.target_critic(aa[i])
        a = curr_agent.target_critic(aa)
            #print("curr_agent.target_critic(trgt_vf_in)",a)
        print(" self.gamma", self.gamma)
        print("(1 - dones[agent_i].view(-1, 1))",(1 - dones[agent_i].view(-1, 1)[0:5]).tolist())
        print("a",a)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        a *(torch.tensor([1], dtype=torch.long)).t())
                        #(1 - dones[agent_i].view(-1, 1)[0:5]).tolist())
        print("target_value",target_value)# critic目标网络
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
            print("vf_in",vf_in.shape,type(vf_in),vf_in)#torch.Size([300, 69]) <class 'torch.Tensor'>
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        print("curr_agent.critic_actual_value")
        "加密"
        p=convolution_enc(vf_in)
        #p = fl_enc(vf_in)
        actual_value = curr_agent.critic(p)#critic当前网络
        print("actual_value",actual_value)
        e=target_value.detach()
        print("target_value.detach()",e)
        vf_loss = MSELoss(actual_value, e)#均方损失函数
        print("vf_loss",vf_loss)
        "后向传播函数待实现"
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        "critic暂时跳过梯度截断、优化器"
        #torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)#梯度截断Clip, 将梯度约束在某一个区间之内
        #curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        print("self.alg_types[agent_i] ")
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        print("pol_loss")
        "加密"
        w=convolution_enc(vf_in)
        #w=fl_enc(vf_in)
        pol_loss = -curr_agent.critic(w).mean()#平均值
        inss= (curr_pol_out**2).mean() * 1e-3
        print("(curr_pol_out**2).mean() * 1e-3",inss)
        pol_loss += inss
        "后向传播函数待实现"
        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        print("update_all_targets")
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            print("soft_update")
            "critic软更新（soft_update2）重写"
            soft_update2(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    # Sets the module in training mode.
    def prep_training(self, device='gpu'):
        print("prep_training")
        for a in self.agents:
            a.policy.train()
            #a.critic.train()
            a.target_policy.train()
            #a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        print("prep_rollouts")
        for a in self.agents:
            # policy网络，局部
            a.policy.eval()
            # MLPNetwork(
            #   (in_fn): BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (fc1): Linear(in_features=18, out_features=64, bias=True)
            #   (fc2): Linear(in_features=64, out_features=64, bias=True)
            #   (fc3): Linear(in_features=64, out_features=5, bias=True)
            # )

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        # print("self.pol_dev",self.pol_dev)# cpu
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
                print("a.policy ",a.policy )
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
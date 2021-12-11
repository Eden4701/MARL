from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam

from .networks2 import MLPNetwork2, EncryptedLR, LR
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits, hard_update2
from .noise import OUNoise

class DDPGAgent(object):
    #print("DDPGAgent")
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        "critic、target_critic加密网络（1）EncryptedLR卷积加密方式，（2）向量加密方式"
        lr_p = LR(4)
        self.critic =EncryptedLR(lr_p)
        self.target_critic =EncryptedLR(lr_p)
        # self.target_critic = MLPNetwork2(num_in_critic, 1,
        #                                  hidden_dim=hidden_dim,
        #                                  constrain_out=False)
        # self.critic=MLPNetwork2(num_in_critic, 1,
        #                          hidden_dim=hidden_dim,
        #                          constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)


        # Copy network parameters from source to target
        hard_update(self.target_policy, self.policy)
        "卷积方式加密网络重写hard_update2"
        hard_update2(self.target_critic, self.critic)
        # print("self.policy.parameters()",self.policy.parameters())
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        # print("self.policy_optimizer",self.policy_optimizer)
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # print("obs",obs) # 智能体局部观测 tensor([[ 0.0000,  0.0000, -0.1660,  0.4406, -0.4615, -0.7495, -0.0405, -0.3630,
        #           0.0043, -0.0702, -0.8338, -0.8360, -0.5405, -1.2560,  0.0000,  0.0000,
        #           0.0000,  0.0000]])

        action = self.policy(obs) # 用本地策略网络选择一个动作
        # print("policy action",action) # tensor([[ 0.1458,  0.0216,  0.0837, -0.0755,  0.0588]], grad_fn=<AddmmBackward0>)
        if self.discrete_action:
            if explore:
                # 用这个
                # 从Gumbel-Softmax分布中抽取样本，并可选地离散化
                action = gumbel_softmax(action, hard=True)
            else:

                action = onehot_from_logits(action)# 给定一批逻辑，使用ε贪婪策略返回一个热点样本(基于给定的ε)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        #print("action",action)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                #'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
               # 'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}
                #'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        #self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        #self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        #self.critic_optimizer.load_state_dict(params['critic_optimizer'])

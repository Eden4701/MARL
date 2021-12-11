import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    print("MLPNetwork")
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        #初始化内部模块状态
        super(MLPNetwork, self).__init__()
        print("input_dim", input_dim)

        if norm_in:  # normalize inputs

            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        # Applies a linear transformation to the incoming data:
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        print("forward1")
        # print(X.shape, X)  # [1,18]，2维
        # tensor([[-0.5059,  0.2812,  1.0706,  0.7235, -2.0060, -0.3048, -1.1406,  0.1716,
        #          -1.6278, -1.1894, -1.3294, -0.2890, -1.3882, -0.8531,  0.0000,  0.0000,
        #           0.0000,  0.0000]])
        # print(self.fc1.weight.shape)#[64, 18]
        # print(self.fc1.bias)#【64] ,1维
        # tensor([-0.0679, -0.1409,  0.2001,  0.0598, -0.2118,  0.1319, -0.0828,  0.0689,
        #          0.2162, -0.0726, -0.0466,  0.1018,  0.0831, -0.1359, -0.1881, -0.1421,
        #         -0.1277, -0.2280, -0.0047,  0.1787, -0.2353, -0.1534, -0.1181, -0.1822,
        #          0.0539,  0.1399,  0.0956,  0.0695, -0.1912, -0.0358, -0.0398,  0.1810,
        #         -0.2180, -0.0335, -0.1744, -0.2167, -0.2016, -0.1961, -0.0655,  0.0714,
        #         -0.1589,  0.1236,  0.0703, -0.1167, -0.1780, -0.0407, -0.0600,  0.2119,
        #          0.1130,  0.1196, -0.1086, -0.0076, -0.2055, -0.0912,  0.1406,  0.1173,
        #         -0.1531,  0.2131, -0.0173,  0.1503, -0.0479,  0.0274, -0.0391, -0.0278],
        #        requires_grad=True)


        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print("X",X)
        #  tensor([[ 0.3359,  0.0000, -0.6222,  0.6148,  1.1239, -0.1628,  1.3888, -0.3674,
        #           1.1241, -0.9170,  0.1934, -1.1184,  1.5929, -0.8693,  0.0000,  0.0000,
        #           0.0000,  0.0000]])
        #print("self.in_fn(X) ",self.in_fn(X))
        # tensor([[ 0.3359,  0.0000, -0.6222,  0.6148,  1.1238, -0.1628,  1.3888, -0.3674,
        #           1.1241, -0.9170,  0.1934, -1.1184,  1.5929, -0.8692,  0.0000,  0.0000,
        #           0.0000,  0.0000]], grad_fn=<NativeBatchNormBackward0>)
        #print("self.fc1(self.in_fn(X))",self.fc1(self.in_fn(X)))
        h1 = self.nonlin(self.fc1(self.in_fn(X)))

        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        #print(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import  tenseal as ts
import numpy as np

from enc.he import enc1


class MLPNetwork2(nn.Module):
    print("MLPNetwork2")
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
        super(MLPNetwork2, self).__init__()

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
        print("forward2")
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """

        #print("X",X.shape) #torch.Size([1024, 69])
        #  tensor([[ 0.3359,  0.0000, -0.6222,  0.6148,  1.1239, -0.1628,  1.3888, -0.3674,
        #           1.1241, -0.9170,  0.1934, -1.1184,  1.5929, -0.8693,  0.0000,  0.0000,
        #           0.0000,  0.0000]])
        #print("self.in_fn(X) ",self.in_fn(X))
        # tensor([[ 0.3359,  0.0000, -0.6222,  0.6148,  1.1238, -0.1628,  1.3888, -0.3674,
        #           1.1241, -0.9170,  0.1934, -1.1184,  1.5929, -0.8692,  0.0000,  0.0000,
        #           0.0000,  0.0000]], grad_fn=<NativeBatchNormBackward0>)

        # print(X.shape,X) #[1024, 69]
        # print(self.fc1.weight.shape)#[64, 69]
        # print(self.fc1.bias.shape)#64
        out = ts.pack_vectors(X)
        #print("self.in_fn.weight",self.in_fn.weight)

        a1=out.mm_(self.in_fn.weight).add_plain_(self.in_fn.bias)
        h1 = self.nonlin(a1)

        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class LR(torch.nn.Module):
    print("LR初始化")

    def __init__(self, n_features):

        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 4)#全连接层
        self.co = torch.nn.Conv2d(1,1,64,padding=1)#卷积层

    def forward(self, x):
        out = torch.sigmoid(self.co(x))
        print("x",x)
        return out
class EncryptedLR:
    print("EncryptedLR初始化")
    def __init__(self, torch_lr):

        self.weight = torch_lr.lr.weight.data.tolist()[0]
        print("self.weight初始化",len(self.weight),self.weight)#[0.26818111538887024, 0.3282051384449005, -0.02741098403930664, 0.1605906
        self.bias = torch_lr.lr.bias.data.tolist()
        self.ker=torch_lr.co.weight
        print("self.ker",self.ker)
        print("self.bias", self.bias)
        # we accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        print("forword2")
        print("self.weight",len(self.weight),self.weight)
        out=[]
        n=0
        print("self.ker",self.ker.data[0][0].shape,self.ker.data[0][0].tolist())
        enc_out = enc_x.conv2d_im2col(self.ker.data[0][0].tolist(),4)#卷积层

          #enc_out = enc_x.mm_([self.weight]).add_plain_(self.bias)
        #enc_out = enc_x.dot(self.weight) + self.bias
        #enc_out = EncryptedLR.sigmoid(enc_out)
        "此处非线性计算，待实现communication"
        a=enc_out.decrypt()
        c=torch.tensor(a,requires_grad=True)
        print("enc_out.decrypt()",c)
        b=F.relu(c)
        print("b",b)
        enc_out2=enc1(b.tolist())
        print("self.weight",self.weight)
        #enc_out3 = enc_out2.mm_([self.weight]).add_plain_(self.bias)
        enc_out3 = enc_out2.dot(self.weight)#全连接层
        "前向传播的结果暂时解密输出"
        d=enc_out3.decrypt()
        d = torch.tensor(d, requires_grad=True)
        print("enc_out3",d)
        return d

    # def backward(self, enc_x, enc_out, enc_y):
    #     print("backward")
    #     out_minus_y = (enc_out - enc_y)
    #     self._delta_w += enc_x * out_minus_y
    #     self._delta_b += out_minus_y
    #     self._count += 1

    # def update_parameters(self):
    #     print("update_critic")
    #     if self._count == 0:
    #         raise RuntimeError("You should at least run one forward iteration")
    #     # update weights
    #     # We use a small regularization term to keep the output
    #     # of the linear layer in the range of the sigmoid approximation
    #     self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
    #     self.bias -= self._delta_b * (1 / self._count)
    #     # reset gradient accumulators and iterations count
    #     self._delta_w = 0
    #     self._delta_b = 0
    #     self._count = 0

    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])

    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()

    # def encrypt(self, context):
    #     self.weight = ts.ckks_vector(context, self.weight)
    #     self.bias = ts.ckks_vector(context, self.bias)
    #
    # def decrypt(self):
    #     self.weight = self.weight.decrypt()
    #     self.bias = self.bias.decrypt()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

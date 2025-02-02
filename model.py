import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# 用的
# class GaussianPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(GaussianPolicy, self).__init__()
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#
#         self.mean_linear = nn.Linear(hidden_dim, num_actions)
#         self.log_std_linear = nn.Linear(hidden_dim, num_actions)
#
#         print(num_inputs, hidden_dim,num_actions)
#
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):               # 256,17
#         x = F.relu(self.linear1(state))     # 256,256
#         x = F.relu(self.linear2(x))         # 256,256
#         mean = self.mean_linear(x)          # 256,6
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std
#
#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean
#
#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(GaussianPolicy, self).to(device)

class gradients_clip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight_sum,ann_out):


        z = (weight_sum + ann_out)/(1 + weight_sum)

        valid_mask = ann_out.lt(1.) * ann_out.gt(-weight_sum)

        ctx.save_for_backward(weight_sum,ann_out,valid_mask)

        return z

    @staticmethod
    def backward(ctx, grad_input):
        bigger_clip = 11

        weight_sum, ann_out, valid_mask = ctx.saved_tensors

        grad_weight_sum = (1 - ann_out)/((1+weight_sum)**2)     # 激活值关于两项的导数
        grad_ann_out = 1 / (1 + weight_sum)

        grad_weight_sum[grad_weight_sum > bigger_clip] = bigger_clip
        grad_weight_sum[grad_weight_sum < -bigger_clip] = -bigger_clip

        grad_ann_out[grad_ann_out > bigger_clip] = bigger_clip
        grad_ann_out[grad_ann_out < -bigger_clip] = -bigger_clip

        grad_weight_sum = grad_weight_sum * valid_mask
        grad_ann_out = grad_ann_out * valid_mask

        return grad_input*grad_weight_sum, grad_input*grad_ann_out
grad_clip = gradients_clip.apply

class piecewise_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pre_acti, w_sum):
        correct_maximum = 1.

        input[pre_acti < -w_sum] = 0.
        input[pre_acti >= correct_maximum] = correct_maximum

        out = input
        if len(torch.where(input<0)[0]) != 0:
            print('有小于0的激活值')
        if len(torch.where(input>correct_maximum)[0]) != 0:
            print('有大于1的激活值')

        valid_mask = pre_acti.lt(correct_maximum) * pre_acti.gt(-w_sum)

        ctx.save_for_backward(valid_mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        dE = grad_output * mask.float()
        return dE, None, None, None, None
class SNNactivation(nn.Module):
    def __init__(self):
        super(SNNactivation, self).__init__()

    def forward(self, input, pre_acti, w_sum):
        return piecewise_function.apply(input, pre_acti, w_sum)
class MLPblock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MLPblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        valid_inputs = (inputs > 0).float().clone().detach()
        weight_sum = self.linear(valid_inputs)

        linear_out = self.linear(inputs)
        outputs = grad_clip(weight_sum,linear_out)

        return outputs, linear_out, weight_sum

class Finalblock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Finalblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')

        self.W_sum = None

    def forward(self, inputs):
        valid_inputs = (inputs > 0).detach().float()
        weight_sum = self.linear(valid_inputs)

        linear_out = self.linear(inputs)

        self.W_sum = weight_sum

        return linear_out

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = MLPblock(hidden_dim, hidden_dim)
        self.linear2_lif = SNNactivation()

        self.mean_linear = Finalblock(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):  # 256,17
        x = F.relu(self.linear1(state))             # 256,256
        out, pre_acti, w_sum = self.linear2(x)
        x = self.linear2_lif(out,pre_acti,w_sum)    # 256,256
        mean = self.mean_linear(x)                  # 256,6
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

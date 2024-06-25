import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EGARCH(nn.Module):
    def __init__(self, p=1, q=1):
        super(EGARCH, self).__init__()
        self.omega = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, returns):
        T = returns.size(0)
        log_sigma2 = torch.zeros(T)
        sigma2 = torch.zeros(T)
        sigma2[0] = returns.var()
        log_sigma2[0] = torch.log(sigma2[0])
        
        for t in range(1, T):
            g = self.theta * (returns[t-1] / torch.sqrt(sigma2[t-1])) + \
                self.gamma * (torch.abs(returns[t-1] / torch.sqrt(sigma2[t-1])) - torch.sqrt(2/np.pi))
            log_sigma2[t] = self.omega + self.beta * log_sigma2[t-1] + self.alpha * g
            sigma2[t] = torch.exp(log_sigma2[t])
        
        return sigma2
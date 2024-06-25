import torch
import torch.nn as nn

class likelihood_loss(nn.Module):
    def __init__(self, beta, metric='BCE'):
        super(likelihood_loss, self).__init__()
        self.metric = metric
        self.beta = beta
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, rec, real, kl_div):
        """calculates likelihood loss between input and its reconstruction

        Args:
            r (tensor): reconstructed data
            x (tensor): input data

        Returns:
            tensor: likelihood loss between reconstructed and input data
        """
        
        r = rec.view(rec.size()[0], -1)
        x = real.view(real.size()[0], -1)
        
        if self.metric == 'BCE':
            likelihood = -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * (torch.log(1 - r + 1e-8) * x), dim=-1)
        elif self.metric == 'MSE':
            likelihood = -torch.sum(self.mse_loss(x, r), dim=-1)

        elbo = likelihood - self.beta * kl_div
        loss = -torch.mean(elbo)
        
        # metric
        rec = torch.mean(-likelihood)
        kl = torch.mean(kl_div)

        return loss, rec, kl, likelihood, elbo
    
def likelihood_loss_func(r, x, beta, kl_div, metric='BCE'):
    """calculates likelihood loss between input and its reconstruction

    Args:
        r (tensor): reconstructed data
        x (tensor): input data

    Returns:
        tensor: likelihood loss between reconstructed and input data
    """
    r = r.view(r.size()[0], -1)
    x = x.view(x.size()[0], -1)

    if metric == 'BCE':
        likelihood = -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
    elif metric == 'MSE':
        mse_loss = torch.nn.MSELoss(reduction='none')
        likelihood = torch.sum(mse_loss(x, r), dim=-1)

    elbo = -likelihood - beta * kl_div
    loss = -torch.mean(elbo)
    return loss
import torch
import torch.nn as nn

from einops import rearrange
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_model, d_model) # D or BxD or BxBxD
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, Q, K, V, mask = None):

        Q = self.fc_q(Q) # B C L D
        K = self.fc_k(K) # B C L D
        V = self.fc_v(V) # B C L D

        Q = rearrange(Q, 'B C L (H D) -> B H C L D', H = self.n_heads, C = 2)
        K = rearrange(K, 'B C L (H D) -> B H C L D', H = self.n_heads, C = 2)
        V = rearrange(V, 'B C L (H D) -> B H C L D', H = self.n_heads, C = 2)

        attention_score = Q @ K.transpose(-2,-1)/self.scale # B H L L

        if mask is not None:
            attention_score[mask] = -1e10

        attention_weights = torch.softmax(attention_score, dim=-1) # B H C L L

        attention = attention_weights @ V # BHLD

        x = rearrange(attention, 'B H C L D -> B C L (H D)') # BHLD -> BLD
        x = self.fc_o(x) # B L D

        return x, attention_weights
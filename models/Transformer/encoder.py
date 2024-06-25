import torch
import torch.nn as nn
from feedforward import FeedForward
from mha import MHA
import numpy as np

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):

        residual, atten_enc = self.self_atten(x, x, x, enc_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_enc

class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model)).to("cuda")
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.pos = torch.arange(max_len).to("cuda")
        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])
    
    def forward(self, src, mask, atten_map_save = True):
        # src.shape = B L C D
        x = self.scale*self.input_embedding(src) + self.pos_embedding(self.pos)[:src.shape[2],:] # 브로드캐스팅 규칙에 따라 Length, Dimetion 형태를 더해도 됨

        atten_encs = torch.tensor([]).to("cuda")
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs , atten_enc[0].unsqueeze(0)], dim=0)

        return x, atten_encs
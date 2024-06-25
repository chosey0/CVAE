import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.input_embedding = nn.Linear(vocab_size, d_model)#nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        self.decoder = Decoder(self.input_embedding, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        
        self.n_heads = n_heads

        # for m in self.modules():
        #     if hasattr(m,'weight') and m.weight.dim() > 1: # layer norm에 대해선 initial 안하겠다는 뜻
        #         nn.init.kaiming_uniform_(m.weight) # Kaiming의 분산은 2/Nin

        for m in self.modules():
            if hasattr(m,'weight') and m.weight.dim() > 1: # 인풋 임베딩은 그대로 쓰기 위함
                nn.init.xavier_uniform_(m.weight) # xavier의 분산은 2/(Nin+Nout) 즉, 분산이 더 작다. => 그래서 sigmoid/tanh에 적합한 것! (vanishing gradient 막기 위해)

    def make_enc_mask(self, src): # src.shape = Batch, Length or Batch, Length, Feature

        enc_mask = (src == -99).unsqueeze(1).unsqueeze(2)
        enc_mask = enc_mask.expand(src.shape[0], self.n_heads, src.shape[1], src.shape[2], src.shape[2], src.shape[3]).squeeze(-1)

        return enc_mask

    def make_dec_mask(self, trg):

        trg_pad_mask = (trg == -99).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[2], trg.shape[2], trg.shape[3]).squeeze(-1)

        trg_future_mask = (torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[2], trg.shape[2], trg.shape[3])) == 0).squeeze(-1)
        trg_future_mask = trg_future_mask.to("cuda")

        dec_mask = trg_pad_mask | trg_future_mask
        return dec_mask

    def make_enc_dec_mask(self, src, trg):

        enc_dec_mask = (src == -99).unsqueeze(1).unsqueeze(2)

        enc_dec_mask = enc_dec_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[2], src.shape[2], trg.shape[3]).squeeze(-1)

        """ src pad mask
        F F T T
        F F T T
        F F T T
        F F T T
        F F T T
        """
        return enc_dec_mask

    def forward(self, src, trg):


        enc_out, atten_encs = self.encoder(src*(src != -99).float(), None)
        out, atten_decs, atten_enc_decs = self.decoder(trg*(trg != -99).float(), enc_out, None, None)

        return out, enc_out, atten_encs, atten_decs, atten_enc_decs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Feed_Forward_block(nn.Module):
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class last_query_model(nn.Module):
    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_in, seq_len, use_lstm=True):
        super().__init__()
        self.seq_len = seq_len
        # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        self.embd_cat = nn.Embedding(total_cat, embedding_dim=dim_model)
        self.embd_in = nn.Embedding(
            total_in, embedding_dim=dim_model)  # positional embedding

        # multihead attention    ## todo add dropout, LayerNORM
        self.multi_en = nn.MultiheadAttention(
            embed_dim=dim_model, num_heads=heads_en, dropout=0.1)
        # feedforward block     ## todo dropout, LayerNorm
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=dim_model,
                                hidden_size=dim_model, num_layers=1)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, in_ex, in_cat, in_in, labels, first_block=True):
        first_block = True
        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_ex = nn.Dropout(0.1)(in_ex)

            in_cat = self.embd_cat(in_cat)
            in_cat = nn.Dropout(0.1)(in_cat)

            in_in = self.embd_in(in_in)
            in_in = nn.Dropout(0.1)(in_in)

            out = in_ex + in_cat + in_in
        else:
            out = in_ex

        # B,L,D  torch.Size([32, 100, 512]) => L,B,D torch.Size([100, 32, 512])
        out = out.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out

        # out의 마지막 항목(Last query)만 가져와서 multi-head attention에 입력으로 사용
        out, attn_wt = self.multi_en(out[-1:, :, :], out, out)         # Q,K,V
        out = out + skip_out

        # LSTM
        if self.use_lstm:
            # seq_len, batch, input_size
            out, _ = self.lstm(out)
            # out = out[-1:, :, :]
            # torch.Size([1, 32, 512])
        # feed forward
        # L,B,D torch.Size([1, 32, 512]) => B,L,D  torch.Size([32, 1, 512])
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        out = self.out(out)

        return out.squeeze(-1), 0

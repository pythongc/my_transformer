# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/13 10:09
@Auth ： gc
"""
import torch
import torch.nn as nn
from einops import rearrange


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.d_model = (embed_size // self.num_heads)
        # self.all_hide_size = self.d_model*self.num_heads
        assert embed_size % self.num_heads == 0
        self.w_q = nn.Linear(self.embed_size, self.embed_size)
        self.w_k = nn.Linear(self.embed_size, self.embed_size)
        self.w_v = nn.Linear(self.embed_size, self.embed_size)
        self.fc = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        # batch_size, l = q.size(0), q.size(1)
        # q = q.view(batch_size, l, self.num_heads, self.d_model)
        # k = k.view(batch_size, l, self.num_heads, self.d_model)
        # v = v.view(batch_size, l, self.num_heads, self.d_model)
        #
        # q = self.w_q(q)
        # k = self.w_k(k)
        # v = self.w_v(v)
        #
        # energy = torch.einsum("nqhd, nkhd -> nhqk", (q, k))
        # if mask:
        #     torch.masked_fill_(energy, mask, -1e9)
        # attention_scores = nn.functional.softmax(energy*self.d_model**(-0.5), dim=-1)
        # attention = torch.einsum("nhql, nlhd -> nqhd", (attention_scores, v)).reshape(batch_size, l, self.num_heads*self.all_hide_size)
        # print(attention.shape)
        # output = self.fc(attention)
        # q [bs, seq_len, embed_size]
        batch_size, l = q.size(0), q.size(1)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q = rearrange(q, "n l (h d) -> n l h d", h=self.num_heads)
        k = rearrange(k, "n l (h d) -> n l h d", h=self.num_heads)
        v = rearrange(v, "n l (h d) -> n l h d", h=self.num_heads)
        energy = torch.einsum("nqhd, nkhd -> nhqk", (q, k))
        if mask:
            torch.masked_fill_(energy, mask, -1e9)
        attention_scores = nn.functional.softmax(energy*self.d_model**(-0.5), dim=-1)
        attention = torch.einsum("nhql, nlhd -> nqhd", (attention_scores, v)).reshape(batch_size, l, self.embed_size)
        output = self.fc(attention)
        return output

    def transpose_for_scores(self, x):
        batch_size, l = x.size(0), x.size(1)
        return x.view(batch_size, l, self.num_heads, self.embed_size // self.num_heads)


# 位置编码
class PositionEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        position = torch.arange(0, self.max_len).float()
        div_term = 1/torch.pow(torch.tensor(10000, dtype=torch.float),
                               torch.arange(0, self.embed_size).float() / self.embed_size).float()
        self.pe = torch.einsum("i,d->id", [position, div_term])
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.sin(self.pe[:, 0::2])

    def forward(self, x):
        # x.shape: n,sl,embed_size
        len_x = x.shape[1]
        return self.pe[:len_x] + x


# 归一化加残差层
class AddNorm(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_size)

    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


# 位置前馈层
class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_size, ffn_dim, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(nn.functional.relu(self.fc1(x))))


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ffn_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # self.embedding = PositionEmbedding(embed_size)
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFeedForward(embed_size, ffn_dim, dropout)
        self.norm2 = AddNorm(embed_size, dropout)

    def forward(self, x, mask=None):
        # x = self.embedding(x)
        x = self.norm1(x, self.self_attention(x, x, x, mask))
        return self.norm2(x, self.ffn(x))


# 编码器
class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, ffn_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = PositionEmbedding(embed_size)
        self.layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == '__main__':
    # attention = MultiHeadAttention(512, 8)
    # x = torch.randn(2, 10, 512)
    # q, k, v = x, x, x
    # print(attention(q, k, v).shape)
    #
    # position = PositionEmbedding(512)
    # print(position(x).shape)
    x = torch.randn(32, 512, 768)

    encoder = Encoder(embed_size = 768, num_layers = 12, num_heads = 12, ffn_dim = 3078)
    print(encoder.forward(x).shape)
    for name, param in encoder.named_children():
        print(name, param)

    # from transformers import BertModel, AutoConfig
    #
    # cinfig = AutoConfig.from_pretrained('config.json')
    # model = BertModel(cinfig)
    # for name, param in model.named_children():
    #     print(name, param)
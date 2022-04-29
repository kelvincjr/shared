import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
import copy


def make_decoder(head_num, d_model, attn_layers, attn_dropout=0.1):
    attn = MultiHeadedAttention(head_num, d_model)
    ff = PositionwiseFeedForward(d_model, d_model, attn_dropout)
    decoder = Decoder(DecoderLayer(d_model, attn, ff, attn_dropout), attn_layers)
    return decoder


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, src_attn, feed_forward, dropout):
        super().__init__()
        self.d_model = d_model
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, memory, src_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0, f'd_model={d_model}  heads={h}'
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (bsz, q_len, dim)
        :param key: (bsz, seq_len, dim)
        :param value: (bsz, seq_len, dim)
        :param mask: (bsz, q_len, seq_len)   等于0的位置会被遮罩
        :return:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # query ~ (bsz, heads, q_len, head_dim)
        # key ~ (bsz, heads, seq_len, head_dim)
        # value ~ (bsz, heads, seq_len, head_dim)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # x~(bsz, heads, q_len, head_dim)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous()  # x~(bsz, q_len, heads, head_dim)
        x = x.view(nbatches, -1, self.h * self.d_k)  # x~(bsz, q_len, dim)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        MLP: 输入输出格式不变
        """
        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)
        return x


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention
    :param query: (bsz, heads, q_len, dim)
    :param key: (bsz, heads, seq_len, dim)
    :param value: (bsz, heads, seq_len, dim)
    :param mask: (bsz, 1, 1, seq_len) or (bsz, 1, q_len, seq_len)  等于0的位置会被遮罩
    :param dropout:
    :return: value~(bsz, heads, q_len, dim), p_attn~(bsz, heads, q_len, seq_len)
    """
    d_k = query.size(-1)
    key = key.transpose(-2, -1)  # key~(bsz,heads,dim,seq_len)
    scores = torch.matmul(query, key) / math.sqrt(d_k)  # scores~(bsz,heads,q_len,seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # mask == 0 的位置才mask为很小的值
    p_attn = F.softmax(scores, dim=-1)  # p_attn~(bsz,heads,q_len,seq_len)
    if dropout is not None:
        p_attn = dropout(p_attn)
    value = torch.matmul(p_attn, value)  # value~(bsz,heads,q_len,dim)
    return value, p_attn


class TransformerDecoder(nn.Module):
    def __init__(self, N, d_model, d_ff, h, dropout):
        super().__init__()
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attention_layer = Decoder(DecoderLayer(d_model, attn, ff, dropout), N)

    def forward(self, input_q, input_k, att_m):
        attention_output = self.attention_layer(input_q, input_k, att_m)
        # (batch, num_query, hidden)
        return attention_output


def test_decoder():
    head_num = 2
    dim = 10
    dropout = 0.1
    decoder_layers = 2
    decoder = make_decoder(head_num, dim, decoder_layers, attn_dropout=dropout)
    bsz = 1
    seq_len = 6
    q_len = 4
    query = torch.rand(bsz, q_len, dim)
    value = torch.rand(bsz, seq_len, dim)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    y = decoder(query, value, mask)  # ~ (bsz, q_len, dim)
    print(y)


if __name__ == '__main__':
    test_decoder()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bilstm_attention_crf
# @Author   : 研哥哥
# @Time     : 2020/6/22 9:55

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base.base_model import BaseModel
from base.layers import MultiHeadAttention
from config.config import DEVICE
from utils.log import logger
from utils.build_word2vec_weights import load_word2vec


# 80%
class BiLstmAttentionCrf(BaseModel):
    def __init__(self, args):
        super(BiLstmAttentionCrf, self).__init__(args)
        self.args = args
        self.use_dae = args.use_dae
        self.word_vocab = args.word_vocab
        self.tag_vocab = args.tag_vocab
        self.vocab_size = args.vocab_size
        self.tag_num = args.tag_num
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.drop = nn.Dropout(self.dropout)
        self.vector_path = args.vector_path
        self.use_attn = args.use_attn
        self.attn_dropout = args.attn_dropout
        self.key_dim = args.key_dim
        self.val_dim = args.val_dim
        self.num_heads = args.num_heads
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        if args.use_vectors:
            logger.info('Loading word vectors from {}...'.format(self.vector_path))
            embed_weights = load_word2vec(self.vector_path, self.word_vocab, embedding_dim=self.embedding_dim)
            logger.info('Finished load word vectors')
            self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=False).to(DEVICE)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            bidirectional=self.bidirectional, num_layers=self.num_layers,
                            dropout=self.dropout).to(DEVICE)
        self.multi_head_attention = MultiHeadAttention(self.key_dim, self.val_dim, self.hidden_dim, self.num_heads,
                                                       self.attn_dropout).to(DEVICE)
        self.lm_decoder = nn.Linear(self.hidden_dim, self.vocab_size).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(DEVICE)
        self.crf_layer = CRF(self.tag_num).to(DEVICE)
        # self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        crf_loss = -self.crf_layer(emissions, y, mask=mask) / y.size(1)
        if self.use_dae:
            src_encoding = self.encode(x, sent_lengths)
            lm_output = self.decode_lm(src_encoding)
            lm_loss = self.criterion(lm_output.view(-1, self.vocab_size), x.view(-1))
            return crf_loss + lm_loss
        else:
            return crf_loss

    def forward(self, x, sent_lengths):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crf_layer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(DEVICE)).to(DEVICE)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE))
        if self.use_attn:
            output = lstm_out.permute(1, 0, 2)
            attn_out, _ = self.multi_head_attention(output, output, output, None)
            # attn_out: [batch_size, seq_len, dim]
            attn_out = attn_out.permute(1, 0, 2)
            y = self.hidden2label(attn_out.to(DEVICE))
        else:
            y = self.hidden2label(lstm_out.to(DEVICE))  # emissions要求:[seq_len, batch_size, num_tags]
        return y.to(DEVICE)

    def encode(self, source, length):
        embed = self.embedding(source)
        packed_src_embed = pack_padded_sequence(embed, length)
        _, hidden = self.lstm(packed_src_embed)
        embed = self.drop(self.embedding(source))
        packed_src_embed = pack_padded_sequence(embed, length)
        lstm_output, _ = self.lstm(packed_src_embed, hidden)
        lstm_output = pad_packed_sequence(lstm_output)
        lstm_output = self.drop(lstm_output[0])
        return lstm_output

    def decode_lm(self, src_encoding):
        decoded = self.lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))
        return lm_output
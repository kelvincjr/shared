#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bert_bilstm_crf
# @Author   : 研哥哥
# @Time     : 2020/6/22 17:17

import os
import torch
import torch.nn as nn
from torchcrf import CRF as _CRF
from sklearn.metrics import classification_report
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from base.base_config import BaseConfig
from config.config_bert_crf import DEVICE
from base.crf import CRF
from utils.log import logger


class Config(BaseConfig):
    """
    模型的各种参数及配置
    """

    def __init__(self, cus_config, tag_vocab, mask_vocab, **kwargs):
        super(Config, self).__init__()
        for k, v in cus_config.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.mask_vocab = mask_vocab
        self.tag_vocab = tag_vocab
        self.tag_num = len(self.tag_vocab)


class BertBiLstmCRF(BertPreTrainedModel):
    def __init__(self, bert_config, num_tag, my_config):
        super(BertBiLstmCRF, self).__init__(bert_config)
        self.hidden_dim = bert_config.hidden_size
        self.bert = BertModel(bert_config)
        self.lstm_hid_dim = 512
        self.use_lstm = my_config.use_lstm
        self.num_layers = my_config.num_layers
        self.lstm = nn.LSTM(self.hidden_dim, self.lstm_hid_dim // 2, bidirectional=True, num_layers=self.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if self.use_lstm:
            self.classifier = nn.Linear(self.lstm_hid_dim, num_tag)
        else:
            self.classifier = nn.Linear(bert_config.hidden_size, num_tag)
        self.apply(self.init_bert_weights)
        self.crf = CRF(num_tag)
        self._crf = _CRF(num_tag)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.rand(self.num_layers * 2, batch_size, self.lstm_hid_dim // 2).to(DEVICE)
        c0 = torch.rand(self.num_layers * 2, batch_size, self.lstm_hid_dim // 2).to(DEVICE)
        return h0, c0

    def forward(self, input_ids, token_type_ids, attention_mask, label_id=None, output_all_encoded_layers=False):
        batch_size = input_ids.size(0)
        output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                              output_all_encoded_layers=output_all_encoded_layers)
        # lstm layer
        if self.use_lstm:
            hidden = self.init_hidden(batch_size)
            output, _ = self.lstm(output.to(DEVICE), hidden)
        output = self.dropout(output.to(DEVICE))
        output = self.classifier(output.to(DEVICE))
        return output.to(DEVICE)

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask, x):
        x = x.transpose(0, 1)
        mask_crf = torch.ne(x, 0)
        pri_mask_crf = mask_crf.to('cpu').numpy().tolist()
        # print('mask_crf: ', pri_mask_crf)
        output_mask = torch.transpose(output_mask, 0, 1)
        output_mask = torch.ne(output_mask, 0)
        pri_output_mask = output_mask.to('cpu').numpy().tolist()
        # print('output_mask: ', pri_output_mask)
        bert_encode = bert_encode.transpose(0, 1)
        return self._crf.decode(bert_encode, mask=mask_crf)
        # predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        # predicts = predicts.view(1, -1).squeeze()
        # predicts = predicts[predicts != -1]
        # return predicts

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred, digits=5)
        print('\n\nclassify_report:\n', classify_report)
        return classify_report

    def load(self, path=None, map_location=torch.device('cpu')):
        """
        load the model
        :param map_location:
        :param path:
        :return:
        """
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location))
        logger.info('loadding model from {}'.format(model_path))

    def save(self, path=None):
        """
        save the model
        :param path:
        :return:
        """
        assert path is not None
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))

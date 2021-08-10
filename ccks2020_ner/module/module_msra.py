#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module_msra
# @Author   : 研哥哥
# @Time     : 2020/7/15 21:26

import codecs
import os
import sys
import json
import warnings
import numpy
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from model.bilstm_crf import BiLstmCrf
from model.bilstm_attention_crf import BiLstmAttentionCrf
from model.transformer_bilstm_crf import TransformerEncoderModel, ModelConfig
from model.bert_bilstm_crf import BertBiLstmCRF
from config.config import DEVICE
from config.config_msra import DEFAULT_CONFIG
from utils.tool import tool
from utils.log import logger
from base.base_module import BaseModule

warnings.filterwarnings('ignore')


# 1. Bi-LSTM + CRF : %
# 2. Bi-LSTM + Attention + CRF : %
# 3. TransformerEncoder + Bi-LSTM + CRF : %
# 4. BERT + Bi-LSTM + CRF : %

class CCKS2020_NER(BaseModule):
    def __init__(self):
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None
        self.model_name = DEFAULT_CONFIG['model_name']

    def train(self):
        logger.info('Loading data ...')
        self.train_data = tool.load_msra_data(DEFAULT_CONFIG['train_path'])
        self.dev_data = tool.load_msra_data(DEFAULT_CONFIG['dev_path'])
        logger.info('Finished load data')
        logger.info('Building vocab ...')
        self.word_vocab = tool.get_text_vocab(self.train_data, self.dev_data)
        self.tag_vocab = tool.get_tag_vocab(self.train_data, self.dev_data)
        logger.info('Finished build vocab')
        logger.info('Building iterator ...')
        self.train_iter = tool.get_msra_iterator(self.train_data, batch_size=DEFAULT_CONFIG['batch_size'])
        self.dev_iter = tool.get_msra_iterator(self.dev_data, batch_size=DEFAULT_CONFIG['batch_size'])
        logger.info('Finished build iterator')
        config = ModelConfig(cus_config=DEFAULT_CONFIG, word_vocab=self.word_vocab, tag_vocab=self.tag_vocab)
        if self.model_name == 'bilstm_crf':
            self.model = BiLstmCrf(config).to(DEVICE)
        elif self.model_name == 'bilstm_attention_crf':
            self.model = BiLstmAttentionCrf(config).to(DEVICE)
        elif self.model_name == 'transformer_bilstm_crf':
            self.model = TransformerEncoderModel(config).to(DEVICE)
        else:
            logger.error('Error: The model name : {} could not be found...'.format(self.model_name))
            sys.exit()
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        if not os.path.exists(config.save_model_path):
            os.mkdir(config.save_model_path)
        if not os.path.exists(config.result_path):
            os.mkdir(config.result_path)
        p_max = 0
        r_max = 0
        f1_max = 0
        best_epoch = -1
        info = {'epoch': [], 'p': [], 'r': [], 'f1': [], 'loss': []}
        logger.info('Beginning train ...')
        for epoch in range(config.epoch):
            self.model.train()
            acc_loss = 0
            for item in tqdm(self.train_iter):
                optimizer.zero_grad()
                text = item.text[0]
                text_len = item.text[1]
                tag = item.tag
                # loss = (-self.model.loss(text, text_len, tag)) / item.tag.size(1)
                loss = self.model.loss(text, text_len, tag)
                acc_loss += loss.view(-1).cpu().data.tolist()[0]
                loss.backward()
                optimizer.step()
            p, r, f1, stat_info, prf_dict = self.evaluate()
            info['epoch'].append(epoch + 1)
            info['p'].append(p)
            info['r'].append(r)
            info['f1'].append(f1)
            info['loss'].append(acc_loss)
            logger.info('epoch: {} loss: {} average: p: {} r: {} f1: {}'.format(epoch, acc_loss, p, r, f1))
            if f1 > f1_max:
                p_max = p
                r_max = r
                f1_max = f1
                best_epoch = epoch
                logger.info('save best model...')
                torch.save(self.model.state_dict(),
                           config.save_path + '{}/msra_model_{}.pkl'.format(config.model_name, config.experiment_name))
                logger.info(
                    'best model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                               best_epoch))
        if config.epoch != 0:
            with codecs.open(DEFAULT_CONFIG['pred_info_path'], 'a', encoding='utf-8') as f:
                f.write(stat_info + '\nbest model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'
                        .format(p_max, r_max, f1_max, best_epoch) + '\n')
            tool.record_info2graph(info=info, experiment_name=DEFAULT_CONFIG['experiment_name'])
            print(stat_info)
            logger.info('Finished train')
            logger.info('best model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                                   best_epoch))

    def evaluate(self):
        self.model.eval()
        tag_true_all = []
        tag_pred_all = []
        entities_total = {'PERSON': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          'LOCATION': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          'ORGANIZATION': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0}}
        with codecs.open(DEFAULT_CONFIG['pred_info_path'], 'w', encoding='utf-8') as f:
            f.write('我要O泡果奶哦哦哦~~~' + '\n')
        for item in tqdm(self.dev_iter):
            text = item.text[0]
            text_len = item.text[1]
            tag = torch.transpose(item.tag, 0, 1)
            tag = tag.to('cpu').numpy().tolist()
            result = self.model(text, text_len)
            text_transpose = torch.transpose(text, 0, 1)
            text_list = []
            for text_i in text_transpose:
                text_list.append(''.join([self.word_vocab.itos[k] for k in text_i]))
            assert len(tag) == len(result), 'tag_len: {} != result_len: {}'.format(len(tag), len(result))
            for i, result_list in zip(range(text.size(1)), result):
                tag_list = tag[i][:text_len[i]]
                assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(len(tag_list),
                                                                                                   len(result_list))
                tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                tag_true_all.extend(tag_true)
                tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                tag_pred_all.extend(tag_pred)

                entities = self._evaluate(text=text_list[i], tag_true=tag_true, tag_pred=tag_pred)
                assert len(entities_total) == len(entities), 'entities_total: {} != entities: {}'.format(
                    len(entities_total), len(entities))
                for entity in entities_total:
                    entities_total[entity]['TP'] += entities[entity]['TP']
                    entities_total[entity]['S'] += entities[entity]['S']
                    entities_total[entity]['G'] += entities[entity]['G']
        TP = 0
        S = 0
        G = 0
        print('\n-------------------------------------------------------------------------------------')
        print('label_type\t\tp\t\t\tr\t\t\tf1\t\t\tTP\t\t\tpred_num\tlabel_num')
        str = '{0:<11}\t{1:<10.3f}\t{2:<10.3f}\t{3:<10.3f}\t{4:<10}\t{5:<10}\t{6:<10}'
        for entity in entities_total:
            entities_total[entity]['p'] = entities_total[entity]['TP'] / entities_total[entity]['S'] \
                if entities_total[entity]['S'] != 0 else 0
            entities_total[entity]['r'] = entities_total[entity]['TP'] / entities_total[entity]['G'] \
                if entities_total[entity]['G'] != 0 else 0
            entities_total[entity]['f1'] = 2 * entities_total[entity]['p'] * entities_total[entity]['r'] / \
                                           (entities_total[entity]['p'] + entities_total[entity]['r']) \
                if entities_total[entity]['p'] + entities_total[entity]['r'] != 0 else 0
            print(str.format(entity, entities_total[entity]['p'], entities_total[entity]['r'],
                             entities_total[entity]['f1'], entities_total[entity]['TP'],
                             entities_total[entity]['S'], entities_total[entity]['G']), chr(12288))
            TP += entities_total[entity]['TP']
            S += entities_total[entity]['S']
            G += entities_total[entity]['G']
        p = TP / S if S != 0 else 0
        r = TP / G if G != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        print(str.format('average\t\t', p, r, f1, TP, S, G), chr(12288))
        print('-------------------------------------------------------------------------------------')
        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        stat_info = classification_report(tag_true_all, tag_pred_all, labels=labels)
        prf_dict = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=True)
        return p, r, f1, stat_info, prf_dict

    def _evaluate(self, text=None, tag_true=None, tag_pred=None):
        """
        先对true进行还原成 [{}] 再对pred进行还原成 [{}]
        :param tag_true: list[]
        :param tag_pred: list[]
        :return:
        """
        true_list = self._build_list_dict(_len=len(tag_true), _list=tag_true)
        pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
        entities = {'PERSON': {'TP': 0, 'S': 0, 'G': 0},
                    'LOCATION': {'TP': 0, 'S': 0, 'G': 0},
                    'ORGANIZATION': {'TP': 0, 'S': 0, 'G': 0}}
        for true in true_list:
            label_type = true['label_type']
            entities[label_type]['G'] += 1
        for pred in pred_list:
            start_pos = pred['start_pos']
            end_pos = pred['end_pos']
            label_type = pred['label_type']
            entities[label_type]['S'] += 1
            for true in true_list:
                if label_type == true['label_type'] and start_pos == true['start_pos'] and end_pos == true['end_pos']:
                    entities[label_type]['TP'] += 1
        tool.record_pred_info(text=text, true_list=true_list, pred_list=pred_list,
                              path=DEFAULT_CONFIG['pred_info_path'])
        return entities

    def _build_list_dict(self, _len, _list):
        build_list = []
        for index, tag in zip(range(_len), _list):
            start_pos = index
            end_pos = index + 1
            label_type = tag[2:]
            if tag[0] == 'B' and end_pos != _len:
                # two !=
                while _list[end_pos][0] == 'I' and _list[end_pos][2:] == label_type and end_pos + 1 != _len:
                    end_pos += 1
                if _list[end_pos][0] == 'E':
                    build_list.append({'start_pos': start_pos,
                                       'end_pos': end_pos + 1,
                                       'label_type': label_type})
            elif tag[0] == 'S':
                build_list.append({'start_pos': start_pos,
                                   'end_pos': end_pos,
                                   'label_type': label_type})
        return build_list

    def predict(self):
        DEFAULT_CONFIG['epoch'] = 0
        self.train()
        self.model.load_state_dict(
            torch.load('../model/save_model/{}/msra_model_{}.pkl'.format(DEFAULT_CONFIG['model_name'],
                                                                         DEFAULT_CONFIG['experiment_name'])))
        logger.info('Beginning eval...')
        self.model.eval()
        self.test_data = tool.read_json(DEFAULT_CONFIG['test_path'])
        with codecs.open(DEFAULT_CONFIG['result_path'], 'w', encoding='utf-8') as f:
            for example in tqdm(self.test_data):
                text = example['originalText']
                # split to text
                # text_list = self.split_text(text)
                # tag_pred_list = []
                # for text in text_list:
                text = torch.tensor(
                    numpy.array([self.word_vocab.stoi[word] for word in text], dtype='int64')).unsqueeze(1).to(
                    DEVICE)
                text_len = torch.tensor(numpy.array([text.size(0)], dtype='int64')).to(DEVICE)
                result_list = self.model(text, text_len)[0]
                tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                # tag_pred_list.append(tag_pred)
                #
                # tag_pred = self.merge_tag(tag_pred_list)
                pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
                pred_dict = {'originalText': example['originalText'], 'entities': pred_list}
                f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
        logger.info('Finished eval')

    def split_text(self, text=None):
        result_list = []
        start = end = split_len = 0
        for ch in text:
            split_len += 1
            end += 1
            if ch == '。' or ch == ',' or ch == '，' or ch == '；':
                result_list.append(text[start: end])
                start = end
                split_len = 0
            elif end == len(text):
                result_list.append(text[start: end])
        return result_list

    def merge_tag(self, tag_pred_list=None):
        result_list = []
        for tag_pred in tag_pred_list:
            result_list.extend(tag_pred)
        return result_list

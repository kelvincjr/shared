#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:41

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
from sklearn.model_selection import train_test_split
from model.bilstm_crf import BiLstmCrf
from model.bilstm_attention_crf import BiLstmAttentionCrf
from model.transformer_bilstm_crf import TransformerEncoderModel, ModelConfig
from model.cnn_transformer_bilstm_crf import CNNTransformerEncoderModel
from model.flat_lattice import FlatLattice, FlatLatticeModelConfig
from config.config import DEVICE, DEFAULT_CONFIG
from utils.tool import tool
from utils.log import logger
from base.base_module import BaseModule

warnings.filterwarnings('ignore')


# 1. Bi-LSTM + CRF : 78%
# 2. Bi-LSTM + Attention + CRF : 80%
# 3. TransformerEncoder + Bi-LSTM + CRF : 82%
# 4. Flat-Lattice: %
# 5. train_test_split 9 1

class CCKS2020_NER(BaseModule):
    def __init__(self):
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.unlabeled_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.bi_gram_vocab = None
        self.lattice_vocab = None
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None
        self.unlabeled_iter = None
        self.model_name = DEFAULT_CONFIG['model_name']
        self.experiment_name = DEFAULT_CONFIG['experiment_name']

    def train(self):
        logger.info('Loading data ...')
        # train_data_json2list, dev_data_json2list = train_test_split(
        #     tool.read_json(DEFAULT_CONFIG['train_path']), test_size=0.05
        # )
        # self.train_data = tool.load_data(train_data_json2list)
        # self.dev_data = tool.load_data(dev_data_json2list)
        self.train_data = tool.load_data(tool.read_json(DEFAULT_CONFIG['train_path']))
        self.dev_data = tool.load_data(tool.read_json(DEFAULT_CONFIG['dev_path']))
        self.test_data = tool.load_unlabel_data(tool.read_json(DEFAULT_CONFIG['test_path']))
        self.unlabeled_data = tool.load_unlabeled_data(tool.read_unlabeled(DEFAULT_CONFIG['unlabeled_path']))
        logger.info('Finished load data')
        logger.info('Building vocab ...')
        if self.model_name == 'flat_lattice':
            self.bi_gram_vocab = tool.get_bi_gram_vocab(self.train_data, self.dev_data)
            self.lattice_vocab = tool.get_lattice_vocab(self.train_data, self.dev_data)
        else:
            self.word_vocab = tool.get_text_vocab(self.train_data, self.dev_data, self.test_data, self.unlabeled_data)
        self.tag_vocab = tool.get_tag_vocab(self.train_data, self.dev_data)
        logger.info('Finished build vocab')
        logger.info('Building iterator ...')
        self.unlabeled_iter = tool.get_iterator(self.unlabeled_data, batch_size=DEFAULT_CONFIG['batch_size'])
        if self.model_name == 'flat_lattice':
            self.train_iter = tool.get_lattice_iterator(self.train_data, batch_size=DEFAULT_CONFIG['batch_size'])
            self.dev_iter = tool.get_lattice_iterator(self.dev_data, batch_size=DEFAULT_CONFIG['batch_size'])
            config = FlatLatticeModelConfig(cus_config=DEFAULT_CONFIG, bi_gram_vocab=self.bi_gram_vocab,
                                            lattice_vocab=self.lattice_vocab, tag_vocab=self.tag_vocab)
        else:
            self.train_iter = tool.get_iterator(self.train_data, batch_size=DEFAULT_CONFIG['batch_size'])
            self.dev_iter = tool.get_iterator(self.dev_data, batch_size=DEFAULT_CONFIG['batch_size'])
            config = ModelConfig(cus_config=DEFAULT_CONFIG, word_vocab=self.word_vocab, tag_vocab=self.tag_vocab)
        logger.info('Finished build iterator')
        if self.model_name == 'bilstm_crf':
            self.model = BiLstmCrf(config).to(DEVICE)
        elif self.model_name == 'bilstm_attention_crf':
            self.model = BiLstmAttentionCrf(config).to(DEVICE)
        elif self.model_name == 'transformer_bilstm_crf':
            self.model = TransformerEncoderModel(config).to(DEVICE)
        elif self.model_name == 'cnn_transformer_bilstm_crf':
            self.model = CNNTransformerEncoderModel(config).to(DEVICE)
        elif self.model_name == 'flat_lattice':
            self.model = FlatLattice(config).to(DEVICE)
        else:
            logger.error('Error: The model name : {} could not be found...'.format(self.model_name))
            sys.exit()
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, patience=10)
        # 以 acc 为例，当 mode 设置为 “max” 时，如果 acc 在给定 patience 内没有提升，则以 factor 的倍率降低 lr。
        # for epoch in range(100):
        #     acc = 10
        #     scheduler.step(acc)
        #     tool.adjust_learning_rate(optimizer=optimizer, epoch=epoch)
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
        # for epoch in range(config.epoch // 5):
        #     self.model.train()
        #     acc_loss = 0
        #     for item in tqdm(self.unlabeled_iter):
        #         optimizer.zero_grad()
        #         text = item.text[0]
        #         text_len = item.text[1]
        #         loss = self.model.dae_loss(text, text_len)
        #         acc_loss += loss.view(-1).cpu().data.tolist()[0]
        #         loss.backward()
        #         optimizer.step()
        #     logger.info('epoch: {} loss: {}'.format(epoch, acc_loss))

        for epoch in range(config.epoch):
            self.model.train()
            total_loss = {'crf_loss': 0, 'dae_loss': 0, 'dice_loss': 0, 'refactor_loss': 0}
            print(optimizer.param_groups[0]['lr'])
            # tool.adjust_learning_rate(optimizer, epoch + 1)
            for item in tqdm(self.train_iter):
                optimizer.zero_grad()
                tag = item.tag
                if self.model_name == 'flat_lattice':
                    bi_gram = item.bi_gram[0]
                    lattice = item.lattice[0]
                    lattice_len = item.lattice[1]
                    loss = self.model.loss(bi_gram, lattice, lattice_len, tag)
                else:
                    text = item.text[0]
                    text_len = item.text[1]
                    loss = self.model.loss(text, text_len, tag)
                refactor_loss = loss['refactor_loss']
                total_loss = self.acc_total_loss(total_loss, loss)
                refactor_loss.backward()
                optimizer.step()
            p, r, f1, stat_info = self.evaluate()
            scheduler.step(f1)
            info['epoch'].append(epoch + 1)
            info['p'].append(p)
            info['r'].append(r)
            info['f1'].append(f1)
            info['loss'].append(total_loss['refactor_loss'])
            logger.info('epoch: {} average: p: {:.4f} r: {:.4f} f1: {:.4f}'.format(epoch + 1, p, r, f1))
            logger.info('crf_loss: {:.2f} dae_loss: {:.2f} dice_loss: {:.2f} refactor_loss: {:.2f}'.format(
                total_loss['crf_loss'], total_loss['dae_loss'], total_loss['dice_loss'], total_loss['refactor_loss']))
            if f1 > f1_max:
                p_max = p
                r_max = r
                f1_max = f1
                best_epoch = epoch + 1
                best_stat_info = stat_info
                logger.info('save best model...')
                torch.save(self.model.state_dict(),
                           config.save_model_path + 'model_{}.pkl'.format(self.experiment_name))
                logger.info(
                    'best model: precision: {:.4f} recall: {:.4f} f1: {:.4f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                               best_epoch))
        if config.epoch != 0:
            with codecs.open(
                    DEFAULT_CONFIG['result_path'] + '{}_pred_info.txt'.format(self.experiment_name), 'a',
                    encoding='utf-8') as f:
                f.write('\n' + best_stat_info + '\nbest model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'
                        .format(p_max, r_max, f1_max, best_epoch) + '\n')
            tool.record_info2graph(info=info, result_path=DEFAULT_CONFIG['result_path'],
                                   experiment_name=self.experiment_name)
            print(stat_info)
            logger.info('Finished train')
            logger.info('best model: precision: {:.5f} recall: {:.5f} f1: {:.5f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                                   best_epoch))

    def evaluate(self):
        self.model.eval()
        tag_true_all = []
        tag_pred_all = []
        entities_total = {'疾病和诊断': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '影像检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '实验室检验': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '手术': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '解剖部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0}}
        with codecs.open(DEFAULT_CONFIG['result_path'] + '{}_pred_info.txt'.format(self.experiment_name), 'w',
                         encoding='utf-8') as f:
            f.write('我要O泡果奶哦哦哦~~~' + '\n')
        for item in tqdm(self.dev_iter):
            tag = torch.transpose(item.tag, 0, 1)
            tag = tag.to('cpu').numpy().tolist()
            if self.model_name == 'flat_lattice':
                bi_gram = item.bi_gram[0]
                lattice = item.lattice[0]
                bi_gram_len = item.bi_gram[1]
                lattice_len = item.lattice[1]
                result = self.model(bi_gram, lattice, lattice_len)
                lattice_transpose = torch.transpose(lattice, 0, 1)
                assert len(tag) == len(result), 'tag_len: {} != result_len: {}'.format(len(tag), len(result))
                for i, result_list in zip(range(bi_gram.size(1)), result):
                    text_i = ''.join([self.lattice_vocab.itos[k] for k in lattice_transpose[i][: bi_gram_len[i]]])
                    tag_list = tag[i][: bi_gram_len[i]]
                    assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(len(tag_list),
                                                                                                       len(result_list))
                    tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                    tag_true_all.extend(tag_true)
                    tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                    tag_pred_all.extend(tag_pred)

                    entities = self._evaluate(text=text_i, tag_true=tag_true, tag_pred=tag_pred)
                    assert len(entities_total) == len(entities), 'entities_total: {} != entities: {}'.format(
                        len(entities_total), len(entities))
                    for entity in entities_total:
                        entities_total[entity]['TP'] += entities[entity]['TP']
                        entities_total[entity]['S'] += entities[entity]['S']
                        entities_total[entity]['G'] += entities[entity]['G']

            else:
                text = item.text[0]
                text_len = item.text[1]
                tag = torch.transpose(item.tag, 0, 1)
                tag = tag.to('cpu').numpy().tolist()
                result = self.model(text, text_len)
                text_transpose = torch.transpose(text, 0, 1)
                assert len(tag) == len(result), 'tag_len: {} != result_len: {}'.format(len(tag), len(result))
                for i, result_list in zip(range(text.size(1)), result):
                    text_i = ''.join([self.word_vocab.itos[k] for k in text_transpose[i][: text_len[i]]])
                    tag_list = tag[i][:text_len[i]]
                    assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(len(tag_list),
                                                                                                       len(result_list))
                    tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                    tag_true_all.extend(tag_true)
                    tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                    tag_pred_all.extend(tag_pred)

                    entities = self._evaluate(text=text_i, tag_true=tag_true, tag_pred=tag_pred)
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
        return p, r, f1, stat_info

    def _evaluate(self, text=None, tag_true=None, tag_pred=None):
        """
        先对true进行还原成 [{}] 再对pred进行还原成 [{}]
        :param tag_true: list[]
        :param tag_pred: list[]
        :return:
        """
        true_list = self._build_list_dict(_len=len(tag_true), _list=tag_true)
        pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
        entities = {'疾病和诊断': {'TP': 0, 'S': 0, 'G': 0},
                    '影像检查': {'TP': 0, 'S': 0, 'G': 0},
                    '实验室检验': {'TP': 0, 'S': 0, 'G': 0},
                    '手术': {'TP': 0, 'S': 0, 'G': 0},
                    '药物': {'TP': 0, 'S': 0, 'G': 0},
                    '解剖部位': {'TP': 0, 'S': 0, 'G': 0}}
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
                              path=DEFAULT_CONFIG['result_path'] + '{}_pred_info.txt'.format(self.experiment_name))
        return entities

    def _build_list_dict(self, _len, _list):
        build_list = []
        tag_dict = {'disease_and_diagnosis': '疾病和诊断',
                    'check': '影像检查',
                    'checkout': '实验室检验',
                    'operation': '手术',
                    'medicine': '药物',
                    'anatomical_site': '解剖部位'}
        if DEFAULT_CONFIG['tag_type'] == 'BIO':
            for index, tag in zip(range(_len), _list):
                if tag[0] == 'B':
                    start_pos = index
                    end_pos = index + 1
                    label_type = tag[2:]
                    while _list[end_pos][0] == 'I' and _list[end_pos][2:] == label_type:
                        end_pos += 1
                    build_list.append({'start_pos': start_pos,
                                       'end_pos': end_pos,
                                       'label_type': tag_dict[label_type]})
        elif DEFAULT_CONFIG['tag_type'] == 'BME_SO':
            for index, tag in zip(range(_len), _list):
                start_pos = index
                end_pos = index + 1
                label_type = tag[2:]
                if tag[0] == 'B' and end_pos != _len:
                    # two !=
                    while _list[end_pos][0] == 'M' and _list[end_pos][2:] == label_type and end_pos + 1 != _len:
                        end_pos += 1
                    if _list[end_pos][0] == 'E':
                        build_list.append({'start_pos': start_pos,
                                           'end_pos': end_pos + 1,
                                           'label_type': tag_dict[label_type]})
                elif tag[0] == 'S':
                    build_list.append({'start_pos': start_pos,
                                       'end_pos': end_pos,
                                       'label_type': tag_dict[label_type]})
        else:
            logger.error('tag_type != BIO and tag_type != BME_SO !')
            sys.exit()
        return build_list

    def predict(self):
        DEFAULT_CONFIG['epoch'] = 0
        self.train()
        self.model.load_state_dict(
            torch.load(DEFAULT_CONFIG['save_model_path'] + 'model_{}.pkl'.format(self.experiment_name)))
        logger.info('Beginning eval...')
        self.model.eval()
        self.test_data = tool.read_json(DEFAULT_CONFIG['test_path'])
        with codecs.open(DEFAULT_CONFIG['result_path'] + 'result_{}.txt'.format(self.experiment_name), 'w',
                         encoding='utf-8') as f:
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

    def ready(self):
        DEFAULT_CONFIG['epoch'] = 0
        self.train()
        self.model.load_state_dict(
            torch.load(DEFAULT_CONFIG['save_model_path'] + 'model_{}.pkl'.format(self.experiment_name)))
        logger.info('Beginning eval...')
        self.model.eval()
        pass

    def pred(self, content: str, type: str) -> dict:
        text = torch.tensor(
            numpy.array([self.word_vocab.stoi[word] for word in content], dtype='int64')).unsqueeze(1).to(
            DEVICE)
        text_len = torch.tensor(numpy.array([text.size(0)], dtype='int64')).to(DEVICE)
        result_list = self.model(text, text_len)[0]
        tag_pred = [self.tag_vocab.itos[k] for k in result_list]
        pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
        i, text_list, label_list = 0, [], []
        for index, pred in enumerate(pred_list):
            if type == 'ST':
                if index in [4, 8]:
                    continue
            elif type == 'MT':
                if index in [4]:
                    continue

            start_pos = pred['start_pos']
            end_pos = pred['end_pos']
            text_o = content[i: start_pos]
            text_label = content[start_pos: end_pos]
            i = end_pos
            if text_o != '':
                text_list.append(text_o)
                label_list.append('O')
            text_list.append(text_label)
            label_list.append(pred['label_type'])

        if i < len(content):
            text_list.append(content[i:])
            label_list.append('O')
        result_dict = {
            'text_list': text_list,
            'label_list': label_list
        }
        return result_dict
        pass

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

    def acc_total_loss(self, total_loss=None, loss=None):
        for loss_name in loss:
            if loss[loss_name] is not None:
                total_loss[loss_name] += loss[loss_name].view(-1).cpu().data.tolist()[0]
        return total_loss

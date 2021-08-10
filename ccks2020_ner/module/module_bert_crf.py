#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module_bert_crf
# @Author   : 研哥哥
# @Time     : 2020/7/29 10:47


import os
import time
import codecs
import sys
import json
import warnings
import numpy
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertAdam
from data.bert_loader import create_batch_iter, init_params
from data.bert_processor import InputExample, convert_examples_to_features, convert_example
from model.bert_bilstm_crf import BertBiLstmCRF, Config
from config.config_bert_crf import DEVICE, DEFAULT_CONFIG
from config.config import DEFAULT_CONFIG as _DEFAULT_CONFIG
from utils.tool import tool
from utils.log import logger
from base.base_module import BaseModule

warnings.filterwarnings('ignore')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class CCKS2020_NER_BERT(BaseModule):
    def __init__(self):
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.unlabeled_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.mask_vocab = None
        self.tokenizer = None
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None
        self.unlabeled_iter = None
        self.model_name = DEFAULT_CONFIG['model_name']

    def train(self):
        # logger.info('Loading data ...')
        # self.train_data = tool.load_data(tool.read_json(_DEFAULT_CONFIG['train_path']))
        # self.test_data = tool.load_unlabel_data(tool.read_json(_DEFAULT_CONFIG['test_path']))
        # self.unlabeled_data = tool.load_unlabeled_data(tool.read_unlabeled(_DEFAULT_CONFIG['unlabeled_path']))
        # logger.info('Finished load data')
        # logger.info('Building vocab ...')
        # self.mask_vocab = tool.get_text_vocab(self.train_data, self.test_data, self.unlabeled_data).stoi
        # self.tag_vocab = tool.get_tag_vocab(self.train_data).stoi
        # logger.info('Finished build vocab')
        processor, tokenizer = init_params(DEFAULT_CONFIG['vocab_path'])
        _, self.tag_vocab = processor.get_label_vocab(path=DEFAULT_CONFIG['train_path'],
                                                      split='\t')  # 获取训练集之后再获取label_vocab
        self.tag_vocab['B-X'] = len(self.tag_vocab)
        self.mask_vocab = processor.get_mask_vocab(path=DEFAULT_CONFIG['vocab_path'])
        # print(self.tag_vocab)
        logger.info('Building iterator ...')
        self.train_iter, num_train_steps = create_batch_iter("train", DEFAULT_CONFIG, processor, tokenizer,
                                                             self.tag_vocab,
                                                             self.mask_vocab)  # 获取训练数据
        self.dev_iter = create_batch_iter("dev", DEFAULT_CONFIG, processor, tokenizer, self.tag_vocab,
                                          self.mask_vocab)  # 获取验证数据
        logger.info('Finished build iterator')
        config = Config(cus_config=DEFAULT_CONFIG, tag_vocab=self.tag_vocab, mask_vocab=self.mask_vocab)
        if self.model_name == 'bert_bilstm_crf':
            self.model = BertBiLstmCRF.from_pretrained(config.bert_path, num_tag=len(self.tag_vocab),
                                                       my_config=config).to(DEVICE)
        else:
            logger.error('Error: The model name : {} could not be found...'.format(self.model_name))
            sys.exit()
        # ---------------------优化器----------------------
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion,
                             t_total=t_total)
        global_step = 0
        # optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        p_max = r_max = f1_max = best_epoch = 0
        info = {'epoch': [], 'p': [], 'r': [], 'f1': [], 'loss': []}
        logger.info('Beginning train ...')
        for epoch in range(config.epoch):
            self.model.train()
            step = acc_loss = 0
            for batch in tqdm(self.train_iter):
                step += 1
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = self.model(input_ids, segment_ids, input_mask)
                train_loss = self.model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
                train_loss.backward()
                acc_loss += train_loss.item()  # 统计loss
                if (step + 1) % DEFAULT_CONFIG['gradient_accumulation_steps'] == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = config.learning_rate * warmup_linear(global_step / t_total, config.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            p, r, f1, stat_info = self.evaluate()
            info['epoch'].append(epoch + 1)
            info['p'].append(p)
            info['r'].append(r)
            info['f1'].append(f1)
            info['loss'].append(acc_loss)
            print(stat_info)
            logger.info('epoch: {} loss: {} average: p: {} r: {} f1: {}'.format(epoch + 1, acc_loss, p, r, f1))
            if f1 > f1_max:
                p_max = p
                r_max = r
                f1_max = f1
                best_epoch = epoch
                best_stat_info = stat_info
                # print(best_stat_info)
                logger.info('save best model...')
                torch.save(self.model.state_dict(),
                           config.save_path + '{}/model_{}.pkl'.format(config.model_name, config.experiment_name))
                logger.info(
                    'best model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                               best_epoch))
        if config.epoch != 0:
            with codecs.open(DEFAULT_CONFIG['pred_info_path'], 'a', encoding='utf-8') as f:
                f.write(best_stat_info + '\nbest model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'
                        .format(p_max, r_max, f1_max, best_epoch) + '\n')
            tool.record_info2graph(info=info, experiment_name=DEFAULT_CONFIG['experiment_name'])
            print(best_stat_info)
            logger.info('Finished train')
            logger.info('best model: precision: {:.3f} recall: {:.3f} f1: {:.3f} epoch: {}'.format(p_max, r_max, f1_max,
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
        with codecs.open(DEFAULT_CONFIG['pred_info_path'], 'w', encoding='utf-8') as f:
            f.write('我要O泡果奶哦哦哦~~~' + '\n')
        for batch in tqdm(self.dev_iter):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = self.model(input_ids, segment_ids, input_mask)
            result = self.model.predict(bert_encode, output_mask, x=input_ids)
            id2tag_vocab = {v: k for k, v in self.tag_vocab.items()}
            # result_list = predicts.cpu().numpy().tolist()
            # tag = torch.transpose(label_ids, 0, 1)
            tag = label_ids.to('cpu').numpy().tolist()
            assert len(tag) == len(result), 'tag_len: {} != result_len: {}'.format(len(tag), len(result))
            for tag_list, result_list in zip(tag, result):
                while -1 in tag_list:
                    tag_list.remove(-1)
                result_list = result_list[1: -1]
                # assert len(tag_list) == len(result_list), 'tag_list_len: {} != result_list_len: {}'.format(
                #     len(tag_list), len(result_list))
                tag_true = [id2tag_vocab[k] for k in tag_list]
                tag_true_all.extend(tag_true)

                tag_pred = [id2tag_vocab[k] for k in result_list]
                tag_pred_all.extend(tag_pred)

                # label_ids = label_ids.view(1, -1)
                # label_ids = label_ids[label_ids != -1]
                # tag_list = label_ids.cpu().numpy().tolist()

                entities = self._evaluate(tag_true=tag_true, tag_pred=tag_pred)
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
        labels = [k for k, v in self.tag_vocab.items()]
        labels.remove('O')
        # stat_info = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=False)
        stat_info = None
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
        # tool.record_pred_info(text=text, true_list=true_list, pred_list=pred_list,
        #                       path=DEFAULT_CONFIG['pred_info_path'])
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
        self.model.load_state_dict(torch.load('../model/save_model/{}/model_{}.pkl'.format(DEFAULT_CONFIG['model_name'],
                                                                                           DEFAULT_CONFIG[
                                                                                               'experiment_name'])))
        logger.info('Beginning eval...')
        self.model.eval()
        self.test_data = tool.read_json(DEFAULT_CONFIG['test_path'])
        with codecs.open(DEFAULT_CONFIG['result_path'], 'w', encoding='utf-8') as f:
            for example in tqdm(self.test_data):
                text = example['originalText']
                text = torch.tensor(
                    numpy.array([self.word_vocab.stoi[word] for word in text], dtype='int64')).unsqueeze(1).to(
                    DEVICE)
                text_len = torch.tensor(numpy.array([text.size(0)], dtype='int64')).to(DEVICE)
                result_list = self.model(text, text_len)[0]
                tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
                pred_dict = {'originalText': example['originalText'], 'entities': pred_list}
                f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
        logger.info('Finished eval')

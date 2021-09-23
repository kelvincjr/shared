"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.data_utils import covert_to_tokens, search_spo_index, search, Example, sequence_padding, save, load
from utils.utils import logger

def fine_grade_tokenize(raw_text, tokenizer, return_orig_index=True):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []
    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    cur_pos = 0
    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)
        tok_to_orig_start_index.append(cur_pos)
        tok_to_orig_end_index.append(cur_pos)
        cur_pos += 1

    if return_orig_index:
        return tokens, tok_to_orig_start_index, tok_to_orig_end_index
    return tokens

def read_examples(args, json_file):
    
    examples = []
    with open(json_file, 'r') as f:
        p_id = 0
        data_list = json.load(f)
        for src_data in tqdm(data_list):
            p_id += 1
            text_raw = src_data['text']
            text_raw = text_raw.replace('®', '')
            text_raw = text_raw.replace('◆', '')
            tokens, tok_to_orig_start_index, tok_to_orig_end_index = fine_grade_tokenize(text_raw, args.tokenizer, return_orig_index=True)
            assert len(tokens) == len(text_raw)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            sub_po_dict, sub_ent_list, spo_list = dict(), list(), list()
            spoes = {}
            for spo in src_data.get('spo_list', []):
                # spo_dict = dict()
                object_entity_label = args.s2id[spo["object-type"]]
                    
                pred_key = spo["subject-type"] + "_" + spo['predicate'] + "_" + spo["object-type"]
                predicate_label = args.spo_conf[pred_key]
                
                subject_sub_tokens = fine_grade_tokenize(spo['subject'], args.tokenizer, return_orig_index=False)
                object_sub_tokens = fine_grade_tokenize(spo['object'], args.tokenizer, return_orig_index=False)
                    
                subject_entity_label = args.s2id[spo["subject-type"]]
                sub_ent_list.append(spo['subject'])
                        
                subject_start, object_start = search_spo_index(tokens, subject_sub_tokens, object_sub_tokens)
                #subject_start = spo['subject-start']
                #object_start = spo['object-start']
                
                ###########################################
                if subject_start == -1:
                    subject_start = search(subject_sub_tokens, tokens)
                if object_start == -1:
                    object_start = search(object_sub_tokens, tokens)
                ###########################################
                
                if subject_start != -1 and object_start != -1:
                    #subject_start += 1
                    #object_start += 1
                    s = (subject_start,
                                 subject_start + len(subject_sub_tokens) - 1,
                                 subject_entity_label)
                    o = (object_start,
                                 object_start + len(object_sub_tokens) - 1,
                                 object_entity_label,
                                 predicate_label)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)

            orig_text = ''
            if 'orig_text' in src_data:
                orig_text = src_data['orig_text']
            examples.append(
                Example(
                    p_id=p_id,  # 1
                    context=text_raw,  # '《邪少兵王》是冰火未央写的网络小说连载于旗峰天下'
                    tok_to_orig_start_index=tok_to_orig_start_index,
                    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                    tok_to_orig_end_index=tok_to_orig_end_index,
                    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                    bert_tokens=tokens,
                    # ['[CLS]', '《', '邪', '少', '兵', '王', '》', '是', '冰', '火', '未', '央', '写', '的', '网', '络', '小', '说', '连', '载', '于', '旗', '峰', '天', '下', '[SEP]']
                    sub_entity_list=sub_ent_list,  # ['邪少兵王']
                    gold_answer=src_data.get('spo_list', []),
                    # [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '冰火未央'}, 'subject': '邪少兵王'}]
                    spoes=spoes,  # {(2, 5): [(8, 11, 1)]}
                    tmp_spoes=orig_text
                ))
        
    logger.info('examples size is {}'.format(len(examples)))

    return examples


class mhs_DuIEDataset(Dataset):
    def __init__(self, args, examples, data_type):
        self.spo_config = args.spo_conf
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.q_ids = list(range(len(examples)))
        self.examples = examples
        self.E_num = args.E_num
        self.R_num = args.R_num
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.examples[index]

    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids = []
            batch_token_type_ids = []
            batch_subject_type_ids = []
            batch_subject_labels = []
            batch_object_labels = []
            for example in examples:
                spoes = example.spoes
                token_ids = self.tokenizer.encode(example.bert_tokens[1:-1],
                                                  is_split_into_words=False,
                                                  max_length=self.max_len,
                                                  truncation=True)  # TODO
                if self.is_train:
                    # subject标签
                    subject_type_ids = np.zeros(len(token_ids), dtype=np.long)  # [L]
                    subject_labels = np.zeros((len(token_ids), 2, self.E_num + 1),
                                              dtype=np.float32)  # [L, 2, E_num + 1]
                    object_labels = np.zeros((len(token_ids), len(token_ids), self.R_num),
                                             dtype=np.float32)  # [L, L, R_num]
                    for s, os in spoes.items():
                        if s[1] <= self.max_len - 1:  # TODO
                            subject_type_ids[s[1]] = s[2]
                            subject_labels[s[0], 0, s[2]] = 1
                            subject_labels[s[1], 1, s[2]] = 1
                        for o in os:
                            if o[1] <= self.max_len - 1:  # TODO
                                subject_type_ids[o[1]] = o[2]
                                subject_labels[o[0], 0, o[2]] = 1
                                subject_labels[o[1], 1, o[2]] = 1
                            if o[1] <= self.max_len - 1 and s[1] <= self.max_len - 1:
                                object_labels[s[1], o[1], o[3]] = 1
                    print('=============== before print, pid {}==============='.format(p_ids[0]))
                    if p_ids[0] == 0:
                        token_labels = list()
                        token_count = 0
                        print("text_raw: ", example.context)
                        print("spo_list: ", example.gold_answer)
                        print("spoes: ", example.spoes)
                        print('tokens: ', token_ids)
                        print('=============== subject_labels ===============')
                        for token_label in subject_labels[1:-1]:
                            token_label_arr = np.array(token_label)
                            token_label_arr_0 = token_label_arr[0]
                            token_label_arr_1 = token_label_arr[1]
                            arg_label_0 = np.argwhere(token_label_arr_0 == 1).tolist()
                            arg_label_1 = np.argwhere(token_label_arr_1 == 1).tolist()
                            #token_labels.append(arg_label)
                            print("token: {}, subject_label_0: {}, subject_label_1: {}".format(example.context[token_count], arg_label_0, arg_label_1))
                            token_count += 1

                        x_token_count = 0
                        print('=============== object_labels ===============')
                        print('len of object_labels[1:-1]: ', len(object_labels[1:-1]))
                        print('len of object_labels[1:-1][0]: ', len(object_labels[1:-1][0]))
                        for layer2_token_label in object_labels[1:-1]:
                            x_token = example.context[x_token_count]
                            y_token_count = 0
                            for token_label in layer2_token_label[1:-1]:
                                token_label_arr = np.array(token_label)
                                arg_label = np.argwhere(token_label_arr == 1).tolist()
                                #token_labels.append(arg_label)
                                y_token = example.context[y_token_count]
                                if len(arg_label) > 0:
                                    print("x_token: {}, y_token: {}, token_label: {}".format(x_token, y_token, arg_label))
                                y_token_count += 1
                            x_token_count += 1

                    batch_token_ids.append(token_ids)
                    batch_subject_type_ids.append(subject_type_ids)
                    batch_subject_labels.append(subject_labels)
                    batch_object_labels.append(object_labels)
                else:
                    batch_token_ids.append(token_ids)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            if not self.is_train:
                return p_ids, batch_token_ids
            else:
                batch_subject_type_ids = sequence_padding(batch_subject_type_ids, is_float=False)
                batch_subject_labels = sequence_padding(batch_subject_labels,
                                                        padding=np.zeros((2, self.E_num + 1)),
                                                        is_float=True)
                max_len = batch_token_ids.size()[1]
                for index, object_label in enumerate(batch_object_labels):
                    pad_len = max_len - object_label[4].shape[0]
                    object_label = torch.Tensor(object_label)
                    object_label = F.pad(object_label, (0, 0, 0, pad_len, 0, pad_len), mode="constant", value=0)
                    batch_object_labels[index] = object_label
                batch_object_labels = torch.stack(batch_object_labels, dim=0)
                ''' 
                batch_token_ids: [B, L]
                batch_subject_type_ids: [B, L]
                batch_subject_labels: [B, L, 2, E_num+1]
                batch_object_labels: [B, L, L, R_num]
                '''
                return batch_token_ids, \
                       batch_subject_type_ids, \
                       batch_subject_labels, \
                       batch_object_labels

        return partial(collate)


if __name__ == '__main__':
    pass

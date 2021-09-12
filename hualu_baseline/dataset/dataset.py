"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import random
import math
import json
import torch
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizerFast
from torch.utils.data import Dataset

from utils.finetuning_argparse import get_argparse

def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens

class DuIEDataset(Dataset):
    def __init__(self, args, json_path, tokenizer):
        examples = []

        id2predicate, predicate2id, n = {0: "O", 1: "I"}, {"O": 0, "I": 1}, 2
        with open('flyai_data/schema.csv') as f:
            predicate2type = {}
            line_count = 0
            for l in f:
                if line_count == 0:
                    line_count += 1
                    continue
                line = l.strip()
                if line != 'null':
                    items = line.split(',')
                    predicate_type = items[0]+"_"+items[1]+"_"+items[2]
                    subject_type = items[0]
                    object_type = items[2]
                    predicate2type[predicate_type] = (subject_type, object_type)
                    key = predicate_type
                    id2predicate[n] = key
                    predicate2id[key] = n
                    n += 1
                line_count += 1
        label_map = predicate2id
        print('label_map len: ', len(label_map))
        print(label_map)

        D = []
        with open(json_path, encoding='utf-8') as f:
            #l = f.readlines()
            data_list = json.load(f)
            for data in tqdm(data_list):
                d = {'text': data['text'], 'spo_list': []}
                if 'spo_list' in data:
                    for spo in data['spo_list']:
                        sub_category = spo['subject-type']
                        sub_mention = spo['subject']
                        obj_category = spo['object-type']
                        obj_mention = spo['object']
                        sub_start = spo['subject-start']
                        obj_start = spo['object-start']
                        predicate = spo['predicate']
                        predicate = sub_category+"_"+predicate+"_"+obj_category
                        d['spo_list'].append(
                            (sub_mention, predicate, obj_mention, sub_start, obj_start)
                        )
                D.append(d)
        '''
        if json_path.endswith('labeled.json'):
            with open("emergency_data/train2/labeled.json", encoding='utf-8') as f:
                #l = f.readlines()
                data_list = json.load(f)
                for data in tqdm(data_list):
                    d = {'text': data['text'], 'spo_list': []}
                    for spo in data['spo_list']:
                        sub_category = spo['subject-type']
                        sub_mention = spo['subject']
                        obj_category = spo['object-type']
                        obj_mention = spo['object']
                        #sub_start = text.find(sub_mention)
                        #obj_start = text.find(obj_mention)
                        predicate = spo['predicate']
                        predicate = sub_category+"_"+predicate+"_"+obj_category
                        d['spo_list'].append(
                            (sub_mention, predicate, obj_mention)
                        )
                    D.append(d)
        '''
        examples = []
        tokenized_examples = []
        num_labels = 2 * (len(label_map.keys()) - 2) + 2

        print('num_labels {}'.format(num_labels))

        #limit = math.ceil(len(lines)*0.1)
        #random.shuffle(lines)
        #lines = lines[:limit]
        print_first = True
        for example in D:
            #example = json.loads(line)
            # spo_list = example['spo_list'] if "spo_list" in example.keys() else None
            spo_list = example['spo_list'] if "spo_list" in example.keys() else []

            text_raw = example['text']
            
            pre_tokens = fine_grade_tokenize(text_raw, tokenizer)
            assert len(pre_tokens) == len(text_raw)
            tokenized_example = tokenizer.encode_plus(
                pre_tokens,
                max_length=args.max_len,
                padding="max_length",
                is_pretokenized=True,
                truncation=True,
                return_offsets_mapping=True
            )
            '''
            tokenized_example = tokenizer.encode_plus(
                text_raw,
                max_length=args.max_len,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True
            )
            '''
            #
            seq_len = sum(tokenized_example["attention_mask"])
            tokens = tokenized_example["input_ids"]
            #print('tokens: ', tokens)
            labels = [[0] * num_labels for i in range(args.max_len)]
            for spo in spo_list:
                label_subject = label_map[spo[1]]
                label_object = label_subject + len(label_map) - 2
                
                sub_pre_tokens = fine_grade_tokenize(spo[0], tokenizer)
                assert len(sub_pre_tokens) == len(spo[0])
                obj_pre_tokens = fine_grade_tokenize(spo[2], tokenizer)
                assert len(obj_pre_tokens) == len(spo[2])

                subject_tokens = tokenizer.encode_plus(sub_pre_tokens, is_pretokenized=True, add_special_tokens=False)["input_ids"]
                object_tokens = tokenizer.encode_plus(obj_pre_tokens, is_pretokenized=True, add_special_tokens=False)["input_ids"]
                '''
                subject_tokens = tokenizer.encode_plus(spo[0], add_special_tokens=False)["input_ids"]
                object_tokens = tokenizer.encode_plus(spo[2], add_special_tokens=False)["input_ids"]
                subject_tokens_len = len(subject_tokens)
                object_tokens_len = len(object_tokens)
                '''
                sub_start = spo[3]
                obj_start = spo[4]
                index = sub_start
                if index >= args.max_len:
                    print('index {}, max_len {}'.format(index, args.max_len))
                    import sys
                    sys.exit()
                index += 1
                labels[index][label_subject] = 1
                for i in range(subject_tokens_len - 1):
                    labels[index + i + 1][1] = 1

                index = obj_start
                if index != -1:
                    if index >= args.max_len:
                        print('index {}, max_len {}'.format(index, args.max_len))
                        import sys
                        sys.exit()
                    index += 1
                    labels[index][label_object] = 1
                    for i in range(object_tokens_len - 1):
                        labels[index + i + 1][1] = 1

                # assign token label
                # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
                # to prevent single token from being labeled into two different entity
                # we tag the longer entity first, then match the shorter entity within the rest text
                '''
                forbidden_index = None
                if subject_tokens_len > object_tokens_len:
                    for index in range(seq_len - subject_tokens_len + 1):
                        if tokens[index: index + subject_tokens_len] == subject_tokens:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(seq_len - object_tokens_len + 1):
                        if tokens[index: index + object_tokens_len] == object_tokens:
                            if forbidden_index is None:
                                labels[index][label_object] = 1
                                for i in range(object_tokens_len - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            # check if labeled already
                            elif index < forbidden_index or index >= forbidden_index + len(subject_tokens):
                                labels[index][label_object] = 1
                                for i in range(object_tokens_len - 1):
                                    labels[index + i + 1][1] = 1
                                break
                else:
                    for index in range(seq_len - object_tokens_len + 1):
                        if tokens[index:index + object_tokens_len] == object_tokens:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(seq_len - subject_tokens_len + 1):
                        if tokens[index:index +
                                        subject_tokens_len] == subject_tokens:
                            if forbidden_index is None:
                                labels[index][label_subject] = 1
                                for i in range(subject_tokens_len - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            elif index < forbidden_index or index >= forbidden_index + len(
                                    object_tokens):
                                labels[index][label_subject] = 1
                                for i in range(subject_tokens_len - 1):
                                    labels[index + i + 1][1] = 1
                                break
                '''
            for i in range(seq_len):
                if labels[i] == [0] * num_labels:
                    labels[i][0] = 1
            tokenized_example["labels"] = labels
            tokenized_example["seq_len"] = seq_len

            examples.append(example)
            tokenized_examples.append(tokenized_example)

            if print_first:
                token_labels = list()
                token_count = 0
                print("text_raw: ", text_raw)
                print("text_raw len {}, seq_len: {}".format(len(text_raw), seq_len))
                print("spo_list: ", spo_list)
                print('tokens: ', tokens)
                offset_mapping = tokenized_example['offset_mapping']
                import numpy as np
                for token_label in labels[1:seq_len - 1]:
                    token_label_arr = np.array(token_label)
                    arg_label = np.argwhere(token_label_arr == 1).tolist()
                    token_labels.append(arg_label)
                    print("token: {}, token_label: {}, offset_mapping: {}".format(text_raw[token_count], arg_label, offset_mapping[token_count]))
                    token_count += 1
                print_first = False

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"][:max_len] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
    }


if __name__ == '__main__':
    args = get_argparse().parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")
    dataset = DuIEDataset(args, "../data/duie_train.json", tokenizer)
    a = 1

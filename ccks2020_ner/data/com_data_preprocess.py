import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter

train_text_file = r'commodity/train/input.seq.char'
train_bio_file = r'commodity/train/output.seq.bioattr'
test_text_file = r'commodity/test/input.seq.char'
test_bio_file = r'commodity/test/output.seq.bioattr'
schema_file = r'commodity/vocab_attr.txt'
out_train_bio_file = r'commodity/train/output.seq.bio.format'
out_test_bio_file = r'commodity/test/output.seq.bio.format'

def load_texts(filename):
    D = []
    num_lines = 0
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text = l.strip().replace("\n", '').replace(r'[SPA]','#')
            tokens = text.split(' ')
            D.append(tokens)
            num_lines += 1
            if (num_lines % 1000) == 0:
                print('{} number of text lines read'.format(num_lines))
    return D

def load_bioattr_labels(filename):
    D = []
    num_lines = 0
    key_bio_lines = {}
    with open(filename, encoding='utf-8') as f:
        for l in f:
            tags = l.split(' ')
            D.append(tags)
            num_lines += 1
            if (num_lines % 1000) == 0:
                print('{} number of bio lines read'.format(num_lines))
    return D

def read_schema(filename):
    # 读取schema
    with open(filename, encoding='utf-8') as f:
        D = []
        for l in f:
            if l.strip() != 'null':
                D.append(l.strip())
        return D

def gen_bio_file(tokens_list, tags_list, filename):
    outfile = open(filename, 'w+', encoding='utf-8')
    for i in range(len(tokens_list)):
        tokens = tokens_list[i]
        tags = tags_list[i]
        for j in range(len(tokens)):
            token = tokens[j]
            tag = tags[j]
            outfile.write('{} {}\n'.format(token, tag))
        #outfile.write('\n')

train_texts = load_texts(train_text_file)
train_tags = load_bioattr_labels(train_bio_file)
test_texts = load_texts(test_text_file)
test_tags = load_bioattr_labels(test_bio_file)
ner_labels = read_schema(schema_file)
print(ner_labels)
gen_bio_file(train_texts, train_tags, out_train_bio_file)
gen_bio_file(test_texts, test_tags, out_test_bio_file)
# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 19:34
# @File    : demo.py

"""
file description:

"""
import sys
sys.path.insert(0, '../')  # 添加路径

import torch
from modules.model_ner import SeqLabel
from modules.model_rel import AttBiLSTM
from utils.config_ner import ConfigNer, USE_CUDA
from utils.config_rel import ConfigRel

from data_loader.process_ner import ModelDataPreparation
from data_loader.process_rel import DataPreparationRel

from mains import trainer_ner, trainer_rel
import json
from transformers import BertForSequenceClassification
from collections import defaultdict
import codecs
from tqdm import tqdm

def get_entities(pred_ner, text):
    token_types = [[] for _ in range(len(pred_ner))]
    entities = [[] for _ in range(len(pred_ner))]
    for i in range(len(pred_ner)):
        token_type = []
        entity = []
        j = 0
        word_begin = False
        while j < len(pred_ner[i]):
            if pred_ner[i][j][0] == 'B':
                if word_begin:
                    token_type = []  # 防止多个B出现在一起
                    entity = []
                token_type.append(pred_ner[i][j])
                entity.append(text[i][j])
                word_begin = True
            elif pred_ner[i][j][0] == 'I':
                if word_begin:
                    token_type.append(pred_ner[i][j])
                    entity.append(text[i][j])
            else:
                if word_begin:
                    token_types[i].append(''.join(token_type))
                    token_type = []
                    entities[i].append(''.join(entity))
                    entity = []
                word_begin = False
            j += 1
    return token_types, entities


def gen_submission(rel_test_path, rel_pred, submission_file):
    D = []
    total_count = 0
    data_dict = dict()
    with open(rel_test_path, encoding='utf-8') as f:
        data_list = json.load(f)
        for data in tqdm(data_list):
            text = data['text']
            id_ = data['id']
            spo_list = data['spo_list']
            if id_ not in data_dict:
                data_dict[id_] = {'id': id_, 'text': text, 'spo_list': []}
            d = data_dict[id_]
            rel = rel_pred[total_count]
            total_count += 1
            #rel = rel_pred[text]
            if rel != 'N':
                for spo in spo_list:
                    sub_category = spo['subject-type']
                    sub_mention = spo['subject']
                    obj_category = spo['object-type']
                    obj_mention = spo['object']
                    predicate = rel
                    d['spo_list'].append({
                            "subject": sub_mention,
                            'subject-type': sub_category,
                            "predicate": predicate,
                            "object": obj_mention,
                            "object-type": obj_category
                            })

    print('total_count {}'.format(total_count))
    sorted_data_dict = sorted(data_dict.items(), key=lambda x: x[0], reverse=False)
    for key, item in sorted_data_dict:
        D.append(item)

    with codecs.open(submission_file, 'w', 'utf-8') as f:
        #json_str = json.dumps(D, ensure_ascii=False, indent=4)
        #f.write(json_str)
        for d in D:
            json_str = json.dumps(d, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
    return D

def test():
    '''
    test_path = '../data/dev.json'
    PATH_NER = '../models/sequence_labeling/88m-f36.10n5047.16ccks2019_ner.pth'
    config_ner = ConfigNer()
    config_ner.batch_size = 1
    ner_model = SeqLabel(config_ner)
    ner_model_dict = torch.load(PATH_NER)
    ner_model.load_state_dict(ner_model_dict['state_dict'])
    
    ner_data_process = ModelDataPreparation(config_ner)
    _, _, test_loader = ner_data_process.get_train_dev_data(path_test=test_path)
    trainerNer = trainer_ner.Trainer(ner_model, config_ner, test_dataset=test_loader)
    pred_ner = trainerNer.predict()
    #print("=========================================================")
    #print(pred_ner)

    text = list()
    idx = list()
    index_count = 0
    for data_item in test_loader:
        text.extend(data_item['text'])
        idx.append(index_count)
        index_count += 1
    
    token_types, entities = get_entities(pred_ner, text)
    
    #for i in range(len(pred_ner)):
    #    print(text[i])
    #    print(token_types[i])
    #    print(entities[i])
    #    print("=========================================================")
    
    rel_list = []
    with open('./rel_predict.json', 'w', encoding='utf-8') as f:
        for i in range(len(pred_ner)):
            texti = text[i]
            idxi = idx[i]
            for j in range(len(entities[i])):
                subject_type = token_types[i][j].split('-')[-1]
                for k in range(len(entities[i])):
                    object_type = token_types[i][k].split('-')[-1]
                    if j >= k or entities[i][j] == entities[i][k]:
                        continue
                    rel_list.append({"id":idxi, "text":texti, "spo_list":[{"subject": entities[i][j], "subject-type": subject_type, "object": entities[i][k], "object-type": object_type}]})
        json.dump(rel_list, f, ensure_ascii=False, indent=4)
    '''
    
    PATH_REL = '/kaggle/input/entity-pipeline-output/entity_relation_pipeline/models/rel_cls/29m-acc0.77ccks2019_rel.pth'

    config_rel = ConfigRel()
    config_rel.batch_size = 1
    rel_model = BertForSequenceClassification.from_pretrained('/kaggle/working', num_labels=config_rel.num_relations)
    # rel_model = AttBiLSTM(config_rel)
    rel_model_dict = torch.load(PATH_REL, map_location='cpu')
    rel_model.load_state_dict(rel_model_dict['state_dict'])
    rel_test_path = './rel_predict.json'

    rel_data_process = DataPreparationRel(config_rel)
    _, _, test_loader = rel_data_process.get_train_dev_data(path_test=rel_test_path, is_test=True)
    trainREL = trainer_rel.Trainer(rel_model, config_rel, test_dataset=test_loader)
    rel_pred = trainREL.bert_predict()
    #print(rel_pred)
    gen_submission(rel_test_path, rel_pred, './result.json')
    

if __name__ == '__main__':
    test()

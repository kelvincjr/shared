"""
@Time : 2021/2/78:58
@Auth : 周俊贤
@File ：utils.py
@DESCRIPTION:

"""

import codecs
import json
import logging
import os
import random
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def find_entity(text_raw,
                id_,
                predictions,
                offset_mapping):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):  # 超过序列的长度
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            #print('id_ {}, i {}, offset_mapping {}, {}'.format(id_, i, offset_mapping[i][0], offset_mapping[i + j][1]))
            #entity = ''.join(text_raw[offset_mapping[i][0]:
                                      #offset_mapping[i + j][1]])
            entity = ''.join(text_raw[i:i + j + 1])
            entity_list.append((entity, i))
    return list(set(entity_list))


def decoding(example_all,
             id2spo,
             logits_all,
             seq_len_all,
             offset_mapping_all
             ):
    """
    model output logits -> formatted spo (as in data set file)
    """
    single_spo_ids = {}
    for id_, object_type in id2spo['object_type'].items():
        if object_type == "":
            single_spo_ids[id_] = id_

    formatted_outputs = []
    for (i, (example, logits, seq_len, offset_mapping)) in \
            enumerate(zip(example_all, logits_all, seq_len_all, offset_mapping_all)):
        logits = logits[1:seq_len - 2 + 1]  # slice between [CLS] and [SEP] to get valid logits
        import copy
        origin_logits = copy.deepcopy(logits)
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
    
        offset_mapping = offset_mapping[1:seq_len - 2 + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        
        if i == 0:
            token_labels = list()
            token_count = 0
            torch.set_printoptions(precision=5)
            print("text_raw: ", text_raw)
            print("text_raw len {}, seq_len: {}".format(len(text_raw), seq_len))
            for token in logits:
                token_logit = np.argwhere(token == 1).tolist()
                print("token: {}, token_logit: {}".format(text_raw[token_count], token_logit))
                #print('offset_mapping: ', offset_mapping[token_count])
                token_count += 1

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 60 and (cls_label + 59) in flatten_predictions:
                subject_id_list.append(cls_label)
            elif cls_label in single_spo_ids:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        if i == 0: 
            print('flatten_predictions: ', flatten_predictions)
            print('subject_id_list: ', subject_id_list)

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            subjects = find_entity(text_raw,
                                       id_,
                                       predictions,
                                       offset_mapping)
            #print('id {}, subjects: {}'.format(id_, subjects))
            objects = find_entity(text_raw,
                                      id_ + 59,
                                      predictions,
                                      offset_mapping)
            #print('objects: ', objects)
            if id_ in single_spo_ids:
                for subject_, sub_ind in subjects:
                    spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": id2spo['object_type'][id_],
                            'subject_type': id2spo['subject_type'][id_],
                            "object": "",
                            "subject": subject_
                        })
            else:
                for subject_, sub_ind in subjects:
                    for object_, obj_ind in objects:
                        if abs((sub_ind - obj_ind)) >= 15:
                            continue 
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": id2spo['object_type'][id_],
                            'subject_type': id2spo['subject_type'][id_],
                            "object": object_,
                            "subject": subject_
                        })
            
        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs

def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        # zipfile_path = file_path + '.zip'
        # f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        # f.write(file_path)

    # return zipfile_path

def get_precision_recall_f1(golden_file, predict_file):
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'.
            format(golden_file, predict_file))
    result = r.read()
    print("test", result)
    r.close()
    d_result = json.loads(result)
    precision = d_result["precision"]
    recall = d_result["recall"]
    f1 = d_result["f1-score"]

    return precision, recall, f1

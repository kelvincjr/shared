import warnings
warnings.filterwarnings("ignore")

import os
import jieba
import torch
import pickle
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import json
import numpy as np
import sys
#sys.path.insert(0, '/data/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/ark-nlp-main')
sys.path.insert(0, './ark-nlp-main')
from ark_nlp.factory.utils.seed import set_seed 
from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBert
from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBertConfig
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer

from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.lr_scheduler import get_default_linear_schedule_with_warmup
from ark_nlp.factory.utils.attack import FGM, PGD
from ark_nlp.factory.utils.conlleval import get_entity_bio

set_seed(42)

#data_path = '/data/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/baseline/data/'
data_path = './data/'

#from NEZHA.modeling_nezha import NeZhaModel
#from NEZHA.configuration_nezha import NeZhaConfig

#from ark_nlp.nn.base.nezha import NeZhaModel
#from ark_nlp.nn.configuration.configuration_nezha import NeZhaConfig

#nezha_bert = NeZhaModel.from_pretrained('peterchou/nezha-chinese-base')
#sys.exit(0)

import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--max_len", default=128, type=int, help="最大长度")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="验证Batch size的大小")

    # 训练的参数
    parser.add_argument("--model_name_or_path", default="hfl/chinese-bert-wwm", type=str, help="预训练模型的路径")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="训练轮数")
    parser.add_argument("--early_stop", default=8, type=int, help="早停")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="transformer层学习率")
    parser.add_argument("--linear_learning_rate", default=1e-3, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--eval_size", type=int, default=400, help="eval set size")
    parser.add_argument("--output_dir", default="./model_save", type=str, help="保存模型的路径")
    
    return parser

def get_raw_data():
    datalist = []
    max_len = 0
    len_count_32 = 0
    len_count_64 = 0
    with open(data_path + 'train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')
        
        text = []
        labels = []
        label_set = set()
        
        for line in lines: 
            if line == '\n':                
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx+1]
                    })
                    
                if text == '':
                    continue
                
                datalist.append({
                    'text': text,
                    'label': entity_labels
                })

                if len(text) > max_len:
                    max_len = len(text)
                if len(text) > 32:
                    len_count_32 += 1
                if len(text) > 64:
                    len_count_64 += 1
                
                text = []
                labels = []
                
            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
    print('===== data preprocess done, datalist len: {}, len_count_32: {}, len_count_64: {}, max_len: {} ====='.format(len(datalist), len_count_32, len_count_64, max_len))
    return datalist, label_set

def prepare_dataset(datalist, label_set, eval_size):
    #train_data_df = pd.DataFrame(datalist[:100])
    train_data_df = pd.DataFrame(datalist[:-eval_size])
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

    #dev_data_df = pd.DataFrame(datalist[-100:])
    dev_data_df = pd.DataFrame(datalist[-eval_size:])
    dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))
    print('===== dataframe init done =====')

    label_list = sorted(list(label_set))
    print('===== label_list =====')
    print(label_list)
    ner_train_dataset = Dataset(train_data_df, categories=label_list)
    print('===== cat2id =====')
    print(ner_train_dataset.cat2id)
    #sys.exit(0)
    ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)
    print('===== dataset init done =====')
    return ner_train_dataset, ner_dev_dataset

#tokenizer = Tokenizer(vocab='hfl/chinese-bert-wwm', max_seq_len=128)
#model_path = '../../model/jd_bert'
#model_path = '../bert-base/'
#model_path = './bert-base/'
#model_path = 'hfl/chinese-bert-wwm'
#model_path = 'hfl/chinese-macbert-large'
#model_path = 'nghuyong/ernie-1.0'
#model_path = 'junnyu/uer_large'
#model_path = 'peterchou/nezha-chinese-base'
#model_path = 'uer/chinese_roberta_L-12_H-768'
#model_path = 'hfl/chinese-roberta-wwm-ext'

def convert_to_ids(model_path, ner_train_dataset, ner_dev_dataset):
    tokenizer = Tokenizer(vocab=model_path, max_seq_len=128)
    print('===== tokenizer init done =====')

    ner_train_dataset.convert_to_ids(tokenizer)
    print('===== train data convert_to_ids done =====')
    ner_dev_dataset.convert_to_ids(tokenizer)
    print('===== dev data convert_to_ids done =====')
    return tokenizer #, ner_train_dataset, ner_dev_dataset

def build_model(model_path, tokenizer, ner_train_dataset, ner_dev_dataset, num_epoches, batch_size):

    config = GlobalPointerBertConfig.from_pretrained(model_path, num_labels=len(ner_train_dataset.cat2id))
    torch.cuda.empty_cache()
    dl_module = GlobalPointerBert.from_pretrained(model_path, config=config)
    optimizer = get_default_model_optimizer(dl_module)

    # 设置运行次数
    #num_epoches = 5
    #batch_size = 2 #16

    class AttackTask(Task):
        
        def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
        ):
            print('===== AttackTask =====')
            if hasattr(train_data, 'id2cat'):
                self.id2cat = train_data.id2cat
                self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

            # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
            if self.class_num is None:
                if hasattr(train_data, 'class_num'):
                    self.class_num = train_data.class_num
                else:
                    warnings.warn("The class_num is None.")

            if train_to_device_cols is None:
                self.train_to_device_cols = train_data.to_device_cols
            else:
                self.train_to_device_cols = train_to_device_cols

            train_generator = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self._train_collate_fn
            )
            self.train_generator_lenth = len(train_generator)

            self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
            self.optimizer.zero_grad()

            self.scheduler = get_default_linear_schedule_with_warmup(self.optimizer, num_epoches * self.train_generator_lenth)

            self.module.train()
            
            #self.fgm = PGD(self.module)
            self.fgm = FGM(self.module)

            self._on_train_begin_record(**kwargs)

            return train_generator
        
        def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
        ):

            # 如果GPU数量大于1
            if self.n_gpu > 1:
                loss = loss.mean()
            # 如果使用了梯度累积，除以累积的轮数
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            
            self.fgm.attack()
            logits = self.module(**inputs)
            _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
            attck_loss.backward()
            self.fgm.restore() 

            #add kelvin
            #self.optimizer.step()
            #self.scheduler.step()
            #self.optimizer.zero_grad()
            
            self._on_backward_record(loss, **kwargs)

            return loss
       
        def _on_optimize(
            self,
            inputs,
            outputs,
            logits,
            loss,
            grad_clip=2,
            **kwargs
        ):

            # 梯度裁剪
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.module.parameters(),
                    grad_clip
                )

            # 更新权值
            self.optimizer.step()

            #if self.ema_decay:
                #self.ema.update(self.module.parameters())

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 清空梯度
            self.optimizer.zero_grad()

            self._on_optimize_record(inputs, outputs, logits, loss, **kwargs)

        def _on_optimize_record(
            self,
            inputs,
            outputs,
            logits,
            loss,
            **kwargs
        ):
            self.logs['global_step'] += 1
            self.logs['epoch_step'] += 1

        def _on_evaluate_end(
            self,
            evaluate_save=True,
            save_module_path=None,
            **kwargs
        ):

            if evaluate_save:
                if save_module_path is None:
                    prefix = './model_save/' + str(self.module.__class__.__name__) + '_'
                    save_module_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

                torch.save(self.module.state_dict(), save_module_path)

            self._on_evaluate_end_record()

            if self.ema_decay:
                self.ema.restore(self.module.parameters())


    #model = Task(dl_module, optimizer, 'gpce', cuda_device=0)
    #model = AttackTask(dl_module, 'adamw', 'lsce', cuda_device=0, ema_decay=0.995)
    model = AttackTask(dl_module, optimizer, 'gpce', cuda_device=0)
    return model

def train(model, ner_train_dataset, ner_dev_dataset, num_epoches, batch_size):
    print('===== start to train =====')
    model.fit(ner_train_dataset, 
              ner_dev_dataset,
              lr=2e-5,
              epochs=num_epoches, 
              batch_size=batch_size
             )

    torch.save(model.module.state_dict(), './model_save.pth')
    print('===== model save done =====')

def load_model(model):
    model.module.load_state_dict(torch.load('./model_save/model_save.pth', map_location='cpu'))
    print('===== model load done =====')
    return model

def predict(model, tokenizer, ner_train_dataset, ner_dev_dataset):
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
    class GlobalPointerNERPredictor(object):
        """
        GlobalPointer命名实体识别的预测器

        Args:
            module: 深度学习模型
            tokernizer: 分词器
            cat2id (:obj:`dict`): 标签映射
        """  # noqa: ignore flake8"

        def __init__(
            self,
            module,
            tokernizer,
            cat2id
        ):
            self.module = module
            self.module.task = 'TokenLevel'

            self.cat2id = cat2id
            self.tokenizer = tokernizer
            self.device = list(self.module.parameters())[0].device

            self.id2cat = {}
            for cat_, idx_ in self.cat2id.items():
                self.id2cat[idx_] = cat_

        def _convert_to_transfomer_ids(
            self,
            text
        ):

            tokens = self.tokenizer.tokenize(text)
            token_mapping = self.tokenizer.get_token_mapping(text, tokens)

            input_ids = self.tokenizer.sequence_to_ids(tokens)
            input_ids, input_mask, segment_ids = input_ids

            zero = [0 for i in range(self.tokenizer.max_seq_len)]
            span_mask = [input_mask for i in range(sum(input_mask))]
            span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
            span_mask = np.array(span_mask)

            features = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'span_mask': span_mask
            }

            return features, token_mapping

        def _get_input_ids(
            self,
            text
        ):
            if self.tokenizer.tokenizer_type == 'vanilla':
                return self._convert_to_vanilla_ids(text)
            elif self.tokenizer.tokenizer_type == 'transfomer':
                return self._convert_to_transfomer_ids(text)
            elif self.tokenizer.tokenizer_type == 'customized':
                return self._convert_to_customized_ids(text)
            else:
                raise ValueError("The tokenizer type does not exist")

        def _get_module_one_sample_inputs(
            self,
            features
        ):
            return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

        def predict_one_sample(
            self,
            text='',
            threshold=0
        ):
            """
            单样本预测

            Args:
                text (:obj:`string`): 输入文本
                threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
            """  # noqa: ignore flake8"

            features, token_mapping = self._get_input_ids(text)
            self.module.eval()

            with torch.no_grad():
                inputs = self._get_module_one_sample_inputs(features)
                scores = self.module(**inputs)[0].cpu()
                
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf

            entities = []

            for category, start, end in zip(*np.where(scores > threshold)):
                if end-1 > token_mapping[-1][-1]:
                    break
                if token_mapping[start-1][0] <= token_mapping[end-1][-1]:
                    entitie_ = {
                        "start_idx": token_mapping[start-1][0],
                        "end_idx": token_mapping[end-1][-1],
                        "entity": text[token_mapping[start-1][0]: token_mapping[end-1][-1]+1],
                        "type": self.id2cat[category]
                    }

                    if entitie_['entity'] == '':
                        continue

                    entities.append(entitie_)

            return entities

    ner_predictor_instance = GlobalPointerNERPredictor(model.module, tokenizer, ner_train_dataset.cat2id)

    from tqdm import tqdm

    predict_results = []

    print('===== start to predict =====')
    with open(data_path + 'sample_per_line_preliminary_A.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            label = len(_line) * ['O']
            for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
                if 'I' in label[_preditc['start_idx']]:
                    continue
                if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                    continue
                if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                    continue

                label[_preditc['start_idx']] = 'B-' +  _preditc['type']
                label[_preditc['start_idx']+1: _preditc['end_idx']+1] = (_preditc['end_idx'] - _preditc['start_idx']) * [('I-' +  _preditc['type'])]
                
            predict_results.append([_line, label])

    print('===== start to save result =====')
    with open(data_path + 'submit_result.txt', 'w', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                if word == '\n':
                    continue
                f.write(f'{word} {tag}\n')
            f.write('\n')

    print('===== save result done =====')

if __name__ == "__main__":
    args = get_argparse().parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    datalist, label_set = get_raw_data()
    ner_train_dataset, ner_dev_dataset = prepare_dataset(datalist, label_set, args.eval_size)

    num_epoches = args.num_train_epochs
    batch_size = args.per_gpu_train_batch_size
    model_path = args.model_name_or_path
    tk_model_path = 'hfl/chinese-bert-wwm'
    tokenizer = convert_to_ids(tk_model_path, ner_train_dataset, ner_dev_dataset)
    model = build_model(model_path, tokenizer, ner_train_dataset, ner_dev_dataset, num_epoches, batch_size)
    train(model, ner_train_dataset, ner_dev_dataset, num_epoches, batch_size)

    predict(model, tokenizer, ner_train_dataset, ner_dev_dataset)
from __future__ import print_function
import sys
import os
os.environ['TF_KERAS'] = '1'
import json
import random
import re
import numpy as np
from tqdm import tqdm
#sys.path.insert(0, '/opt/kelvin/python/knowledge_graph/ai_contest/df_textgen/nlpcc2017/nlpcc2017ds/bert4keras-0.10.8')

from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
maxlen = 128
batch_size = 16
epochs = 10

# bert配置
#config_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
#
config_path = '/kaggle/working/bert_config.json'
checkpoint_path = '/kaggle/working/bert_model.ckpt'
dict_path = '/kaggle/working/vocab.txt'

bracket_pattern_str = '（.*?）'

def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]

def load_data(filename):
    D = []
    with open(filename) as f:
        lines = f.readlines()
        print('===== train data lines num {} ====='.format(len(lines)))
        for line in lines:
            line = re.sub(bracket_pattern_str, "", line)
            sents = text_segmentate(line.strip(), 1, u'，。')
            for i in range(len(sents) - 1):
                D.append((sents[i+1], sents[i]))
    return D

# 加载数据集
train_data = load_data('data/train_data.csv')
valid_data = load_data('data/dev_data.csv')
print('===== before deduplicate, train_data len: {}, valid_data len: {}'.format(len(train_data), len(valid_data)))
train_data = list(set(train_data))
valid_data = list(set(valid_data))
print('===== before deduplicate, train_data len: {}, valid_data len: {}'.format(len(train_data), len(valid_data)))
random.shuffle(train_data)
random.shuffle(valid_data)
print('train_data first 10: ')
for i in range(10):
    print('next sent: {}, cur sent: {}'.format(train_data[i][0], train_data[i][1]))

print('valid_data first 10: ')
for i in range(10):
    print('next sent: {}, cur sent: {}'.format(valid_data[i][0], valid_data[i][1]))

print('===== train_data len {} ====='.format(len(train_data)))
print('===== valid_data len {} ====='.format(len(valid_data)))
#train_data = train_data[:1800]
valid_data = valid_data[:200]
#sys.exit(0)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

print('===== load_data done =====')

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path, # 模型的配置文件
    checkpoint_path, # 模型的预训练权重
    application='unilm', # 模型的用途
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-6))
model.summary()

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


def just_show():
    #s1 = u'2020年08月09日08点15分'
    #s2 = u'于当天5时43分许行驶至127公里500米处时'
    #s1 = u'[TIME]在[AL]'
    #s2 = u'[AO]发生一起建筑施工行业高处坠落事故'
    s1 = u'[AO]发生一起道路运输行业道路运输事故'
    s2 = u'结果小型轿车尾部与[P]身体发生碰撞'
    s3 = u'造成[P]受伤和车辆损坏的交通事故'
    for s in [s1, s2, s3]:
        print(u'预测下一句:', autotitle.generate(s))
    print()

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        #self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        #if metrics['bleu'] > self.best_bleu:
            #self.best_bleu = metrics['bleu']
        if metrics['rouge-1'] > self.best_bleu:
            self.best_bleu = metrics['rouge-1']
            print('===== best_model save done =====')
            model.save_weights('./tmp/best_model.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
        just_show()

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autotitle.generate(content, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu = 0
                '''
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
                '''
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':
    '''
    model.load_weights('./model_save/bert4keras/best_model.weights')
    print('===== model load done =====')
    just_show()
    '''
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('./model/best_model.weights')


from __future__ import print_function
import sys
import os
os.environ['TF_KERAS'] = '1'
import json
import random
import numpy as np
from tqdm import tqdm
#sys.path.insert(0, './bert4keras-0.10.8')

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
maxlen = 512
batch_size = 2
epochs = 6

# bert配置
#config_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '/opt/kelvin/python/knowledge_graph/baiduee/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
#
config_path = '/kaggle/working/bert_config.json'
checkpoint_path = '/kaggle/working/bert_model.ckpt'
dict_path = '/kaggle/working/vocab.txt'

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            if 'summarization' in l: 
              title = l['summarization']
            else:
              title = None
            content = l['article'].replace('<Paragraph>','\r\n')
            if len(content) + len(title) > maxlen or len(title) > 64: continue
            D.append((title, content))
    return D


# 加载数据集
train_data = load_data('./data/train_with_summ.txt')
valid_data = load_data('./data/train_without_summ.txt')
random.shuffle(train_data)
#train_data = train_data[:16000]
valid_data = valid_data[:1000]

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
    s1 = u'资料图：空军苏27/歼11编队，日方称约40架中国军机在23日8艘海监船驱赶日本船队期间出现在钓鱼岛附近空域日本《产经新闻》4月27日报道声称，中国8艘海监船相继进入钓鱼岛12海里执法的4月23日当天，曾有40多架中国军机出现在钓鱼岛海域周边空域，且中方军机中多半为战斗机，包括中国空军新型战机苏-27和苏-30。日本《产经新闻》声称中国军机是想通过不断的逼近，让日本航空自卫队的战机飞行员形成疲劳。日本政府高官还称：“这是前所未有的威胁。”针对日本媒体的报道，国防部官员在接受环球网采访时称，中国军队飞机在本国管辖海域上空进行正常战备巡逻，日方却颠倒黑白、倒打一耙，肆意渲染“中国威胁”。国防部官员应询表示：4月23日，日方出动多批次F-15战斗机、P3C反潜巡逻机等，对中方正常战备巡逻的飞机进行跟踪、监视和干扰，影响中方飞机正常巡逻和飞行安全。中方对此坚决采取了应对措施。中国军队飞机在本国管辖海域上空进行正常战备巡逻，日方却颠倒黑白、倒打一耙，肆意渲染“中国威胁”。国防部官员称，需要指出的是，今年年初以来，日方不断挑衅，制造事端，并采取“恶人先告状”的手法，抹黑中国军队。事实证明，日方才是地区和平稳定的麻烦制造者。我们要求日方切实采取措施，停止故意制造地区紧张局势的做法。'
    s2 = u'中新网5月26日电/r/n据外媒报道，世界卫生组织日前发布了一份报告，指出自杀已经取代难产，成为全球年轻女性的头号杀手。报道称，根据报告提供的数据显示，东南亚平均每10万名年龄介于15至19岁的女性死者中，就有27.92人死于自杀，男性则为21.41人；欧洲和美洲分别为6.15人及4.72人，全球平均值则是11.73人。报道指出，多年来，难产死亡一直是这个年龄层女性丧命的最主要原因，然而在过去10年，自杀取代难产死亡，成为全球年轻女性死亡的最主要原因。报告将全球分为美洲、东南亚、中东、欧洲、非洲及西太平洋6大地区，自杀唯独在非洲未有列入5大杀手之内，原因是当地难产和艾滋病死因占绝大多数。在东南亚，自杀占少女死因的比率也较其他死因高两倍。专家分析指出，造成这种结果的原因是当地性别歧视较严重。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))
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
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./gdrive/MyDrive/UniLM_Bert4Keras/best_model.weights')  # 保存模型
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
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' ') #,
                    #smoothing_function=self.smooth
                )
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
    #model.load_weights('./model/best_model.weights')
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


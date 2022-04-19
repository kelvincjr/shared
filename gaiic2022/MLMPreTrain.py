import os
import sys
import json
import copy
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizer


class MLMConfig:
    def __init__(self):
        self.mask_probability = 0.15
        self.special_tokens_mask = None
        self.prob_replace_mask = 0.8
        self.prob_replace_rand = 0.1
        self.prob_keep_ori = 0.1
        """
        :param mask_probability: 被mask的token总数
        :param special_token_mask: 特殊token
        :param prob_replace_mask: 被替换成[MASK]的token比率
        :param prob_replace_rand: 被随机替换成其他token比率
        :param prob_keep_ori: 保留原token的比率
        """

    def trainingConfig(self, max_len, batch_size, epoch, learning_rate, weight_decay, device):
        self.max_len = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

    def ioConfig(self, from_path, save_path):
        self.from_path = from_path
        self.save_path = save_path


class MLMDataset(Dataset):
    def __init__(self, input_texts, tokenizer, config):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.config = config
        self.ori_inputs = copy.deepcopy(input_texts)

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        batch_text = self.input_texts[idx]
        features = self.tokenizer(batch_text, max_length=self.config.max_len, truncation=True, padding='max_length', return_tensors='pt')
        inputs, labels = self.mask_tokens(features['input_ids'])
        batch = {"inputs": inputs, "labels": labels}

        return batch

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.config.mask_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def train(model, config, dataloader):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    for epoch in range(config.epoch):
        training_loss = 0
        print("Epoch: {}".format(epoch + 1))
        for ids, batch in enumerate(tqdm(dataloader)):
            #, mininterval=5, desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):
            input_ids = batch['inputs'].squeeze(1).to(config.device)
            labels = batch['labels'].squeeze(1).to(config.device)
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            batch_loss = loss.item()
            training_loss += batch_loss
            if ids%1000 == 0:
                print('batch training loss: ', batch_loss)
            if (ids == 30000) or (ids == 60000):
                print('===== start to save mlm model, epoch {}, ids {} ====='.format(epoch, ids))
                model.save_pretrained('./model_save/pretrained_model_{}_{}'.format(epoch, ids))

        print("Epoch {} Training loss: ".format(epoch), training_loss)

    print('===== start to save mlm model =====')
    #torch.save(model.module.state_dict(), './model_save/pretrained_model.bin')
    model.save_pretrained('./model_save/pretrained_model')
    print('===== mlm model save done =====')

    return 0


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 0
    len_512_count = 0
    with open('./data/unlabeled_train_data.txt', 'r', encoding='utf-8') as f:
        training_texts = []
        for item in f.readlines():
            if item.strip() == "":
                continue
            else:
                arr = item.strip()
                arr.replace(" ", "")
                if len(arr) > 512:
                    len_512_count += 1
                    continue
                training_texts.append(arr)
                if max_len < len(arr):
                    max_len = len(arr)
    print('size of training_texts: {}, max_len: {}, len_512_count: {}'.format(len(training_texts), max_len, len_512_count))
    config = MLMConfig()
    config.trainingConfig(max_len=max_len, batch_size=16, epoch=2, learning_rate=2e-5, weight_decay=0, device=device)
    config.ioConfig('hfl/chinese-bert-wwm', './model_save/FinetuneEmbeddingModel')
    tokenizer = BertTokenizer.from_pretrained(config.from_path)
    train_dataset = MLMDataset(training_texts, tokenizer, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,  shuffle=False, num_workers=0)

    bert_model = BertForMaskedLM.from_pretrained('./pretrained_bert')
    #bert_model = BertForMaskedLM.from_pretrained(config.from_path)
    bert_model.to(device)
    train(bert_model, config, train_dataloader)




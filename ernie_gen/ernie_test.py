import os
import ast
import time
import argparse
import logging
import json

import paddle
from tqdm import tqdm
import paddle.nn as nn
# 利用paddle io的数据集读取办法会在内存和cpu的运行过程中更加的稳定
from paddle.io import DataLoader
# 参数我们选用的是ernie gen的参数。令牌器我们可以选择 ErnieTokenizer, ErnieTinyTokenizer, BertTokenizer, \
    # ElectraTokenizer, RobertaTokenizer令牌器 优先推荐ernie tiny令牌器 效果是相对来说比较好的 可以容纳更多的上下文信息
from paddlenlp.transformers import ErnieForGeneration, ErnieTokenizer, ErnieTinyTokenizer, BertTokenizer, \
    ElectraTokenizer, RobertaTokenizer, LinearDecayWithWarmup
from paddlenlp.datasets import load_dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Rouge1, Rouge2
from paddlenlp.utils.log import logger

import sys
sys.path.insert(0, "./example/")
from encode import convert_example, after_padding
from decode import post_process, beam_search_infilling
from model import StackModel

'''
def read(data_path):
    count = 0
    f = json.load(open(data_path))
    print('===== train data len {} ====='.format(len(f)))
    for line in f:
    	yield {'tokens': "\x02".join(list(line[0])), 'labels': "\x02".join(list(line[1]))}


def read_dev(data_path):
	f = json.load(open(data_path))
	print('===== dev data len {} ====='.format(len(f)))
	for line in f:
		yield {'tokens': "\x02".join(list(line[0])), 'labels': "\x02".join(list(line[1]))}
'''
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

def read(data_path):
    count = 0
    f = open(data_path, encoding='utf-8')
    lines = f.readlines()
    print('===== train data len {} ====='.format(len(lines)))
    for line in lines:
    	sents = text_segmentate(line.strip(), 1, u'，。')
    	for i in range(len(sents) - 1):
    		yield {'tokens': sents[i], 'labels': sents[i+1]}

def read_dev(data_path):
	count = 0
	f = open(data_path, encoding='utf-8')
	lines = f.readlines()
	print('===== dev data len {} ====='.format(len(lines)))
	for line in lines:
		sents = text_segmentate(line.strip(), 1, u'，。')
		for i in range(len(sents) - 1):
			yield {'tokens': sents[i], 'labels': sents[i+1]}

# 第一个参数我们把读取数据集的方法给穿进去 ，第二个参数是数据集的地址，因为bml存在版本容量上限，这里面我们选用的是一个只包含一万
# 段文本生成的数据集
map_ds = load_dataset(read, data_path='test_data/train_data.csv', lazy=False)
dev_ds = load_dataset(read_dev, data_path='test_data/dev_data.csv', lazy=False)
print('===== load_dataset done =====')
#sys.exit(0)

def evaluate(model, data_loader, tokenizer, rouge1, rouge2, attn_id,
             tgt_type_id):
            
    model.eval()

    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    evaluated_sentences_ids = []
    reference_sentences_ids = []
    logger.info("Evaluating...")
    for data in tqdm(data_loader):
        (src_ids, src_tids, src_pids, _, _, _, _, _, _, _, _,
         raw_tgt_labels) = data  # never use target when infer
        # Use greedy_search_infilling or beam_search_infilling to get predictions
        output_ids = beam_search_infilling(
            model,
            src_ids,
            src_tids,
            eos_id=eos_id,
            sos_id=sos_id,
            attn_id=attn_id,
            pad_id=pad_id,
            unk_id=unk_id,
            vocab_size=vocab_size,
            max_decode_len=max_decode_len,
            max_encode_len=max_encode_len,
            beam_width=beam_width,
            length_penalty=length_penalty,
            tgt_type_id=tgt_type_id)

        for ids in output_ids.tolist():
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            evaluated_sentences_ids.append(ids)

        for ids in raw_tgt_labels.numpy().tolist():
            ids = ids[:ids.index(eos_id)]
            reference_sentences_ids.append(ids)
# 计算rouge1 
    score1 = rouge1.score(evaluated_sentences_ids, reference_sentences_ids)
# 计算rouge2
    score2 = rouge2.score(evaluated_sentences_ids, reference_sentences_ids)
# 日志打印 rouge1 rouge2
    logger.info("Rouge-1: %.5f ,Rouge-2: %.5f" % (score1 * 100, score2 * 100))

    evaluated_sentences = []
    reference_sentences = []
    for ids in reference_sentences_ids[:5]:
        reference_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    for ids in evaluated_sentences_ids[:5]:
        evaluated_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    logger.debug(reference_sentences)
    logger.debug(evaluated_sentences)

    model.train()

def test(model_save_path):
    paddle.set_device(device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model = ErnieForGeneration.from_pretrained(model_save_path)
    tokenizer = ErnieTokenizer.from_pretrained(model_save_path)

    train_dataset, dev_dataset = map_ds, dev_ds
    attn_id = tokenizer.vocab['[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']
    tgt_type_id = model.sent_emb.weight.shape[0] - 1

    rouge1 = Rouge1()
    rouge2 = Rouge2()

    trans_func = convert_example(
        tokenizer=tokenizer,
        attn_id=attn_id,
        tgt_type_id=tgt_type_id,
        max_encode_len=max_encode_len,
        max_decode_len=max_decode_len,
        noise_prob=noise_prob,
        use_random_noice=use_random_noice)

    train_dataset = train_dataset.map(trans_func)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # src_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tgt_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # attn_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_labels
    ): after_padding(fn(samples))
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_dataset = dev_dataset.map(trans_func)
    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model.eval()

    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    evaluated_sentences_ids = []
    reference_sentences_ids = []
    logger.info("Testing...")
    for data in tqdm(dev_data_loader):
        (src_ids, src_tids, src_pids, _, _, _, _, _, _, _, _,
         raw_tgt_labels) = data  # never use target when infer
        # Use greedy_search_infilling or beam_search_infilling to get predictions
        output_ids = beam_search_infilling(
            model,
            src_ids,
            src_tids,
            eos_id=eos_id,
            sos_id=sos_id,
            attn_id=attn_id,
            pad_id=pad_id,
            unk_id=unk_id,
            vocab_size=vocab_size,
            max_decode_len=max_decode_len,
            max_encode_len=max_encode_len,
            beam_width=beam_width,
            length_penalty=length_penalty,
            tgt_type_id=tgt_type_id)

        for ids in output_ids.tolist():
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            evaluated_sentences_ids.append(ids)

        for ids in raw_tgt_labels.numpy().tolist():
            ids = ids[:ids.index(eos_id)]
            reference_sentences_ids.append(ids)

    evaluated_sentences = []
    reference_sentences = []
    for ids in reference_sentences_ids[:5]:
        reference_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    for ids in evaluated_sentences_ids[:5]:
        evaluated_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    logger.debug(reference_sentences)
    logger.debug(evaluated_sentences)

def train():
    paddle.set_device(device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model = ErnieForGeneration.from_pretrained(model_name_or_path)
    # 文本特征分割转id的差异
    if "ernie-tiny" in model_name_or_path:
        tokenizer = ErnieTinyTokenizer.from_pretrained(model_name_or_path)
    elif "ernie" in model_name_or_path:
        tokenizer = ErnieTokenizer.from_pretrained(model_name_or_path)
    elif "roberta" in model_name_or_path or "rbt" in model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    elif "electra" in model_name_or_path:
        tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    if init_checkpoint:
        model_state = paddle.load(init_checkpoint)
        model.set_state_dict(model_state)

    train_dataset, dev_dataset = map_ds, dev_ds
    attn_id = tokenizer.vocab[
        '[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']
    tgt_type_id = model.sent_emb.weight.shape[0] - 1

    trans_func = convert_example(
        tokenizer=tokenizer,
        attn_id=attn_id,
        tgt_type_id=tgt_type_id,
        max_encode_len=max_encode_len,
        max_decode_len=max_decode_len,
        noise_prob=noise_prob,
        use_random_noice=use_random_noice)

    train_dataset = train_dataset.map(trans_func)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # src_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tgt_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # attn_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_labels
    ): after_padding(fn(samples))
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_dataset = dev_dataset.map(trans_func)
    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    label_num = model.word_emb.weight.shape[0]
    train_model = StackModel(model)
    if paddle.distributed.get_world_size() > 1:
        # All 'forward' outputs derived from the module parameters using in DataParallel
        # must participate in the calculation of losses and subsequent gradient calculations.
        # So we use StackModel here to make the model only output loss in its 'forward' function.
        train_model = paddle.DataParallel(train_model)

    max_steps = len(train_data_loader) * num_epochs

    lr_scheduler = LinearDecayWithWarmup(learning_rate, max_steps,
                                         warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=adam_epsilon,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        grad_clip=nn.ClipGradByGlobalNorm(1.0),
        apply_decay_param_fun=lambda x: x in decay_params)

    rouge1 = Rouge1()
    rouge2 = Rouge2()

    global_step = 1
    tic_train = time.time()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            (src_ids, src_tids, src_pids, tgt_ids, tgt_tids, tgt_pids, attn_ids,
             mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn,
             tgt_labels, _) = batch
            # import pdb; pdb.set_trace()
            if label_smooth > 0.:
                tgt_labels = nn.functional.label_smooth(
                    nn.functional.one_hot(tgt_labels, label_num),
                    epsilon=label_smooth)
            tgt_pos = paddle.nonzero(attn_ids == attn_id)
            loss = train_model(src_ids, src_tids, src_pids, tgt_ids, tgt_tids,
                               tgt_pids, attn_ids, mask_src_2_src,
                               mask_tgt_2_srctgt, mask_attn_2_srctgtattn,
                               tgt_labels, tgt_pos)
            if global_step % logging_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s, lr: %.3e"
                        % (global_step, epoch, step, loss, logging_steps /
                           (time.time() - tic_train), lr_scheduler.get_lr()))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % save_steps == 0 and paddle.distributed.get_rank(
            ) == 0:
                evaluate(model, dev_data_loader, tokenizer, rouge1, rouge2,
                         attn_id, tgt_type_id)
                output_export_dir = os.path.join(output_dir,
                                                 "model_%d" % global_step)
                if not os.path.exists(output_export_dir):
                    os.makedirs(output_export_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_export_dir)
                tokenizer.save_pretrained(output_export_dir)
            global_step += 1


if __name__ == "__main__":
    adam_epsilon = 1e-08
    # 批次数量
    batch_size = 48
    # 束宽度
    beam_width = 1
    device = 'gpu'
    label_smooth = 0.0
    # 学习率
    learning_rate = 2e-05
    length_penalty = 1.0
    # 日志条数
    logging_steps = 100
    # 输入最大长度
    max_decode_len = 128 #64
    # 输出最大长度
    max_encode_len = 128 #64
    # 基础版本模型选型
    model_name_or_path = 'ernie-1.0'
    noise_prob = 0.0
    num_epochs = 12
    # 模型文件输出地址
    output_dir = './tmp/'
    save_dir = None
    # 多少步保存一次模型
    save_steps = 3000
    # 使用随机噪声
    use_random_noice = False
    # 激活层比例
    warmup_proportion = 0.1
    weight_decay = 0.1
    init_checkpoint =  None
    train()
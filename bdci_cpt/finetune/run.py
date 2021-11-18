import sys
import time
#sys.path.insert(0, "/data/kelvin/python/asr/wav2vec/huggingface_hub-0.0.19/src/")
#sys.path.insert(0, "/data/kelvin/python/asr/wav2vec/tokenizers-0.10.3/py_src/tokenizers")
#sys.path.insert(0, "/data/kelvin/python/asr/wav2vec/powerful_chinese_ASR-main")
#sys.path.insert(0, "/data/kelvin/python/knowledge_graph/ai_contest/df_textgen/cpt/tmg-dialogue")

from transformers import BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
import argparse
from train import Trainer

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../pretrained_models/cpt-base", type=str)
    parser.add_argument("--state_dict", default=None, type=str)
    parser.add_argument("--data_path", default="../dataset/cooked/LCCC", type=str)
    parser.add_argument("--task_type", default="train", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_warmup_steps", default=2000, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--batch_expand_times", default=2, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = CPTForConditionalGeneration.from_pretrained(args.model, cache_dir='/kaggle/working/')
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    if args.task_type == 'train':
    	trainer.train()
    elif if args.task_type == 'test':
    	trainer.test()

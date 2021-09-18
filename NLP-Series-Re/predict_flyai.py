"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config.mpn import spo_config
from dataset.dataset_mhs_flyai import mhs_DuIEDataset, read_examples
from models.model_mhs import model_mhs
from run_evaluation_flyai import convert2ressult, run_evaluate, convert_spo_contour2
from run_mhs_flyai import evaluate
from utils.bert_optimizaation import BertAdam
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, ProgressBar, init_logger, logger, write_prediction_results

def main():
    parser = get_argparse()
    parser.add_argument("--fine_tunning_model",
                        type=str,
                        required=True,
                        help="fine_tuning model path")
    args = parser.parse_args()
    args.time = time.strftime("%m-%d_%H:%M", time.localtime())
    args.cache_data = "./data/mhs"
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # 设置保存目录
    if not os.path.exists(args.output_dir):
        print('mkdir {}'.format(args.output))
        os.mkdir(args.output_dir)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config
    #args.spo_conf = spo_config.BAIDU_RELATION
    #args.id2rel = {item: key for key, item in args.spo_conf.items()}
    #args.rel2id = args.spo_conf

    with open('flyai_data/relation2idx.json', 'r', encoding='utf-8') as f:
        args.rel2id = json.load(f)
    args.id2rel = dict()
    for key in args.rel2id:
        args.id2rel[args.rel2id[key]] = key
    args.spo_conf = args.rel2id
    #
    args.s2id = {}
    args.s_type = []
    for key, id_ in args.rel2id.items():
        print('key {}, id_ {}'.format(key, id_))
        parts = key.split('_')
        sub_type = parts[0]
        obj_type = parts[2]
        args.s_type.append(sub_type)
        args.s_type.append(obj_type)
    #import sys
    #sys.exit(0)

    #args.s_type = spo_config.SPO_TAG["subject_type"] + spo_config.SPO_TAG["object_type"]
    #args.s_type = [x.split("_")[0] for x in args.s_type]
    i = 1
    args.s_type = list(set(args.s_type))
    args.s_type.sort(key=lambda x: x)
    for st in args.s_type:
        args.s2id[st] = i
        i += 1
    args.R_num = len(args.rel2id)
    args.E_num = len(args.s2id)

    # tokenizer
    args.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

    # Dataset & Dataloader
    train_dataset = mhs_DuIEDataset(args,
                                    examples=read_examples(args, json_file="./flyai_data/fixed_train.json"),
                                    data_type="train")
    eval_dataset = mhs_DuIEDataset(args,
                                   examples=read_examples(args, json_file="./flyai_data/fixed_dev.json"),
                                   data_type="dev")

    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=train_dataset._create_collate_fn(),
                            num_workers=0)

    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=eval_dataset._create_collate_fn(),
                           num_workers=0)

    # model
    model = model_mhs.from_pretrained(args.model_name_or_path,
                                      E_num=args.E_num,
                                      E_em_size=250,
                                      R_num=args.R_num)
    model.load_state_dict(torch.load(args.fine_tunning_model, map_location='cpu'))

    # 训练
    model.eval()
    
    res_dev = evaluate(args, eval_iter, model, mode="eval")
    print("The F1-score is {}".format(res_dev['f1']))
        
if __name__ == "__main__":
    main()



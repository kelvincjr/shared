import json
import argparse
import numpy as np
#from nltk import collections
from collections import Counter
from tqdm import tqdm
#from nltk.translate.bleu_score import sentence_bleu
#from nltk.translate.bleu_score import SmoothingFunction


def read_inputs(input_file):
    references, candidates = [], []
    with open(input_file) as f:
        lines = f.readlines()
        for l in lines:
            data = json.loads(l.strip())
            references.append(data["golden"])
            candidates.append(data["response"])
    return references, candidates



def calculate_metrics(reference, predict, is_smooth=False):
    #-------------------bleu----------
    bleu_1 = bleu(reference, predict, 1, is_smooth)
    bleu_2 = bleu(reference, predict, 2, is_smooth)
    bleu_3 = bleu(reference, predict, 3, is_smooth)
    bleu_4 = bleu(reference, predict, 4, is_smooth)

    # -------------------f1----------
    f1 = f1_score(reference, predict)

    return (bleu_1, bleu_2, bleu_3, bleu_4, f1)


def f1_score(reference, predict):
    reference = list(reference.replace(' ', ''))
    predict = list(predict.replace(' ', ''))
    return compute_f1(reference, predict)

def compute_f1(reference, predict):
    #common = collections.Counter(reference) & collections.Counter(predict)
    common = Counter(reference) & Counter(predict)
    num_same = sum(common.values())
    if len(reference) == 0 or len(predict) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(reference == predict)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(predict)
    recall = 1.0 * num_same / len(reference)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def bleu(reference, predict, n, is_smooth=False):
    if n == 1:
        weights = [1, 0, 0, 0]
    elif n == 2:
        weights = [0.5, 0.5, 0, 0]
    elif n == 3:
        weights = [0.33, 0.33, 0.33, 0]
    elif n == 4:
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        weights = [1, 0, 0, 0]

    if len(predict) < n:
        return 0
    '''
    if is_smooth:
        return sentence_bleu([reference], predict, weights=weights, smoothing_function=SmoothingFunction().method7)
    else:
        return sentence_bleu([reference], predict, weights=weights)
    '''
    return 0


def update_metrics(metrics, results):
    bleu_1, bleu_2, bleu_3, bleu_4, f1 = results
    metrics["BLEU1"].append(bleu_1)
    metrics["BLEU2"].append(bleu_2)
    metrics["BLEU3"].append(bleu_3)
    metrics["BLEU4"].append(bleu_4)
    metrics["F1"].append(f1)

def display_metrics(metrics):
    print("BLEU 1:", np.mean(metrics["BLEU1"]))
    print("BLEU 2:", np.mean(metrics["BLEU2"]))
    print("BLEU 3:", np.mean(metrics["BLEU3"]))
    print("BLEU 4:", np.mean(metrics["BLEU4"]))
    print("F1:", np.mean(metrics["F1"]))

def list_metrics(metrics):
    return {
        'BLEU1': np.mean(metrics['BLEU1']),
        'BLEU2': np.mean(metrics['BLEU2']),
        'BLEU3': np.mean(metrics['BLEU3']),
        'BLEU4': np.mean(metrics['BLEU4']),
    }

def split_evaluate(references, candidates):
    metrics = {"BLEU1": [], "BLEU2": [], "BLEU3": [], "BLEU4": [], "F1": []}
    for reference, candidate in tqdm(zip(references, candidates), desc="Evaluating"):
        results = calculate_metrics(reference, candidate, True)
        update_metrics(metrics, results)
    display_metrics(metrics)

    return list_metrics(metrics)

def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default="./LCCC_beam_generate.txt",
                        help="Input data in json format.")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_argparse()

    references, candidates = read_inputs(args.file)

    split_evaluate(references, candidates)
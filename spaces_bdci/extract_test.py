import numpy as np
import extract_convert_bdci as convert
import extract_vectorize_bdci as vectorize
import extract_model_bdci as extract
from snippets import *

def text_split(text, limited=True):
    """将长句按照标点分割为多个子句。
    """
    #texts = text_segmentate(text, 1, u'\n。；：，')
    texts = text_segmentate(text, 1, u'\n')
    if limited:
        texts = texts[-maxlen:]
    return texts

def predict(text, labels):
    # 抽取
    texts = convert.text_split(text)
    vecs = vectorize.predict(texts)
    extract.model.load_weights('weights/extract_model.%s.weights' %  0)
    preds = extract.model.predict(vecs[None])[0, :, 0]
    sent_num = 0
    print('=== pred_num: {}, label_num: {}'.format(len(preds), len(labels)))
    for pred in preds:
    	label = 0
    	if sent_num in labels:
    		label = 1
    	print('sent_num {}, pred {}, label {}'.format(sent_num, pred, label))
    	sent_num += 1
    preds = np.where(preds > extract.threshold)[0]
    summary = ''.join([texts[i] for i in preds])
    return summary


if __name__ == '__main__':

    from tqdm import tqdm

    data = extract.load_data(extract.data_extract_json)
    valid_data = data #data_split(data, fold, num_folds, 'valid')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(valid_data):
        text = '\n'.join(d[0])
        summary = predict(text, d[1])
        print(summary)
        metrics = compute_metrics(summary, d[2])
        for k, v in metrics.items():
            total_metrics[k] += v

    metrics = {k: v / len(valid_data) for k, v in total_metrics.items()}
    print(metrics)

import codecs
import random
import math
import json
import torch
import pandas as pd
from tqdm import tqdm

def get_test_data(filename):
	data_dict = {}
	with open(filename, encoding='utf-8') as f:
		data_list = json.load(f)
		for data in tqdm(data_list):
   			text = data['text']
   			id_ = data['id']
   			data_dict[text] = id_
	return data_dict

def get_test_result(filename, data_dict):
	D = []
	with open(filename, encoding='utf-8') as f:
		lines = f.readlines()
		for line in tqdm(lines):
			data = json.loads(line)
			text = data['text']
			if text in data_dict:
				id_ = data_dict[text]
			else:
				print('malformed text, {}'.format(text))
			spo_list = data['spo_list']
			d = {'id': id_, 'spo_list': []}
			for spo in spo_list:
				sub_category = spo['subject_type']
				sub_mention = spo['subject']
				obj_category = spo['object_type']
				obj_mention = spo['object']
				predicate = spo['predicate']
				d['spo_list'].append({
            			"subject": sub_mention,
            			'subject-type': sub_category,
                        "predicate": predicate,
                        "object": obj_mention,
                        "object-type": obj_category
                    })
			D.append(d)
	return D

def gen_submission(submission_file, data_list):
	with codecs.open(submission_file, 'w', 'utf-8') as f:
		json_str = json.dumps(data_list, ensure_ascii=False, indent=4)
		f.write(json_str)

if __name__ == "__main__":
	data_dict = get_test_data('emergency_data/train/test.json')
	data_list = get_test_result('output/test_predictions.json', data_dict)
	gen_submission('output/result.json', data_list)
import codecs
import random
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def get_data(filename):
	D = []
	total_count = 0
	fixed_count = 0
	with open(filename, encoding='utf-8') as f:
		data_list = json.load(f)
		for data in tqdm(data_list):
   			text = data['text']
   			id_ = data['id']
   			spo_list = data['spo_list']
   			fixed, change_ent, text = text_fix(id_, text, spo_list)
   			total_count += 1
   			if fixed:
   				fixed_count += 1 
   			d = {'id': id_, 'text': text, 'spo_list': []}
   			for spo in spo_list:
   				sub_category = spo['subject-type']
   				sub_mention = spo['subject']
   				obj_category = spo['object-type']
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
	print('total_count {}, fixed_count {}'.format(total_count, fixed_count))
	return D

def text_fix(id_, text, spo_list):
	debug_print = False
	if id_ >= 10 and id_ < 20:
		debug_print = True
	entity_list = []
	new_spo_list = []
	change_ent = defaultdict(list)
	fixed = True
	for spo_item in spo_list:
		subject_ = spo_item["subject"]
		object_ = spo_item["object"]
		entity_list.append(subject_)
		entity_list.append(object_)
	entity_list = list(set(entity_list))
	missing_entity_list = list()
	for entity in entity_list:
		index = text.find(entity)
		if index == -1:
			missing_entity_list.append(entity)
	sent_endmark = ["!","?",";","。","？","！","；","、"]
	if debug_print:
		print('origin text: ', text)
		print('missing_entity_list: ', missing_entity_list)
	for i in range(len(text)):
		token = text[i]
		found = False
		if token == '、' or token == '和' or token == '及' or token == '或':
			for j in range(len(text) - i - 1):
				if text[i+1+j] in sent_endmark:
					break
				next_span = text[i+1:i+2+j]
				#next_span = next_span #.replace('全', '').replace('等', '')
				for ent1 in missing_entity_list:
					if len(next_span) > 1 and ent1.endswith(next_span) and (len(ent1) != len(next_span)):
						if debug_print:
							print('token {}, index {}'.format(token, i))
							print('entity {}, next_span {}, start_idx {}, end_idx {}'.format(ent1, next_span, i+1, i+2+j))
						found = True
						#change_ent[ent1] = (next_span, i+1, i+2+j)
						change_ent[ent1].append((next_span, len(next_span), i+1, i+2+j))
						break
				for ent1 in missing_entity_list:
					if len(next_span) > 1 and ent1.find(next_span) != -1 and (len(ent1) != len(next_span)):
						if debug_print:
							print('token {}, index {}'.format(token, i))
							print('entity {}, next_span {}, start_idx {}, end_idx {}'.format(ent1, next_span, i+1, i+2+j))
						found = True
						#change_ent[ent1] = (next_span, i+1, i+2+j)
						change_ent[ent1].append((next_span, len(next_span), i+1, i+2+j))
						break
			for k in range(i):
				if text[i-1-k] in sent_endmark:
					break
				pre_span = text[i-1-k:i]
				for ent2 in missing_entity_list:
					if len(pre_span) > 1 and ent2.startswith(pre_span) and (len(ent2) != len(pre_span)):
						if debug_print:
							print('token {}, index {}'.format(token, i))
							print('entity {}, pre_span {}, start_idx {}, end_idx {}'.format(ent2, pre_span, i-1-k, i))
						found = True
						#change_ent[ent2] = (pre_span, i-1-k, i)
						change_ent[ent2].append((pre_span, len(pre_span), i-1-k, i))
						break
			#if found:
				#break
	max_change_ent = {}
	for ent, span_list in change_ent.items():
		sorted_span_list = sorted(span_list, key=lambda x: x[1], reverse=True)
		max_change_ent[ent] = sorted_span_list[0]
	sorted_change_ent = sorted(max_change_ent.items(), key=lambda x: x[1][2], reverse=False)
	if len(sorted_change_ent) != len(missing_entity_list):
		fixed = False
	if debug_print:
		print('sorted_change_ent: ', sorted_change_ent)
		
	fixed_text = ''
	last_end_idx = 0
	ent_num = len(sorted_change_ent)
	if ent_num == 0:
		fixed_text = text
	ent_count = 0
	for ent, (span, span_len, start_idx, end_idx) in sorted_change_ent:
		fixed_text += (text[last_end_idx:start_idx] + ent)
		last_end_idx = end_idx
		if ent_count == (ent_num - 1): 
			fixed_text += text[end_idx:]
		ent_count += 1

	fixed_text = fixed_text.replace(' ', '')
	if debug_print:
		print('id_ {}, fixed text: {}'.format(id_,fixed_text))
		print('=============================================')
      	#import sys
        #sys.exit()
	return fixed, sorted_change_ent, fixed_text

def gen_fixed_file(fixed_file, data_list):
	with codecs.open(fixed_file, 'w', 'utf-8') as f:
		json_str = json.dumps(data_list, ensure_ascii=False, indent=4)
		f.write(json_str)

if __name__ == "__main__":
	data_list = get_data('/opt/kelvin/python/knowledge_graph/ai_contest/hualu/emergency_data/train2/labeled.json')
	gen_fixed_file('../data/fixed_train.json', data_list)
"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import torch
import sys
import pandas as pd

from transformers import BertTokenizerFast
from torch.utils.data import Dataset

tokenizer = BertTokenizerFast.from_pretrained("/kaggle/working")

def find_common_prefix(text1, text2):
    max_common_prefix_len = 0
    if len(text2) > len(text1):
        for index in range(len(text1)):
            if text2[:index+1] == text1[:index+1]:
                max_common_prefix_len += 1
            else:
                break
    elif len(text1) >= len(text2):
        for index in range(len(text2)):
            if text2[:index+1] == text1[:index+1]:
                max_common_prefix_len += 1
            else:
                break
    return max_common_prefix_len

class MrcDataset(Dataset):
    def __init__(self, args, json_path, tokenizer):
        examples = []
        with open(json_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            line_count = 0
            for line in lines:
                if line_count == 0:
                    line_count += 1
                    continue

                id_ = ""
                text = ""
                question = ""
                answer_str = ""
                if line.find('"') > 0:
                    parts = line.strip().split('"')
                    id_ = parts[0].split(',')[0]
                    text = '"' + parts[1] + '"'
                    parts_2 = parts[2].split(',')
                    question = parts_2[1]
                    answer_str = parts_2[2]
                    print('id: [{}], text: [{}], question: [{}], answer_str: [{}]'.format(id_, text, question, answer_str))
                else:
                    parts = line.strip().split(',')
                    id_ = parts[0]
                    text = parts[1]
                    question = parts[2]
                    answer_str = parts[3]

                answers = answer_str.split('||')
                answer_starts = []
                for answer in answers:
                    answer_start = text.find(answer)
                    if answer_start == -1:
                        print('answer: [{}] not found in id: [{}], text: [{}]'.format(answer, id_, text))
                        other_answer_ind = 0
                        suffix = ''
                        found = False
                        for other_answer in answers:
                            if answer != other_answer:
                                max_common_prefix = find_common_prefix(answer, other_answer)
                                if max_common_prefix > 3:
                                    if len(answer_starts) > other_answer_ind:
                                        next_token = text[answer_starts[other_answer_ind] + len(other_answer):answer_starts[other_answer_ind] + len(other_answer)+1]
                                        if next_token == '、' or next_token == '和' or next_token == '及' or next_token == '或':
                                            print('====== found =======')
                                            found = True
                                            suffix = answer[max_common_prefix+1:]
                            other_answer_ind += 1
                        if found:
                            answer_start = text.find(suffix)
                        else:
                            sys.exit()
                    answer_starts.append(answer_start)
                assert len(answers) == len(answer_starts)

                examples.append({
                        "id": int(id_),
                        #"title": title,
                        "context": text,
                        "question": question,
                        "answers": answers,
                        "answer_starts": answer_starts
                    })
                line_count += 1

        print('number of example: {}'.format(len(examples)))
        
        questions_title = [examples[i]['question'] for i in range(len(examples))]
        # title_contexts = [examples[i]['title'] + examples[i]['context'] for i in range(len(examples))]
        contexts = [examples[i]['context'] for i in range(len(examples))]
        tokenized_examples = tokenizer(questions_title,
                                       contexts,
                                       padding="max_length",
                                       max_length=args.max_len,
                                       truncation="only_second",
                                       stride=args.stride,
                                       return_offsets_mapping=True,
                                       return_overflowing_tokens=True)

        df_tmp = pd.DataFrame.from_dict(tokenized_examples, orient="index").T
        tokenized_examples = df_tmp.to_dict(orient="records")

        for i, tokenized_example in enumerate(tokenized_examples):
            input_ids = tokenized_example["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            offsets = tokenized_example['offset_mapping']
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample_mapping']
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']

            # If no answers are given, set the cls_index as answer.
            if len(answer_starts) == 0 or (answer_starts[0] == -1):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_starts[0]
                end_char = start_char + len(answers[0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized_examples[i]["start_positions"] = cls_index
                    tokenized_examples[i]["end_positions"] = cls_index
                    tokenized_examples[i]['answerable_label'] = 0
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples[i]["start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples[i]["end_positions"] = token_end_index + 1
                    tokenized_examples[i]['answerable_label'] = 1

            # evaluate的时候有用
            tokenized_examples[i]["example_id"] = examples[sample_index]['id']
            tokenized_examples[i]["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]

        self.examples = examples
        self.tokenized_examples = tokenized_examples

        for i, example in enumerate(examples): 
            if example['id'] == 317:
            #if i == 2:
                print("id: ", example['id'])
                print("context: ", example['context'])
                print("question: ", example['question'])
                print("answers: ", example['answers'])
                print('answer_starts: ', example['answer_starts'])
                print('token_ids: ', tokenized_examples[i]["input_ids"])
                print('offset_mapping: ', tokenized_examples[i]["offset_mapping"])
                print('overflow_to_sample_mapping: ', tokenized_examples[i]["overflow_to_sample_mapping"])
                print('start_positions: ', tokenized_examples[i]["start_positions"])
                print('end_positions: ', tokenized_examples[i]["end_positions"])
                print('answerable_label: ', tokenized_examples[i]['answerable_label'])

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_answerable_label = torch.tensor([x["answerable_label"] for x in batch])
    all_start_positions = torch.tensor([x["start_positions"] for x in batch])
    all_end_positions = torch.tensor([x["end_positions"] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_start_positions": all_start_positions,
        "all_end_positions": all_end_positions,
        "all_answerable_label": all_answerable_label,
    }


if __name__ == '__main__':
    dataset = MrcDataset(None, "../hualu_data/phase2_A_train.csv", None)
    a = 1

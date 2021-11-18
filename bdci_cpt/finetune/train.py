import math
import os.path
from tqdm import tqdm

import torch.cuda
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import ChitChatDataset
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from modeling_cpt import CPTForConditionalGeneration
from utils.metrics import split_evaluate

class Trainer:
    def __init__(self, model: CPTForConditionalGeneration, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.cls_token_id, self.sep_token_id, self.pad_token_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])

        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size

        self.train_set = ChitChatDataset(self.tokenizer, args, 'train')
        self.dev_set = ChitChatDataset(self.tokenizer, args, 'dev')
        self.test_set = ChitChatDataset(self.tokenizer, args, 'test')
        print('===== ChitChatDataset load done =====')

        self.train_dataloader = self.train_set.get_dataloader(batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.dev_dataloader = self.dev_set.get_dataloader(batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_dataloader = self.test_set.get_dataloader(batch_size=self.batch_size, shuffle=False, num_workers=0)

        print('===== get_dataloader done =====')
        self.optimize_step = 0
        self.backward_step = 0
        self.last_epoch_avg_loss = None
        self.batch_expand_times = self.args.batch_expand_times

        self.total_optimize_step = (math.ceil(len(self.train_set) / self.batch_size) * self.epoch) // self.batch_expand_times

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.num_warmup_steps, num_training_steps=self.total_optimize_step)

        if not os.path.exists("./save"):
            os.mkdir("save")

    def get_training_state(self):
        return {
            "last_lr": self.scheduler.get_last_lr()[0],
            "backward_step": self.backward_step,
            "optimize_step": self.optimize_step,
            "total_optimize_step": self.total_optimize_step,
            "last_epoch_avg_loss": self.last_epoch_avg_loss,
        }

    def save_state_dict(self, filename):
        save_path = os.path.join("./save", filename)
        torch.save({
            "training_state": self.get_training_state(),
            "model": self.model.state_dict(),
        }, save_path)

    def calculate_loss_and_accuracy(self, output, labels):
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)
        loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad_token_id)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        accuracy = correct / num_targets
        loss = loss / num_targets

        return loss, accuracy

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        iterator_bar = tqdm(self.train_dataloader)
        loss_sum = 0.0
        step_num = len(iterator_bar)

        for batch in iterator_bar:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            output = self.model(**batch)
            loss, _ = self.calculate_loss_and_accuracy(output, batch['decoder_input_ids'])

            bar_description = "EPOCH[{}] LOSS[{:.5f}] ".format(epoch, loss.item())
            iterator_bar.set_description(bar_description)

            loss_sum += loss.item()
            loss /= self.batch_expand_times

            loss.backward()
            self.backward_step += 1

            if self.backward_step % self.batch_expand_times == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        avg_loss = loss_sum / step_num
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        pred_result = []
        target = []
        for batch in tqdm(dataloader):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            output = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            for i in range(self.batch_size):
                pred_result.append(self.tokenizer.decode(output[i], skip_special_tokens=True))
                target.append(self.tokenizer.decode(batch['decoder_input_ids'][i], skip_special_tokens=True))
        res = split_evaluate(target, pred_result)
        return sum(i for i in res.values())

    @torch.no_grad()
    def test_epoch(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        pred_result = []
        target = []
        for batch in tqdm(dataloader):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            output = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            for i in range(self.batch_size):
                pred_result.append(self.tokenizer.decode(output[i], skip_special_tokens=True))
        
        id_ = 25001
        output_csv = 'save/submit.csv'
        with codecs.open(output_csv, 'w', 'utf-8') as f:
        	f.write("id|ret"+'\n')
        	for pred_ret in pred_result: 
        		f.write("{}|{}\n".format(id_, pred_ret))
        		id_ += 1

    def train(self):
        min_loss = float('inf')
        max_score = float('-inf')
        print("Start Training")
        for epoch in range(1, self.epoch+1):
            avg_loss = self.train_epoch(epoch)
            self.last_epoch_avg_loss = avg_loss
            print("--- Backward step {}, Current Optimize Step {}, Target Optimize_step {}".format(
                self.backward_step, self.optimize_step, self.total_optimize_step))
            print("--- EPOCH[{}] AVG_LOSS[{:.5f}] LR[{}]".format(epoch, avg_loss, self.scheduler.get_last_lr()[0]))
            avg_score = self.eval_epoch(self.dev_dataloader)
            if avg_loss < min_loss:
                min_loss = avg_loss
            if avg_score > max_score:
                max_score = avg_score
                self.save_state_dict(filename="{}-epoch{}-loss{:.5f}-score{:.5f}.pt".format("cpt_base", epoch, avg_loss, avg_score))
        self.optimizer.zero_grad()

    def test(self):
        if self.args.state_dict is not None:
            self.model.load_state_dict(self.args.state_dict['model'])
        return self.test_epoch(self.test_dataloader)
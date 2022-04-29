import numpy as np
import torch
from torch import nn
from nezha.modeling import attentions

from nezha.modeling.modeling import NeZhaPreTrainedModel, NeZhaModel


def ce_loss_with_probability(pred, target, weight=None, mask=None):
    # pred~(bsz, seq_len)二分类的概率值   target~(bsz, seq_len)取值只能是0或1
    target = target.float()
    pred_1 = (1 - pred).clamp(min=1e-20, max=1)
    pred = pred.clamp(min=1e-20, max=1)
    if weight is None:
        weight0, weight1 = 1, 1
    else:
        weight0, weight1 = weight
    shape_loss = -(1 - target) * torch.log(pred_1) * weight0 - target * torch.log(pred) * weight1
    loss = torch.mean(shape_loss)
    return loss


class Back1NeZhaHierarchyClassification(NeZhaPreTrainedModel):
    #     out_dir = 'data/0114'    0.766126
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 25
        self.bert = NeZhaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.high_merge_labels = [[int(t) for t in labels] for labels in config.high_merge_labels]
        self.high_num = len(config.high_merge_labels)
        self.high_line = nn.Linear(config.hidden_size, self.high_num)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, high_labels=None):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(pooled_out)
        pred_batch_high_probs = torch.sigmoid(self.high_line(pooled_out))
        if self.training:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss2 = ce_loss_with_probability(pred_batch_high_probs, high_labels)
            out = loss + loss2
        else:
            prob = logits.softmax(dim=-1)
            for bsz, pred_high_probs in enumerate(pred_batch_high_probs.tolist()):
                for idx, pred_high_prob in enumerate(pred_high_probs):
                    if pred_high_prob > 0.6:
                        for high_label_idx in self.high_merge_labels[idx]:
                            prob[bsz][high_label_idx] += 0.15
            out = prob
        return out


def init_linear(input_linear, seed=1337):
    """初始化全连接层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    torch.nn.init.uniform_(input_linear.weight, -scope, scope)
    # nn.init.uniform(input_linear.bias, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class AttentionPool(torch.nn.Module):
    # 把序列向量 汇聚成 N维logit输出
    def __init__(self, out_dim, dim=768, n_head=2, dropout=0.1):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(1, out_dim, dim, requires_grad=True), requires_grad=True)
        self.attn = attentions.MultiHeadedAttention(h=n_head, d_model=dim)
        self.line = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None
        init_linear(self.line)

    def forward(self, x):
        # x~(bsz, seq_len, dim)   logit~(bsz, out_dim)
        bsz, seq_len, dim = x.size()
        query = self.query.expand(bsz, -1, -1)  # (bsz, out_dim, dim)
        pool_x = self.attn(query, x, x)  # pool_x~(bsz, out_dim, dim)
        if self.dropout is not None:
            pool_x = self.dropout(pool_x)
        logit = self.line(pool_x)  # ~(bsz, out_dim, 1)
        logit = logit.squeeze(2)  # ~(bsz, out_dim)
        return logit


class NeZhaHierarchyClassification(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 25
        self.bert = NeZhaModel(config)
        self.classifier = AttentionPool(out_dim=self.num_labels, dim=config.hidden_size, n_head=config.num_attention_heads, dropout=0.1)
        self.high_merge_labels = [[int(t) for t in labels] for labels in config.high_merge_labels]
        self.high_num = len(config.high_merge_labels)
        self.high_line = AttentionPool(out_dim=self.high_num, dim=config.hidden_size, n_head=config.num_attention_heads, dropout=0.1)

        self.query = torch.nn.Parameter(torch.rand(1, self.num_labels, config.hidden_size, requires_grad=True), requires_grad=True)
        self.attn = attentions.MultiHeadedAttention(h=config.num_attention_heads, d_model=config.hidden_size)
        self.line = torch.nn.Linear(config.hidden_size, 1)
        self.threshold_prob = 0.6
        self.add_prob = 0.15

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, high_labels=None):
        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(encoder_out)  # (bsz, num_labels)
        # print(f'logits~{logits.size()}')
        pred_batch_high_probs = torch.sigmoid(self.high_line(encoder_out))   # (bsz, high_num)
        # print(f'pred_batch_high_probs~{pred_batch_high_probs.size()}')
        # print(f'pred_batch_high_probs={pred_batch_high_probs}')
        if self.training:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss2 = ce_loss_with_probability(pred_batch_high_probs, high_labels)
            out = loss + loss2
        else:
            prob = logits.softmax(dim=-1)  # (bsz, num_labels)
            # print(f'prob~{prob.size()}')
            
            for bsz, pred_high_probs in enumerate(pred_batch_high_probs.tolist()):
                # print(f'pred_high_probs={pred_high_probs}')
                # print(f'high_probs = ' + ' '.join([f'{t:0.2f}' for t in pred_high_probs]))
                # print(f'old_prob = ' + ' '.join([f'{t:0.2f}' for t in prob[bsz].tolist()]))
                for idx, pred_high_prob in enumerate(pred_high_probs):
                    if pred_high_prob > self.threshold_prob:
                        for high_label_idx in self.high_merge_labels[idx]:
                            prob[bsz][high_label_idx] += self.add_prob
                # print(f'new_prob = ' + ' '.join([f'{t:0.2f}' for t in prob[bsz].tolist()]))
            out = prob
        return out

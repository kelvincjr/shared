import torch
from torch import nn

from nezha.modeling.modeling import NeZhaPreTrainedModel, NeZhaModel


class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 25
        self.bert = NeZhaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(pooled_out)
        outputs = (logits,) + (pooled_out,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

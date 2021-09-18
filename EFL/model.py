import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from loss import focal_loss, ghmc_loss

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AlbertForSequenceClassification)

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}

def compute_loss(pred, label, cmnli_tag=False):
    if cmnli_tag:
        loss = F.cross_entropy(pred.float(), label.long())
    else:
        label = torch.clip(label, 0, 1)
        loss = F.cross_entropy(pred.float(), label.long())
    return loss




class BertMulti(BertForSequenceClassification):

    def __init__(self, args):
        config = BertConfig.from_pretrained(args.model_name_or_path)
        super(BertMulti, self).__init__(config)
        config = BertConfig.from_pretrained(args.model_name_or_path)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=config)
        self.bert_hidden_size = config.hidden_size
        # FC class
        self.task = args.task_name
        self.fc_ocnli = nn.Linear(self.bert_hidden_size, 3)
        # else:
        self.fc_entailment = nn.Linear(self.bert_hidden_size, 2)
        
    def forward(self, **inputs):
        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        labels = inputs['labels']
        outputs, pooled_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ) 
        if self.task in ['ocnli', 'cmnli']:
            outputs = self.fc_ocnli(pooled_outputs)
            loss = compute_loss(outputs, labels, cmnli_tag=True)
        else:
            outputs = self.fc_entailment(pooled_outputs)        # 【B, 3】
            loss = compute_loss(outputs, labels)
        outputs = F.softmax(outputs, dim=1)
        return loss, outputs


class BertMultiTemplate(nn.Module):
    def __int__(self, args):
        super(BertMultiTemplate, self).__init__()
        config = BertConfig.from_pretrained(args.model_name_or_path)
        self.model = BertModel.from_pretrained(args.model_name_or_path, config=config)
        self.bert_hidden_size = config.hidden_size
        # FC class
        self.fc_entailment = nn.Linear(self.bert_hidden_size, 1)

    def forward(self, **inputs):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        labels = inputs['labels']
        B, K, D = input_ids.size()[:3]
        input_ids = input_ids.view(B, K, D)
        attention_mask = attention_mask.view(B, K, D)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        B, D = outputs.size()[:2]
        pooled_output = pooled_output.view(-1, K, D)   # [B, K, D]
        support_hat = self.__squash(self.fc_induction(pooled_output))  # [B, K, D]
        b = torch.zeros(B, K, 1, device=self.current_device, requires_grad=False)  # [B, K, 1]
        for _ in range(self.induction_iters):
            d = F.softmax(b, dim=1)  # [B, K, 1]
            c_hat = torch.mul(d, support_hat).sum(1, keepdims=True)  # [B, 1, D]
            c = self.__squash(c_hat)  # [B, 1, D]
            b = b + torch.mul(support_hat, c).sum(-1, keepdims=True)  # [B, K, 1]
        return out, loss


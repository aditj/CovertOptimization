import torch
from transformers import AlbertModel, BertModel
from transformers import logging
from torch.nn import functional as F
import torch.nn as nn
logging.set_verbosity_error()



class BERTClass(torch.nn.Module):
    def __init__(self,n_classes = 6):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # AlbertModel.from_pretrained('albert-base-v2')
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout(0.1)
        self.l5 = torch.nn.Linear(128, n_classes)
    
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        return output

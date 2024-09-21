## 考虑先做 Diffusion Pretraining, 再在分类模型训练过程中生成新的样本，将新的样本以构建句包特征的形式来进行分类
import torch
import copy
import json

import torch
from info_nce import InfoNCE
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertModel, BertForMaskedLM
from sklearn.metrics import *
from torch.nn import functional as F
from tqdm import tqdm
import time
import random
import pandas as pd
import math

from .CLModel import CLModel
from .Losses import *
from sklearn.cluster import KMeans
import numpy as np
from ..utils import *
from pprint import pprint
from ..CustomDatasets.DiffusionDataset import DifDataset


import torch
import torch.nn as nn

class SelectiveGate(nn.Module):
    """
    This module could calculate the gate value from
    input: tensor: (step, batch, H)
    output: tensor(batch, H)

    so the gate values should be (step,batch)

    then do weighted sum

    """

    def __init__(self, plm_hideen_size):
        super().__init__()

        ##Attention Mechaism can be tried here
        self.selectiveGate = nn.Sequential(
            nn.Linear(plm_hideen_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Initialize linear layers with uniform distribution
        for layer in self.selectiveGate.children():
            if isinstance(layer, nn.Linear):
                # Initialize weight parameters with uniform distribution
                nn.init.uniform_(layer.weight, 1, 1) ## this can also be modified for performance
                # Initialize bias parameters with zeros
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, enhancedSamplesLogits):
        S, B, H = enhancedSamplesLogits.shape
        input_tensor = enhancedSamplesLogits.view(-1, H)  # (S*B,H)
        gateValues = self.selectiveGate(input_tensor)  # (S*B,1)
        input_tensor = input_tensor * gateValues  # (S*B,H)
        weightedRep = input_tensor.view(S, B, H).sum(0)

        return weightedRep


class ContrastiveLearningWithDataAugmentation(nn.Module):
    def __init__(self, Label2Id, diffusionModel=None, PLM_Name='bert-base-chinese', loss_fn=ContrastiveLoss,
                 device='cpu', alpha=0.5, beta=0.1, t=0.025, bag_size=2):
        super().__init__()
        self.loss_fn = loss_fn
        self.device = device
        self.Label2Id = Label2Id
        self.B = None
        self.bag_size = bag_size

        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)

        self.CLSModel = AutoModel.from_pretrained(PLM_Name)


        self.diffusionModel = diffusionModel
        if diffusionModel != None:
            self.CLSModel.embeddings = diffusionModel.get_PLM().bert.embeddings
            self.CLSModel.encoder = diffusionModel.get_PLM().bert.encoder
        self.CLSModel.to(self.device)
        self.diffusionModel.to(self.device)

        self.SelectiveGate = SelectiveGate(self.CLSModel.config.hidden_size).to(device)
        self.classifier = nn.Linear(self.CLSModel.config.hidden_size, len(Label2Id.keys())).to(device)

        self.alpha = alpha
        self.beta = beta
        self.t = t

    def getPLM(self):
        return self.CLSModel

    def BERTWithAugmentation(self, input_ids, attention_mask):
        bert_output = self.CLSModel(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_attentions=True)
        att_score = bert_output.attentions[-1].mean(1)[:, 0, :]  # B,L

        ### This step can apparently affect the performances.
        targetMasks = DifDataset.getTargetMask(att_score, self.bag_size * 2)[:self.bag_size]  # NumStep, B, L; from all zeros to all ones (p.s. the first mask may not be all zeros)

        enhancedSamplesLogits = []
        for step in range(targetMasks.shape[0]):
            diffusionModelOutput = self.diffusionModel({
                'x0_input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_mask': targetMasks[step, :, :]
            }) ['seq']  # B, L

            enhancedSampleLogits = self.CLSModel(
                input_ids=diffusionModelOutput,
                attention_mask=torch.where(
                    (diffusionModelOutput != (torch.ones_like(diffusionModelOutput) * self.tokenizer.pad_token_id)).bool(),
                    torch.ones_like(diffusionModelOutput),
                    torch.zeros_like(diffusionModelOutput)
                )
            ).pooler_output  # B, H

            enhancedSamplesLogits.append(enhancedSampleLogits.unsqueeze(0))

        og_feat = bert_output.pooler_output.unsqueeze(0)
        enhancedSamplesLogits = torch.cat([og_feat]+enhancedSamplesLogits, 0)  # step+1, B , H

        enhancedSamplesLogits = self.SelectiveGate(enhancedSamplesLogits)
        return {
            'logits':enhancedSamplesLogits,
            'og_att_score':att_score
        }

    def forward(self, tokenized_items):

        if self.training:
            if 'domain' in tokenized_items.keys():
                domains = tokenized_items['domain_id']
                self.loss_fn = DACLLoss
            else:
                domains = None

            self.B = tokenized_items['B']

            outputs = self.BERTWithAugmentation(tokenized_items['input_ids'],tokenized_items['attention_mask'])
            pooler_output = outputs['logits']
            att_score = outputs['og_att_score']

            CLS_logits = self.classifier(pooler_output)
            features = pooler_output.view(self.B, -1, self.CLSModel.config.hidden_size)

            loss = self.loss_fn(features=features,
                                logits=CLS_logits,
                                domains=domains,
                                anchor=None,
                                labels=tokenized_items['pn_label_id'],
                                alpha=self.alpha,
                                beta=self.beta,
                                t=self.t)

            return {
                'loss': loss,
                'logits': CLS_logits,
                'attention_score': att_score
            }

        else:
            outputs = self.BERTWithAugmentation(tokenized_items['input_ids'],tokenized_items['attention_mask'])
            pooler_output = outputs['logits']
            att_score = outputs['og_att_score']

            CLS_logits = self.classifier(pooler_output)

            return {
                'loss': nn.CrossEntropyLoss()(CLS_logits, tokenized_items['pn_label_id'].view(-1)),
                'logits': CLS_logits,
                'last_hidden_states': pooler_output,
                'attention_score': att_score
            }


class DifA_Model(CLModel):
    def __init__(self,Label2Id, diffusionModel=None, PLM_Name='bert-base-chinese',
                 device='cpu', alpha=0.5, beta=0.1, t=0.025, bag_size=3):
        super().__init__(Label2Id, PLM_Name, device, alpha, beta, t)
        print(f"Initializing CLModel....")
        self.PRNG = random.getrandbits(20)
        self.PLM_Name = PLM_Name.split('/')[-1]
        self.device = device
        self.alpha, self.beta, self.t = alpha, beta, t

        self.model = ContrastiveLearningWithDataAugmentation(
            Label2Id, diffusionModel=diffusionModel, PLM_Name=PLM_Name, loss_fn=ContrastiveLoss,
            device=device, alpha=alpha, beta=beta, t=t, bag_size=bag_size
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-6, weight_decay=0.01)
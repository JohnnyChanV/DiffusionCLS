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
from .Losses import *
from sklearn.cluster import KMeans
import numpy as np
from ..utils import *
from pprint import pprint

class Diffusion(nn.Module):

    def __init__(self, PLM=None, PLM_Name='bert-base-chinese', loss_fn=nn.CrossEntropyLoss(reduction='none'), device='cpu',predThreshold = 0, predMaxStep = 1):
        print(f"Initializing Diffusion BERT with PLM: {PLM_Name}")
        super().__init__()
        self.predThreshold = predThreshold
        self.predMaxStep = predMaxStep
        self.loss_fn = loss_fn
        self.device = device

        self.MaskedLM = BertForMaskedLM.from_pretrained(PLM_Name)
        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)

        if PLM!=None:
            self.MaskedLM.bert.embeddings = PLM.embeddings
            self.MaskedLM.bert.encoder = PLM.encoder
        self.MaskedLM.to(self.device)

    def get_PLM(self):
        return self.MaskedLM.bert

    def forward(self, dataItem):

        if self.training:
            x0_input_ids, attention_mask, target_mask = dataItem['x0_input_ids'],dataItem['attention_mask'],dataItem['target_mask']

            corrupted_input_ids = torch.ones_like(x0_input_ids) * self.tokenizer.mask_token_id

            step_input_ids = torch.where(target_mask.bool(),corrupted_input_ids,x0_input_ids)

            LM_outputs = self.MaskedLM(input_ids = step_input_ids,attention_mask=attention_mask)

            loss = (self.loss_fn(LM_outputs.logits.view(-1,self.MaskedLM.config.vocab_size),x0_input_ids.view(-1)) * target_mask.view(-1)).mean()

            return {
                'x0_input_ids':x0_input_ids,
                'attention_mask':attention_mask,
                'seq':torch.argmax(LM_outputs.logits,-1),
                'outputs':LM_outputs,
                'loss':loss
            }

        else:
            x0_input_ids, attention_mask, target_mask = dataItem['x0_input_ids'],dataItem['attention_mask'],dataItem['target_mask']

            corrupted_input_ids = torch.ones_like(x0_input_ids) * self.tokenizer.mask_token_id

            step_input_ids = torch.where(target_mask.bool(),corrupted_input_ids,x0_input_ids) if target_mask != None else corrupted_input_ids

            current_input_ids = copy.deepcopy(step_input_ids)
            remain_mask_num = (current_input_ids == corrupted_input_ids).int().sum()
            assert remain_mask_num !=0, "at least one mask!"
            current_step = 0
            while remain_mask_num != 0:
                current_step += 1
                outputs = self.MaskedLM(input_ids=current_input_ids, attention_mask=attention_mask)
                step_max_items = torch.max(F.softmax(outputs.logits,dim=-1),dim=-1)

                step_ids = step_max_items.indices # B, L
                step_probs = step_max_items.values # B, L

                predThresholds = torch.ones_like(step_ids) * self.predThreshold if current_step != self.predMaxStep else torch.zeros_like(step_ids)
                current_input_ids = torch.where(step_probs > predThresholds,
                                                step_ids,
                                                current_input_ids) # 这个操作允许模型修改已生成的内容, 考虑加入temperature?


                remain_mask_num = (current_input_ids == corrupted_input_ids).int().sum()

            # print(current_step)

            seq = current_input_ids # B, L



            return {
                'step_input_ids':step_input_ids,
                'x0_input_ids': x0_input_ids,
                'attention_mask': attention_mask,
                'seq':seq,
                'outputs':outputs
            }

class DiffusionModel:
    def __init__(self, PLM, PLM_Name='bert-base-chinese', device='cpu'):
        print(f"Initializing CLModel....")
        self.PRNG = random.getrandbits(20)
        self.PLM_Name = PLM_Name.split('/')[-1]
        self.device = device

        self.model = Diffusion(PLM, PLM_Name=PLM_Name, device=device).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-6, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min((step + 1) / 200, 1.0)  # Warmup steps here
        )

    def train(self, train_loader, eval_loader, epochs=1):
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(train_loader) as train_loader:
                for idx,batch in enumerate(train_loader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    batch['x0_input_ids'] = batch['x0_input_ids'].to(self.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.device)
                    batch['target_mask'] = batch['target_mask'].to(self.device)

                    outputs = self.model(batch)
                    loss = outputs['loss']
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    total_loss += loss.item()
                    train_loader.set_description(f"Epoch:{epoch}/{epochs}, Mean Loss:{(total_loss / (idx+1)):.4f}")
                self.evaluate(eval_loader)




    def evaluate(self, eval_loader,is_save=False):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Initialize tqdm with description
        pbar = tqdm(eval_loader, desc="Evaluating")

        SampleInputs = []
        SampleGeneration = []
        SampleX0s = []
        with torch.no_grad():
            for batch in pbar:
                batch['x0_input_ids'] = batch['x0_input_ids'].to(self.device)
                batch['attention_mask'] = batch['attention_mask'].to(self.device)
                batch['target_mask'] = batch['target_mask'].to(self.device)

                outputs = self.model(batch)

                logits = outputs['outputs'].logits
                x0_input_ids = outputs['x0_input_ids']
                step_input_ids = outputs['step_input_ids']
                generatedSeq = outputs['seq']

                # Calculate cross entropy loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), generatedSeq.view(-1), ignore_index=self.model.tokenizer.pad_token_id)
                total_loss += loss.item()

                # Count total tokens for normalization
                total_tokens += batch['target_mask'].sum().item()

                # Update tqdm description with current loss
                pbar.set_description(f"Loss: {loss.item():.4f}")

                SampleX0s += (self.model.tokenizer.batch_decode(x0_input_ids,skip_special_tokens=True))
                SampleInputs += (self.model.tokenizer.batch_decode(step_input_ids,skip_special_tokens=True))
                SampleGeneration += (self.model.tokenizer.batch_decode(generatedSeq,skip_special_tokens=True))

        # Calculate perplexity
        perplexity = math.exp(total_loss / total_tokens)
        if is_save:
            pd.DataFrame.from_dict({
                'SampleX0s':SampleX0s,
                'SampleInputs':SampleInputs,
                'SampleGeneration':SampleGeneration,
            }).to_csv('generated.csv')


        print(f"Current PPL: {perplexity}")
        return perplexity


    def get_PLM(self):
        return self.model.get_PLM()



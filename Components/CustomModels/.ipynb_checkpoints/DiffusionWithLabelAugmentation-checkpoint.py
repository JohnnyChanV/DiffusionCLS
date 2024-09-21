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
from ..CustomDatasets.DiffusionWithLabelDataset import DifLabelDataset


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

    def __init__(self, plm_hideen_size,device):
        super().__init__()

        self.device = device

        ##Attention Mechaism can be tried here
        self.selectiveGate = nn.Sequential(
            nn.Linear(plm_hideen_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.mapping_layer = nn.Sequential(
            nn.Linear(plm_hideen_size,plm_hideen_size),
        )

        # Initialize linear layers with uniform distribution
        for layer in self.selectiveGate.children():
            if isinstance(layer, nn.Linear):
                # Initialize weight parameters with uniform distribution
                nn.init.uniform_(layer.weight, 1, 1) ## this can also be modified for performance
                # Initialize bias parameters with zeros
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, OGLogits,enhancedSamplesLogits):
        S, B, H = enhancedSamplesLogits.shape
        input_tensor = enhancedSamplesLogits.view(-1, H)  # (S*B,H)
        # gateValues = self.selectiveGate(input_tensor)  # (S*B,1)
        # input_tensor = input_tensor * gateValues  # (S*B,H)
        weightedRep = input_tensor.view(S, B, H).mean(0)

        weightedRep = self.mapping_layer(weightedRep)

        return weightedRep


class SimilaritySelection(nn.Module):
    def __init__(self, plm_hideen_size,device):
        super().__init__()
        self.device = device

        self.mapping_layer = nn.Sequential(
            nn.Linear(plm_hideen_size,plm_hideen_size),
        )

    def forward(self,OGLogits,enhancedSamplesLogits):
        """
        Args:
            OGLogits: tensor(B, H)
            enhancedSamplesLogits: tensor(S,B,H)

        Returns:

        """
        S, B, H = enhancedSamplesLogits.shape
        input_tensor = enhancedSamplesLogits.view(-1, H)  # (S*B,H)

        similarities = sim(input_tensor,OGLogits).view(S,B,B)  # kh,qh->kq;  (S*B,H),(B,H) -> (S*B,B) -> (S,B,B)
        similarities = (similarities * torch.eye(B).to(self.device)).sum(-1)  # (S,B,B) -> (S,B)
        similarities =  similarities.view(-1).unsqueeze(-1) # (S,B) -> (S*B) -> (S*B, 1)
        # similarities = F.softmax(similarities,0).view(-1).unsqueeze(-1) # (S,B) -> (S*B) -> (S*B, 1)


        input_tensor = input_tensor * similarities
        weightedRep = input_tensor.view(S, B, H).mean(0)
        weightedRep = self.mapping_layer(weightedRep)

        return weightedRep

class ContrastiveLearningWithDataAugmentation(nn.Module):
    def init_balance_module(self):
        self.prob_being_selected = {}
        self.balance_remaining = {}
        for label in self.balance_info.keys():
            self.prob_being_selected[self.Label2Id[label]] = self.balance_info[label] / sum(self.balance_info.values())
            self.balance_remaining[self.Label2Id[label]] = self.balance_info[label]
        return

    def __init__(self, Label2Id, step_group_num, max_step, balance_info = None, diffusionModel=None, PLM_Name='bert-base-chinese', loss_fn=ContrastiveLoss,
                 device='cpu', alpha=0.5, beta=0.1, t=0.025, bag_size=3, selection_mode = 'similarity', ssl_mode = False):
        super().__init__()
        self.loss_fn = loss_fn
        self.device = device
        self.Label2Id = Label2Id
        self.B = None
        self.bag_size = bag_size
        self.step_group_num = step_group_num
        self.max_step = max_step


        self.balance_info = balance_info
        if self.balance_info!=None:
            self.init_balance_module()

        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)


        self.diffusionModel = diffusionModel
        self.diffusionModel.eval()

        self.CLSModel = AutoModel.from_pretrained(PLM_Name).to(self.device)
        if ssl_mode:
            self.CLSModel.embeddings = copy.deepcopy(diffusionModel.get_PLM().embeddings)
            self.CLSModel.encoder = copy.deepcopy(diffusionModel.get_PLM().encoder)

        if selection_mode == 'similarity':
            self.Selection = SimilaritySelection(self.CLSModel.config.hidden_size,device).to(device)
        else:
            self.Selection = SelectiveGate(self.CLSModel.config.hidden_size,device).to(device)

        self.classifier = nn.Linear(self.CLSModel.config.hidden_size, len(Label2Id.keys())).to(device)

        self.alpha = alpha
        self.beta = beta
        self.t = t

    def get_balance_remaining(self):
        return self.balance_remaining

    def getPLM(self):
        return self.CLSModel

    def BalanceSampling(self,enhanced_pred_logits_flatten, labels_repeated):
        selection_probs = torch.zeros_like(labels_repeated)
        for i,l in enumerate(labels_repeated):
            selection_probs[i] = self.prob_being_selected[int(l)]

        temp_selected_label_indices = torch.where(
            torch.randn_like(labels_repeated.float()) < selection_probs,
            labels_repeated,
            -torch.ones_like(labels_repeated)
        )

        temp_selected_label_indices = torch.nonzero(temp_selected_label_indices >= 0).view(-1).tolist()
        label_list = labels_repeated.tolist()
        selected_label_indices = []
        for idx in temp_selected_label_indices:
            if self.balance_remaining[label_list[idx]] > 0:
                selected_label_indices.append(idx)
                self.balance_remaining[label_list[idx]] -= 1
        if len(selected_label_indices)!=0:
            loss = nn.CrossEntropyLoss()(enhanced_pred_logits_flatten[selected_label_indices], labels_repeated[selected_label_indices])
        else:
            loss = 0
        return loss


    def BERTWithAugmentation(self, input_ids, attention_mask, labels):
        """
        labels: torch.tensor(B,NumSamplePerItem)
        """

        loss = 0
        labels = labels.view(-1)
        bert_output = self.CLSModel(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_attentions=True)
        att_score = bert_output.attentions[-1].mean(1)[:, 0, :]  # B,L

        ### This step can apparently affect the performances.
        # targetMasks = DifLabelDataset.getTargetMask_ForAug(att_score, self.bag_size)
        targetMasks = DifLabelDataset.getTargetMask_WithRange(att_score, step_group_num=self.step_group_num,bag_size=self.bag_size,max_step=self.max_step)
        S, B, L = targetMasks.shape
        labelMask = torch.zeros((S,B,3)).to(targetMasks.device)
        targetMasks = torch.cat([labelMask,targetMasks],-1)


        ## construct diffusion model input
        label_list = labels.tolist()
        label_list = [self.diffusionModel.id2LabelToken[label] for label in label_list]
        x0_input_texts = self.diffusionModel.tokenizer.batch_decode(input_ids,skip_special_tokens=True)
        tokenized_dif_items = self.diffusionModel.tokenizer([[label_list[index], x0_input_texts[index]] for index in range(len(x0_input_texts))],
                                         return_tensors='pt', padding='max_length', truncation=True,
                                         max_length=self.diffusionModel.max_length + 3)

        # print(self.diffusionModel.tokenizer.batch_decode(tokenized_dif_items.input_ids,skip_special_tokens=True))


        enhancedSamplesLogits = []
        with torch.no_grad():
            for step in range(targetMasks.shape[0]):
                if self.balance_info != None and sum(self.balance_remaining.values()) == 0:
                    continue
                diffusionModelOutput = self.diffusionModel({
                    'x0_input_ids': tokenized_dif_items.input_ids.to(self.device),
                    'attention_mask': tokenized_dif_items.attention_mask.to(self.device),
                    'target_mask': targetMasks[step, :, :]
                }) ['seq']  # B, L
                # print("INPUT:")
                # print(self.tokenizer.batch_decode(input_ids,skip_special_tokens=True))
                # print("GENERATED:")
                # print(self.tokenizer.batch_decode(diffusionModelOutput,skip_special_tokens=True))

                if self.tokenizer.cls_token_id!=None:
                    diffusionModelOutput = torch.cat([
                        torch.ones((diffusionModelOutput.shape[0],1)).to(self.device) * self.tokenizer.cls_token_id,
                        diffusionModelOutput[:, 3:]
                    ],-1).int()
                else:
                    diffusionModelOutput = diffusionModelOutput[:, 2:].int()
                enhancedSampleLogits = self.CLSModel(
                    input_ids=diffusionModelOutput,
                    attention_mask=torch.where(
                        (diffusionModelOutput != (torch.ones_like(diffusionModelOutput) * self.tokenizer.pad_token_id)).bool(),
                        torch.ones_like(diffusionModelOutput),
                        torch.zeros_like(diffusionModelOutput)
                    )
                ).pooler_output  # B, H

                enhancedSamplesLogits.append(enhancedSampleLogits.unsqueeze(0))

        og_feat = bert_output.pooler_output



        # Supervised Learning
        enhancedSamplesFeats = torch.cat(enhancedSamplesLogits,0) if len(enhancedSamplesLogits)!=0 else None# (Step,B,H)
        enhanced_pred_logits = self.classifier(enhancedSamplesFeats) if enhancedSamplesFeats!=None else None# (Step,B,NumOfClass)

        # (B,Step,NumOfClass) -> # (B*Step,NumOfClass)
        enhanced_pred_logits_flatten = enhanced_pred_logits.transpose(1,0).contiguous().view(-1,enhanced_pred_logits.shape[-1]) if enhanced_pred_logits!=None else None

        # labels.view(-1):(B) -> (B,1) -> (B,Step) -> (B*Step)
        labels_repeated = labels.view(-1).unsqueeze(1).repeat(1, enhanced_pred_logits.shape[0]).view(-1) if enhanced_pred_logits_flatten!=None else None

        # Balance Sampling...
        if self.balance_info == None and enhanced_pred_logits_flatten!=None:
            loss = nn.CrossEntropyLoss()(enhanced_pred_logits_flatten, labels_repeated)
        elif enhanced_pred_logits_flatten!=None:
            loss = self.BalanceSampling(enhanced_pred_logits_flatten, labels_repeated)

        # loss = sum(nn.CrossEntropyLoss()(logit, labels.view(-1)) for logit in enhanced_pred_logits) / len(enhanced_pred_logits)


        return {
            'logits':og_feat,
            'og_att_score':att_score,
            'loss':loss,
            'enhanced_logits':enhancedSamplesFeats
        }

    def forward(self, tokenized_items):

        if self.training:
            if 'domain' in tokenized_items.keys():
                domains = tokenized_items['domain_id']
                self.loss_fn = DACLLoss
            else:
                domains = None

            self.B = tokenized_items['B']
            labels = tokenized_items['pn_label_id']

            outputs = self.BERTWithAugmentation(tokenized_items['input_ids'],tokenized_items['attention_mask'],labels)
            pooler_output = outputs['logits']
            att_score = outputs['og_att_score']
            enhanced_samples_loss = outputs['loss']

            CLS_logits = self.classifier(pooler_output)
            features = pooler_output.view(self.B, -1, self.CLSModel.config.hidden_size)

            # enhanced_logits = outputs['enhanced_logits'].mean(0).view(self.B, -1, self.CLSModel.config.hidden_size)[:, 0, :]

            loss = self.loss_fn(features=features,
                                logits=CLS_logits,
                                domains=domains,
                                anchor=None,
                                labels=labels,
                                alpha=self.alpha,
                                beta=self.beta,
                                t=self.t)

            loss += enhanced_samples_loss

            return {
                'loss': loss,
                'logits': CLS_logits,
                'attention_score': att_score
            }

        else:
            # outputs = self.BERTWithAugmentation(tokenized_items['input_ids'],tokenized_items['attention_mask'])
            # pooler_output = outputs['logits']
            # att_score = outputs['og_att_score']
            bert_output = self.CLSModel(input_ids=tokenized_items['input_ids'],
                                   attention_mask=tokenized_items['attention_mask'],
                                   output_attentions=True)

            pooler_output = bert_output.pooler_output  # B * NumSample, H
            att_score = bert_output.attentions[-1].mean(1)[:, 0, :]

            CLS_logits = self.classifier(pooler_output)
            return {
                'loss': nn.CrossEntropyLoss()(CLS_logits, tokenized_items['pn_label_id'].view(-1)),
                'logits': CLS_logits,
                'last_hidden_states': pooler_output,
                'attention_score': att_score
            }


class DifA_Model:
    def __init__(self,Label2Id, step_group_num, max_step, balance_info = None, diffusionModel=None, PLM_Name='bert-base-chinese',
                 device='cpu', alpha=0.5, beta=0.1, t=0.02, bag_size=3, selection_mode = 'similarity',ssl_mode=False):
        print(f"Initializing DifA_Model....")
        self.PRNG = random.getrandbits(20)
        self.PLM_Name = PLM_Name.split('/')[-1]
        self.device = device
        self.alpha, self.beta, self.t = alpha, beta, t
        self.balance_info = balance_info

        self.model = ContrastiveLearningWithDataAugmentation(
            Label2Id=Label2Id, balance_info=balance_info, step_group_num=step_group_num, max_step = max_step, diffusionModel=diffusionModel, PLM_Name=PLM_Name, loss_fn=ContrastiveLoss,
            device=device, alpha=alpha, beta=beta, t=t, bag_size=bag_size , selection_mode = selection_mode, ssl_mode=ssl_mode
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-5, weight_decay=0.01)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer, lambda step: min((step + 1) / 200, 1.0)  # Warmup steps here
        # )

        self.best_report = None
        self.best_f1 = -100


    def train(self, train_loader, eval_loader, epochs=1, eval_gap=0):
        self.model.train()
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            train_loader = (train_loader)
            if self.model.balance_info!=None:
                self.model.init_balance_module()
            for index, batch in enumerate(train_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                # print(self.model.tokenizer.tokenize(self.model.tokenizer.decode(batch['input_ids'][0])),'\n',outputs['attention_score'][0])
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()  # Adjust learning rate
                running_loss += loss.item()
                # if self.balance_info==None:
                #     train_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {round(running_loss / (index + 1),4)}")
                # else:
                #     train_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {round(running_loss / (index + 1),4)}, balance_info: {self.model.get_balance_remaining()}")
            if epoch % eval_gap ==0:
                self.evaluate(eval_loader)
                # torch.save(self.model.state_dict(),
                #            f"ckpt/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-Epoch-{epoch}-(alpha:{self.alpha}-beta:{self.beta}-t:{self.t}).ckpt")
                self.model.train()

    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0.0

        results = {
            'true_labels': [],
            'predicted_labels': [],
            'contents': [],
        }

        pooler_output = []

        with torch.no_grad():
            eval_loader = tqdm(eval_loader)
            for index, batch in enumerate(eval_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass
                outputs = self.model(batch)
                loss = outputs['loss']
                total_loss += loss.item()

                # Calculate predictions
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                results['true_labels'].extend(batch['pn_label_id'].view(-1).cpu().numpy())
                results['predicted_labels'].extend(preds.cpu().numpy())
                results['contents'] += batch['contents']
                pooler_output.append(outputs['last_hidden_states'].cpu())
                eval_loader.set_description(f"Evaluation Loss: {total_loss / (index + 1)}")

        # Generate classification report for each task
        true_labels = results['true_labels']
        predicted_labels = results['predicted_labels']
        report = classification_report(true_labels, predicted_labels, digits=6)

        print(report)
        pd.DataFrame.from_dict(results).to_csv(
            f"results/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-(alpha:{self.alpha}-beta:{self.beta}-t:{self.t}).csv",
            encoding='utf-8-sig')
        torch.save(torch.cat(pooler_output, 0),
                   f"eval_pt/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-(alpha:{self.alpha}-beta:{self.beta}-t:{self.t}).pt")

        current_f1 = f1_score(true_labels, predicted_labels, average='macro')
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_report = report
        return self.best_report if self.best_report!=None else None
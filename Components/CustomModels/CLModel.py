import torch
from info_nce import InfoNCE
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertModel
from sklearn.metrics import *
from torch.nn import functional as F
from tqdm import tqdm
import time
import random
import pandas as pd
from .Losses import *
from sklearn.cluster import KMeans
import numpy as np
from ..utils import *


class ContrastiveLearning(nn.Module):

    def __init__(self, Label2Id, PLM=None, PLM_Name='bert-base-chinese', loss_fn=ContrastiveLoss, device='cpu', alpha=0.5, beta=0.1, t=0.025):
        super().__init__()
        self.loss_fn = loss_fn
        self.device = device
        self.Label2Id = Label2Id
        self.B = None

        self.PLM = AutoModel.from_pretrained(PLM_Name)
        if PLM!=None:
            self.PLM.embeddings = PLM.embeddings
            self.PLM.encoder = PLM.encoder
        self.PLM.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)

        self.classifier = nn.Linear(self.PLM.config.hidden_size, len(Label2Id.keys()))

        self.anchors = None
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def getPLM(self):
        return self.PLM

    def forward(self, tokenized_items):

        if self.training:
            if 'domain' in tokenized_items.keys():
                domains = tokenized_items['domain_id']
                self.loss_fn = DACLLoss
            else:
                domains = None

            self.B = tokenized_items['B']

            bert_output = self.PLM(input_ids=tokenized_items['input_ids'],
                                    attention_mask=tokenized_items['attention_mask'],
                                    output_attentions=True)

            pooler_output = bert_output.pooler_output # B * NumSample, H
            # print(sim(pooler_output.unsqueeze(1),bert_output.last_hidden_state).shape)
            att_score = bert_output.attentions[-1].mean(1)[:, 0, :]



            CLS_logits = self.classifier(pooler_output)
            features = pooler_output.view(self.B, -1, self.PLM.config.hidden_size)

            if self.anchors == None:
                anchor_features = None
            else:
                anchor_ids = [item[0] for item in tokenized_items['pn_label_id']]
                anchor_features = torch.cat([self.anchors[id_.cpu().int().tolist()] for id_ in anchor_ids]).to(
                    self.device)

            loss = self.loss_fn(features=features,
                                logits=CLS_logits,
                                domains=domains,
                                anchor=anchor_features,
                                labels=tokenized_items['pn_label_id'],
                                alpha=self.alpha,
                                beta=self.beta,
                                t=self.t)

            return {
                'loss': loss,
                'logits': CLS_logits,
                'attention_score':att_score
            }

        else:
            bert_output = self.PLM(input_ids=tokenized_items['input_ids'],
                                   attention_mask=tokenized_items['attention_mask'],
                                   output_attentions=True)

            pooler_output = bert_output.pooler_output  # B * NumSample, H
            att_score = bert_output.attentions[-1].mean(1)[:, 0, :]

            CLS_logits = self.classifier(pooler_output)

            return {
                'loss': nn.CrossEntropyLoss()(CLS_logits, tokenized_items['pn_label_id'].view(-1)),
                'logits': CLS_logits,
                'last_hidden_states': pooler_output,
                'attention_score':att_score
            }

    def setAnchors(self, anchors):
        self.anchors = anchors
        return





class CLModel:
    def __init__(self, Label2Id, PLM=None, PLM_Name='bert-base-chinese', device='cpu', alpha=0.5, beta=0.3, t=0.025):
        print(f"Initializing CLModel....")
        self.PRNG = random.getrandbits(20)
        self.PLM_Name = PLM_Name.split('/')[-1]
        self.device = device
        self.alpha, self.beta, self.t = alpha, beta, t

        self.model = ContrastiveLearning(Label2Id=Label2Id, PLM=PLM, PLM_Name=PLM_Name, device=device, alpha=alpha, beta=beta, t=t).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-6, weight_decay=0.01)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer, lambda step: min((step + 1) / 200, 1.0)  # Warmup steps here
        # )

        self.best_report = None
        self.best_f1 = -100


    def train(self, train_loader, eval_loader, epochs=1, eval_gap = 1):
        self.model.train()
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            train_loader = tqdm(train_loader)
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
                train_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (index + 1)}")
            if epoch % eval_gap == 0:
                self.evaluate(eval_loader)
                self.model.train()
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
        # pd.DataFrame.from_dict(results).to_csv(
        #     f"results/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-(alpha:{self.alpha}-beta:{self.beta}-t:{self.t}).csv",
        #     encoding='utf-8-sig')
        # torch.save(torch.cat(pooler_output, 0),
        #            f"eval_pt/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-(alpha:{self.alpha}-beta:{self.beta}-t:{self.t}).pt")
        current_f1 = f1_score(true_labels, predicted_labels, average='macro')
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_report = report

        return self.best_report if self.best_report!=None else None

    def getAnchors(self, loader):
        self.model.eval()

        results = {
            'true_labels': [],
            'features': []
        }

        with torch.no_grad():
            eval_loader = tqdm(loader)
            for index, batch in enumerate(eval_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass
                outputs = self.model(batch)
                results['true_labels'].extend(batch['pn_label_id'].view(-1).cpu().numpy())
                results['features'].append(outputs['last_hidden_states'].cpu())

        results['features'] = torch.cat(results['features'], 0)
        anchors = dict(enumerate([[] for _ in range(len(self.model.Label2Id.keys()))]))

        for index, true_label in enumerate(results['true_labels']):
            anchors[true_label].append(results['features'][index].unsqueeze(0))

        for label in anchors.keys():
            anchors[label] = torch.cat(anchors[label], 0).mean(0).unsqueeze(0)

        return anchors

    def setAnchors(self, anchors):
        self.model.setAnchors(anchors)
        return

    def getClassAnchors(self, loader):
        self.model.eval()

        results = {
            'true_labels': [],
            'features': [],
            'contents': []
        }

        with torch.no_grad():
            eval_loader = tqdm(loader)
            for index, batch in enumerate(eval_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass
                outputs = self.model(batch)
                results['true_labels'].extend(batch['pn_label_id'].view(-1).cpu().numpy())
                results['features'].append(outputs['last_hidden_states'].cpu())
                results['contents'] += batch['contents']

        results['features'] = torch.cat(results['features'], 0)
        anchors = dict(enumerate([[] for _ in range(len(self.model.Label2Id.keys()))]))
        contents = dict(enumerate([[] for _ in range(len(self.model.Label2Id.keys()))]))

        for index, true_label in enumerate(results['true_labels']):
            anchors[true_label].append(results['features'][index].unsqueeze(0))
            contents[true_label].append(results['contents'][index])

        for label in anchors.keys():
            anchors[label] = torch.cat(anchors[label], 0).mean(0)
        return anchors, contents

    def getfeatureOnClassAndContentsOnClass(self,loader):
        self.model.eval()

        results = {
            'true_labels': [],
            'features': [],
            'contents': []
        }

        with torch.no_grad():
            eval_loader = tqdm(loader)
            for index, batch in enumerate(eval_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass
                outputs = self.model(batch)
                results['true_labels'].extend(batch['pn_label_id'].view(-1).cpu().numpy())
                results['features'].append(outputs['last_hidden_states'].cpu())
                results['contents'] += batch['contents']

        results['features'] = torch.cat(results['features'], 0)
        anchors = dict(enumerate([[] for _ in range(len(self.model.Label2Id.keys()))]))
        contents = dict(enumerate([[] for _ in range(len(self.model.Label2Id.keys()))]))

        for index, true_label in enumerate(results['true_labels']):
            anchors[true_label].append(results['features'][index].unsqueeze(0))
            contents[true_label].append(results['contents'][index])

        for label in anchors.keys():
            anchors[label] = torch.cat(anchors[label], 0)

        return anchors, contents

    def getAttScores(self,loader):
        self.model.eval()

        attention_scores = []

        with torch.no_grad():
            eval_loader = tqdm(loader)
            for index, batch in enumerate(eval_loader):
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass
                outputs = self.model(batch)
                attention_scores.append(outputs['attention_score'])
        return torch.cat(attention_scores,0)


    def getPLM(self):
        return self.model.getPLM()

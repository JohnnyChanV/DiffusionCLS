import torch
from info_nce import InfoNCE
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertModel
from sklearn.metrics import classification_report
from torch.nn import functional as F
from tqdm import tqdm
import time
import random
import pandas as pd
import numpy as np

def ContrastiveLoss(features, logits, labels, alpha=0.3,beta=0.3,t=0.02,anchor=None,domains=None, CLS_loss_fn=nn.CrossEntropyLoss(),
                    CL_loss_fn=None):
    """

    :param features: torch.tensor(B,NumSamplePerItem, H)
    :param logits: torch.tensor(B*NumSamplePerItem, NumClass)
    :param anchor: torch.tensor(B, H)
    :param labels: torch.tensor(B*NumSamplePerItem)
    :param CLS_loss_fn: loss function for classification loss
    :return: loss
    """
    classification_loss = CLS_loss_fn(logits, labels.view(-1))
    CL_loss_fn = InfoNCE(temperature=t, negative_mode='paired')


    query_feat = features[:, 0, :]
    anchor = query_feat if anchor == None else anchor
    neg_feat = features[:, 1:, :]

    contrastive_loss = CL_loss_fn(
        query=query_feat,
        positive_key=anchor,
        negative_keys=neg_feat
    )

    # return  classification_loss + contrastive_loss
    return classification_loss
    # return alpha * classification_loss + (1 - alpha) * contrastive_loss


def DALoss(features,domains,
           general_domain_id = 0,specific_domain_id=1,t=1.5):
    """
    :param features:  torch.tensor(B, H)
    :param domains:  torch.tensor(B*NumSamplePerItem)
    :return:
    """
    general_domain_features = []
    specific_domain_features = []

    for index,domain_id in enumerate(domains.view(-1).tolist()):
        if domain_id == general_domain_id:
            general_domain_features.append(features[index].unsqueeze(0))
        else:
            specific_domain_features.append(features[index].unsqueeze(0))

    if len(general_domain_features)==0 or len(specific_domain_features)==0:
        return 0
    all_domain_features = F.normalize(torch.cat(specific_domain_features+general_domain_features,0), dim=-1)#(Q+K,H)
    general_domain_features = F.normalize(torch.cat(general_domain_features,0), dim=-1)#(Q,H)
    specific_domain_features = F.normalize(torch.cat(specific_domain_features,0), dim=-1) #(K,H)

    # i is domain-specific

    similarities_kq = torch.exp(
        (torch.einsum('kh,qh->kq',specific_domain_features,general_domain_features) / t)
    )# (K,Q)

    similarities_ik = torch.exp(
        (torch.einsum('kh,qh->kq',specific_domain_features,all_domain_features) / t)
    ).sum(-1).unsqueeze(-1)# (K,K+Q) -> (K,1)


    L_di = - (torch.log(similarities_kq / similarities_ik).mean(0)) # (Q,K) -> (K)

    loss = L_di.mean()

    return loss



def DACLLoss(features, logits, labels, domains, alpha=0.3,beta=0.3,t=1.5,anchor=None, CLS_loss_fn=nn.CrossEntropyLoss(),
                    CL_loss_fn=InfoNCE(temperature=0.025, negative_mode='paired')):
    """

    :param features: torch.tensor(B,NumSamplePerItem, H)
    :param logits: torch.tensor(B*NumSamplePerItem, NumClass)
    :param anchor: torch.tensor(B, H)
    :param labels: torch.tensor(B*NumSamplePerItem)
    :param domains: torch.tensor(B*NumSamplePerItem)
    :param CLS_loss_fn: loss function for classification loss
    :return: loss
    """
    classification_loss = CLS_loss_fn(logits, labels.view(-1))

    query_feat = features[:, 0, :]
    anchor = query_feat if anchor == None else anchor
    neg_feat = features[:, 1:, :]

    contrastive_loss = CL_loss_fn(
        query=query_feat,
        positive_key=anchor,
        negative_keys=neg_feat
    )

    features_in_batch = features.view(-1,features.shape[-1])
    domain_loss = DALoss(features_in_batch,domains,t=t)
    return classification_loss +  domain_loss

    # return alpha * classification_loss + beta * contrastive_loss + (1 - alpha - beta) * domain_loss
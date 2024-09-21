import json
import os
import pandas as pd
from ..utils import *
import random
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import math


def DiffLabel_collate_fn(batch):
    return {
            'x0_input_ids':torch.stack([item['x0_input_ids'] for item in batch]),
            'attention_mask':torch.stack([item['attention_mask'] for item in batch]),
            'target_mask':torch.stack([item['target_mask'] for item in batch]),
            'mask_step_id':torch.stack([item['mask_step_id'] for item in batch]),
        }


class DifLabelDataset(data.Dataset):
    # This dataset embed the label to the input sequence

    @staticmethod
    def getTargetMask(att_score, max_step,lambda_ = 0.5):

        """
        Noise Scheduling
        # Markov Procedure
        """


        target_mask = torch.zeros_like(att_score)  # B, L
        target_masks = []
        for t in range(1,max_step+1):
            S_t = lambda_ * math.sin(t*math.pi / max_step)
            alpha_bar_matrix = 1 - (t / max_step) - S_t * att_score
            transition_prob = 1 - alpha_bar_matrix # transition probability distribution
            transition_prob = torch.where(att_score>0,
                                          transition_prob,
                                          torch.zeros_like(att_score))
            target_mask = torch.where(torch.rand_like(transition_prob) < transition_prob,
                                      torch.ones_like(target_mask),target_mask)

            target_masks.append(copy.deepcopy(target_mask))

        mask = torch.stack(target_masks) # NumStep, B, L; from all zeros to all ones (p.s. the first mask may not be all zeros)
        return mask


    @staticmethod
    def getTargetMask_ForAug(att_score, max_step,lambda_ = 0.5):

        """
        Noise Scheduling
        # Markov Procedure
        """


        target_mask = torch.zeros_like(att_score)  # B, L
        target_masks = []
        for _ in range(1,max_step+1):
            t = max_step // 2
            S_t = lambda_ * math.sin(t*math.pi / max_step)
            alpha_bar_matrix = 1 - (t / max_step) - S_t * att_score
            transition_prob = 1 - alpha_bar_matrix # transition probability distribution
            transition_prob = torch.where(att_score>0,
                                          transition_prob,
                                          torch.zeros_like(att_score))
            target_mask = torch.where(torch.rand_like(transition_prob) < transition_prob,
                                      torch.ones_like(target_mask),target_mask)

            target_masks.append(copy.deepcopy(target_mask))

        mask = torch.stack(target_masks) # NumStep, B, L; from all zeros to all ones (p.s. the first mask may not be all zeros)
        return mask

    @staticmethod
    def getTargetMask_WithRange(att_score, bag_size, step_group_num, max_step=32,lambda_ = 0.5):

        """
        Noise Scheduling
        # Markov Procedure
        """
        assert step_group_num <= (max_step // bag_size), f"step_group_num:{step_group_num}, max_step:{max_step}, bag_size:{bag_size}"
        target_mask = torch.zeros_like(att_score)  # B, L
        target_masks = []
        for t in range(1,max_step+1):
            S_t = lambda_ * math.sin(t*math.pi / max_step)
            alpha_bar_matrix = 1 - (t / max_step) - S_t * att_score
            transition_prob = 1 - alpha_bar_matrix # transition probability distribution
            transition_prob = torch.where(att_score>0,
                                          transition_prob,
                                          torch.zeros_like(att_score))
            target_mask = torch.where(torch.rand_like(transition_prob) < transition_prob,
                                      torch.ones_like(target_mask),target_mask)

            target_masks.append(copy.deepcopy(target_mask))

        mask = torch.stack(target_masks) # NumStep, B, L; from all zeros to all ones (p.s. the first mask may not be all zeros)

        # 将生成的target_masks按bagsize划分为小组
        num_groups = max_step // bag_size + (1 if max_step % bag_size != 0 else 0)
        grouped_masks = [mask[i * bag_size:min((i + 1) * bag_size, max_step)] for i in range(num_groups)]

        # 返回指定step_group_num的小组
        return grouped_masks[step_group_num]

    def __init__(self,data,AttScore, PLM_Name = 'bert-base-chinese', max_step = 32, max_length = 128):
        """
        x0_input_ids, attention_mask, target_mask =\
         dataItem['x0_input_ids'],dataItem['attention_mask'],dataItem['target_mask']
        """
        self.max_step = max_step
        self.max_length = max_length
        [data[index].update({'attention_score':AttScore[index]}) for index in range(len(data))] # label_id, contents, label, id, attention_score

        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)

        self.Label2id = getLabel2IDFromList([item['label'] for item in data])
        self.id2LabelToken = {id_:f"<LABEL-{label}>" for label,id_ in self.Label2id.items()}
        self.LabelTokens = list(self.id2LabelToken.values())
        self.tokenizer.add_tokens(self.LabelTokens)


        length = self.max_length + 3
        tokenized_items = self.tokenizer([[f"<LABEL-{item['label']}>",item['content']] for item in data],
                                         return_tensors='pt',padding='max_length',truncation=True,max_length=length)

        self.mask_step_id = torch.arange(self.max_step).repeat_interleave(tokenized_items['input_ids'].shape[0])

        self.input_ids = tokenized_items['input_ids'].repeat(self.max_step,1,1).view(-1,length) # NumStep, B, L-> NumStep * B, L+3
        self.attention_mask = tokenized_items['attention_mask'].repeat(self.max_step,1,1).view(-1,length) # NumStep, B, L-> NumStep * B, L+3
        self.target_masks = self.getTargetMask(torch.stack([item['attention_score'] for item in data]), self.max_step).view(-1,self.max_length) # NumStep, B, L -> NumStep * B, L
        labelMask = torch.zeros((self.target_masks.shape[0],3)).to(self.target_masks.device)
        self.target_masks = torch.cat([labelMask,self.target_masks],-1)


    def __getitem__(self, index):
        assert self.input_ids[index].shape == self.attention_mask[index].shape and \
               self.attention_mask[index].shape == self.target_masks[index].shape, f"{self.input_ids[index].shape}\t{self.attention_mask[index].shape}\t{self.target_masks[index].shape}"

        return {
            'x0_input_ids':self.input_ids[index],
            'attention_mask':self.attention_mask[index],
            'target_mask':self.target_masks[index],
            'mask_step_id':self.mask_step_id[index]
        }

    def __len__(self):
        return len(self.target_masks)


    def getDataLoader(self,batch_size=8,shuffle=True):
        return data.DataLoader(dataset=self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=False,
                                 collate_fn=DiffLabel_collate_fn)











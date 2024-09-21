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


def Diff_collate_fn(batch):
    return {
            'x0_input_ids':torch.stack([item['x0_input_ids'] for item in batch]),
            'attention_mask':torch.stack([item['attention_mask'] for item in batch]),
            'target_mask':torch.stack([item['target_mask'] for item in batch])
        }


class DifDataset(data.Dataset):

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

    # @staticmethod
    # def getInputsAndTargetsAndMasks(source_input_ids,target_mask, mask_id):
    #     """
    #     :param source_input_ids: torch.Tensor(B, L)
    #     :param target_mask: torch.Tensor(NumStep, B, L) # from all zeros to all ones
    #     :return:
    #     """
    #     #torch.Tensor(NumStep, B, L)
    #     source_input_ids = torch.stack([source_input_ids for _ in range(len(target_mask))])
    #     corrupted_mtx = torch.ones_like(source_input_ids) * mask_id
    #
    #     # from full text to all-masked text
    #     input_ids = torch.where(target_mask.bool(),
    #                              corrupted_mtx,
    #                              source_input_ids)
    #
    #     stepWise_input_ids = input_ids
    #     stepWise_target_ids = input_ids
    #
    #     return {
    #         'stepWise_input_ids':stepWise_input_ids,#torch.Tensor(NumStep, B, L)
    #         'stepWise_target_ids':stepWise_target_ids, #torch.Tensor(NumStep, B, L)
    #     }


    # def itemProcessor(self,item):
    #     tokenized_items = self.tokenizer(item['content'],return_tensors='pt',padding='max_length',truncation=True,max_length=self.max_length)
    #     item.update(
    #         {'input_ids': tokenized_items['input_ids'],
    #          'attention_mask':tokenized_items['attention_mask']}
    #     )
    #
    #     target_masks = self.getTargetMask(item['attention_score'], self.max_step)
    #
    #     item.update(
    #         self.getInputsAndTargets(item['input_ids'],target_masks,self.tokenizer.mask_token_id)
    #     )
    #
    #     item.update({
    #             'stepWise_attention_mask': item['attention_mask'].unsqueeze(0).repeat(self.max_step,1,1)
    #         })
    #     return item



    def __init__(self,data,AttScore, PLM_Name = 'bert-base-chinese', max_step = 32, max_length = 128):
        """
        x0_input_ids, attention_mask, target_mask =\
         dataItem['x0_input_ids'],dataItem['attention_mask'],dataItem['target_mask']
        """
        self.max_step = max_step
        self.max_length = max_length
        [data[index].update({'attention_score':AttScore[index]}) for index in range(len(data))] # label_id, contents, label, id, attention_score
        self.tokenizer = AutoTokenizer.from_pretrained(PLM_Name)
        tokenized_items = self.tokenizer([item['content'] for item in data],return_tensors='pt',padding='max_length',truncation=True,max_length=self.max_length)

        self.input_ids = tokenized_items['input_ids'].repeat(self.max_step,1,1).view(-1,self.max_length) # NumStep, B, L-> NumStep * B, L
        self.attention_mask = tokenized_items['attention_mask'].repeat(self.max_step,1,1).view(-1,self.max_length) # NumStep, B, L-> NumStep * B, L
        self.target_masks = self.getTargetMask(torch.stack([item['attention_score'] for item in data]), self.max_step).view(-1,self.max_length) # NumStep, B, L -> NumStep * B, L


        # stepWiseItem = self.getInputsAndTargetsAndMasks(tokenized_items['input_ids'], target_masks, self.tokenizer.mask_token_id)# NumStep, B, L
        #
        #
        # self.stepWise_input_ids = stepWiseItem['stepWise_input_ids'].view(-1,self.max_length) # NumStep*B, L
        # self.stepWise_target_ids = stepWiseItem['stepWise_target_ids'].view(-1,self.max_length)# NumStep*B, L

        # stepWiseAttentionMask_ = torch.stack([tokenized_items['attention_mask'] for _ in range(self.max_step)])
        # self.stepWiseAttentionMask = stepWiseAttentionMask_.view(-1,self.max_length) # NumStep*B, L



    def __getitem__(self, index):

        return {
            'x0_input_ids':self.input_ids[index],
            'attention_mask':self.attention_mask[index],
            'target_mask':self.target_masks[index]
        }

    def __len__(self):
        return len(self.target_masks)


    def getDataLoader(self,batch_size=8,shuffle=True):
        return data.DataLoader(dataset=self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=False,
                                 collate_fn=Diff_collate_fn)











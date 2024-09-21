import json
import os
import pandas as pd
from ..utils import *
import random
import copy
from tqdm import tqdm

def DA_collate_fn(batch):

    contents = [item['content'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    cluster_ids = torch.tensor([item['cluster_id'] for item in batch])
    cluster_centers = torch.cat([item['cluster_center'].unsqueeze(0) for item in batch],0)

    return {
        'contents':contents,
        'labels':labels,
        'cluster_ids':cluster_ids,
        'cluster_centers':cluster_centers
    }


class DADataset(data.Dataset):

    def __init__(self, clusteredData, clusterCenters):
        # self.clusteredData = clusteredData  ## class -> cluster -> content_list
        # self.clusterCenters = clusterCenters  ## class -> cluster -> tensor
        self.data = []

        item = {}
        print(f"Initializing Data Augmentation Datasets..")
        for label in clusteredData.keys():
            for cluster in clusterCenters.keys():
                for content in clusteredData[label][cluster]:
                    item['content'] = content
                    item['label'] = label
                    item['cluster_center'] = clusterCenters[label][cluster]
                    item['cluster_id'] = cluster
                    self.data.append(item)
                    item = {}




    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)


    def getDataLoader(self,batch_size=8,shuffle=True,num_workers=1):
        return data.DataLoader(dataset=self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=False,
                                 num_workers=num_workers,
                                 collate_fn=DA_collate_fn)



class DA_PT_Dataset(data.Dataset):

    def __init__(self, clusteredData, clusterCenters, tokenizer):
        # self.clusteredData = clusteredData  ## class -> cluster -> content_list
        # self.clusterCenters = clusterCenters  ## class -> cluster -> tensor
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.max_length=128

        print(f"Initializing Data Augmentation Datasets..")
        for label in clusteredData.keys():
            for cluster in clusterCenters.keys():
                for content in clusteredData[label][cluster]:
                    input_text = content
                    output_text = random.choice(clusteredData[label][cluster])
                    input_dicts = self.tokenizer(input_text, padding='max_length', truncation=True,max_length=self.max_length)

                    input_ids = input_dicts.input_ids
                    attention_mask = input_dicts.attention_mask
                    labels = self.tokenizer(output_text, truncation=True,max_length=self.max_length)['input_ids']

                    while len(labels) < self.max_length:
                        labels = labels + [-100]

                    self.input_ids.append(input_ids)
                    self.attn_masks.append(attention_mask)
                    self.labels.append(labels)


    def __getitem__(self, index):
        return {
            'input_ids':(self.input_ids[index]),
            'attention_mask':(self.attn_masks[index]),
            'labels':(self.labels)
        }

    def __len__(self):
        return len(self.input_ids)












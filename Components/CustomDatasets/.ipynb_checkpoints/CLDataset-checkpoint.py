import json
import os
from collections import Counter

import pandas as pd
from ..utils import *
import random
import copy
from tqdm import tqdm

class CLDataset(data.Dataset):

    def __init__(self, data_path,num_pos = 5,num_neg = 6,is_train=True):
        raw_data = json.load(open(data_path))
        self.raw_data = raw_data
        for idx in range(len(raw_data)):
            self.raw_data[idx]['label'] = str(self.raw_data[idx]['label'])

        self.is_train = is_train
        self.Label2id = getLabel2IDFromList([item['label'] for item in raw_data])

        if 'domain' in raw_data[0].keys():
            self.Domain2id = getLabel2IDFromList([item['domain'] for item in raw_data])
            for item in raw_data:
                item['domain_id'] = self.Domain2id[item['domain']]
            print(self.Domain2id)

        if is_train:
            self.dataOnLabel = {}

            for label in self.Label2id.keys():
                itemsOnLabel = []
                for item in raw_data:
                    if item['label'] == label:
                        item['label_id'] = self.Label2id[label]
                        itemsOnLabel.append(item)
                self.dataOnLabel[label] = itemsOnLabel

            self.data = []
            for item in tqdm(raw_data):
                new_item = copy.deepcopy(item)

                neg_datas = []
                pos_datas = []
                for label in self.dataOnLabel.keys():
                    if label != item['label']:
                        neg_datas += self.dataOnLabel[label]
                    else:
                        pos_datas += self.dataOnLabel[label]

                neg_samples = random.choices(neg_datas,k=num_neg)
                pos_samples = random.choices(pos_datas,k=num_pos)

#                 for label in self.dataOnLabel.keys():
#                     if label != item['label']:
#                         neg_datas += [random.choice(self.dataOnLabel[label])]
#                     else:
#                         pos_datas += self.dataOnLabel[label]

#                 neg_samples = neg_datas
#                 pos_samples = random.choices(pos_datas,k=num_pos)

                random.shuffle(neg_samples)
                random.shuffle(pos_samples)

                new_item['neg'] = neg_samples
                new_item['pos'] = pos_samples
                new_item['num_pos'] = num_pos
                new_item['num_neg'] = num_neg


                self.data.append(new_item)


        else:
            self.data = []
            for item in tqdm(copy.deepcopy(raw_data)):
                item['label_id'] = self.Label2id[item['label']]
                self.data.append(item)

    def __getitem__(self, index):

        return self.data[index]

    def get_rawData(self):
        return self.data

    def getLabel2Id(self):
        return self.Label2id

    def getBalanceInfo(self):
        labels = [item['label'] for item in self.raw_data]
        distribution = Counter(labels)
        max_num = max(distribution.values())

        balance_info = {}
        for label in distribution.keys():
            balance_info[label] = max_num - distribution[label]

        return balance_info

    def __len__(self):
        return len(self.data)




def get_datasets(data_dir):
    train_path, test_path = None, None

    for file in os.listdir(data_dir):
        if '.json' in file:
            if 'train' in file:
                train_path = file

            if 'test' in file:
                test_path = file
    datasets = {
        'train': CLDataset(f"{data_dir}/{train_path}",is_train=True),
        'test': CLDataset(f"{data_dir}/{test_path}",is_train=False)
    }
    return datasets


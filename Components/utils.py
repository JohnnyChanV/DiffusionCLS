"""
This utils can be used for further exp.
"""
import torch
from torch.utils import data
from transformers import AutoTokenizer
from torch.nn import functional as F



def sim(a,b):
    a = F.normalize(a, dim=-1) # kh
    b = F.normalize(b, dim=-1) # qh
    if len(a.shape)==2:
        s = torch.einsum('kh,qh->kq', a, b)  # kq
    else:
        s = torch.einsum('bkh,bqh->bkq', a, b)  # kq
    return s


def getLabel2IDFromList(labels):
    label_set = list(set(labels))
    label_set.sort()
    res = {}
    for index,label in enumerate(label_set):
        res.update({
            label:index
        })
    return res


def getDataLoader(dataset,collate_fn,batch_size=8,shuffle=False,num_workers=1):
    return data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)


class BatchProcessor:
    def __init__(self, tokenizer_name,max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def get_collate_fn(self,is_train):
        if is_train:
            return self.train_collate_fn
        else:
            return self.eval_collate_fn

    def train_collate_fn(self,batch):
        B = len(batch)
        instances = []
        for item in batch:
            query = item['content']
            query_label_id = item['label_id']
            query_label = item['label']


            neg_content = [n['content'] for n in item['neg']]
            neg_label_id = [n['label_id'] for n in item['neg']]
            neg_label = [n['label'] for n in item['neg']]

            # pos_content = [n['content'] for n in item['pos']]
            # pos_label_id = [n['label_id'] for n in item['pos']]
            # pos_label = [n['label'] for n in item['pos']]
            #
            # # instance = {
            # #     'pn_content':[query] + pos_content + neg_content,
            # #     'pn_label':[query_label] + pos_label + neg_label,
            # #     'pn_label_id':[query_label_id] + pos_label_id + neg_label_id
            # # }


            instance = {
                'pn_content': [query] + neg_content,
                'pn_label': [query_label] + neg_label,
                'pn_label_id': [query_label_id] + neg_label_id,
            }


            if 'domain' in item.keys():
                query_domain_id = item['domain_id']
                query_domain = item['domain']
                neg_domain_id = [n['domain_id'] for n in item['neg']]
                neg_domain = [n['domain'] for n in item['neg']]

                instance.update(
                    {'domain': [query_domain] + neg_domain,
                    'domain_id': [query_domain_id] + neg_domain_id})

            instances.append(instance)
              


        inputs = []
        for b in instances:
            inputs += b['pn_content']
        newBatch = self.tokenizer(inputs,return_tensors='pt',padding='max_length' , truncation=True,max_length=self.max_length)
        newBatch.update(
            {
                # 'num_pos':batch[0]['num_pos'],
                # 'num_neg': batch[0]['num_neg'],
                'B': B,
                'pn_label': [item['pn_label'] for item in instances],
                'pn_label_id': torch.tensor([item['pn_label_id'] for item in instances])
            }
        )

        if 'domain' in instances[0].keys():
            newBatch.update(
                {
                    'domain':[item['domain'] for item in instances],
                    'domain_id':torch.tensor([item['domain_id'] for item in instances])
                }
            )

        return newBatch

    def eval_collate_fn(self,batch):
        ids = torch.tensor([item['id'] for item in batch])
        contents = [item['content'] for item in batch]
        labels = [item['label'] for item in batch]
        label_ids = torch.tensor([item['label_id'] for item in batch])

        tokenized_items = self.tokenizer(contents,return_tensors='pt',padding='max_length' , truncation=True,max_length=self.max_length)
        tokenized_items.update({
            'contents':contents,
            'pn_label_id':label_ids,
            'labels':labels,
            'ids':ids
        })
        return tokenized_items


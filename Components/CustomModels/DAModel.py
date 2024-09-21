import torch
from info_nce import InfoNCE
from torch import nn
from transformers import *
from rouge_chinese import Rouge

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,AutoPeftModelForCausalLM
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
from .CLModel import *
from ..CustomDatasets.DADataset import *
from ..CustomDatasets.CLDataset import *


class CLSRewardModel(nn.Module):
    def __init__(self, PLM, tokenizer_name):
        print(f"\tInitializing Reward Model....")
        super().__init__()
        self.PLM = PLM.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # self.reward_weights = nn.Parameter(torch.ones())


    def rouge_score(self,refs,gens):
        rouge = Rouge()
        refs = [" ".join(self.tokenizer.tokenize(ref)) for ref in refs]
        gens = [" ".join(self.tokenizer.tokenize(gen)) for gen in gens]
        try:
            scores = [rouge.get_scores(refs[index],gens[index])[0]['rouge-2']['f'] for index in range(len(refs))]
        except:
            scores = [0]
        return sum(scores)


    def forward(self, references, anchorFeats, generation, seqProb):
        """
        :param refFeats: list(string) -> Torch.tensor(batch,H)
        :param anchorFeats: torch.tensor(batch,H)
        :param generation: list(string) -> Torch.tensor(batch,H)
        :param seqProb: Torch.tensor(batch)
        :return:
        """
        if len(references)==0 or len(generation)==0:
            return 0
        print(references[0])
        print(generation[0])
        assert  len(references) == len(generation) == seqProb.shape[0], \
            f"refNum:{len(references)}, GenNum:{len(generation)}, seqProbShape:{seqProb.shape}"

        genTokenized_And_refTokenized = self.tokenizer(generation+references,
                                                       return_tensors='pt',
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=128).to(self.PLM.device)  # Torch.tensor(batch,L)
        genFeats_And_refFeats = self.PLM(genTokenized_And_refTokenized['input_ids'],genTokenized_And_refTokenized['attention_mask']).pooler_output  # Torch.tensor(batch*2,H)
        genFeats_And_refFeats = genFeats_And_refFeats.view(2,len(references),-1)
        genFeats = genFeats_And_refFeats[0]
        refFeats = genFeats_And_refFeats[1]

        ref_gen_sim = sim(refFeats, genFeats)
        ref_gen_sim = (ref_gen_sim * torch.eye(ref_gen_sim.shape[0]).to(self.PLM.device)).sum(1) #(b)

        anchor_gen_sim = sim(anchorFeats, genFeats) # (b,b)
        anchor_gen_sim = (anchor_gen_sim * torch.eye(anchor_gen_sim.shape[0]).to(self.PLM.device)).sum(1) #(b)
        # print(ref_gen_sim,anchor_gen_sim,seqProb)
        rouge_score = torch.tensor(self.rouge_score(references,generation)).to(self.PLM.device)
        reward = (ref_gen_sim + anchor_gen_sim + rouge_score) * torch.exp(seqProb)
        reward = reward.sum()


        return reward


class DGModel(nn.Module):  # Model for data generation
    def __init__(self, geneativeModelName,device):
        self.device = device
        print(f"\tInitializing DGModel....")
        super().__init__()
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        # )
        self.generativeModel = AutoModelForCausalLM.from_pretrained(geneativeModelName)
        self.generativeModelTokenizer = AutoTokenizer.from_pretrained(geneativeModelName)
        self.generationConfig = GenerationConfig(temperature=0.7,
                                                 do_sample=True,
                                                 max_length=256,
                                                 output_scores=True,
                                                 return_dict_in_generate=True)
        # self.LoRaModel = get_peft_model(self.generativeModel, peft_config)
        # print(f"LoRA Model Initialized, with ")
        # self.LoRaModel.print_trainable_parameters()

    def forward(self,sources):
        # prompt = "根据原文仿写。\n\n原文: {source}\n\n仿写:"
        prompt = "{source}"
        prompts = [prompt.format(source=source) for source in sources]
        tokenized_prompts = self.generativeModelTokenizer(prompts,return_tensors='pt',padding=True , truncation=True, max_length=128).to(self.device)

        # outputs = self.LoRaModel.generate(tokenized_prompts,self.generationConfig)
        outputs = self.generativeModel.generate(tokenized_prompts['input_ids'],self.generationConfig)
        sentences = self.generativeModelTokenizer.batch_decode(outputs.sequences,skip_special_tokens=True)
        # sentences = [s.split("相似的句子:")[-1] for s in sentences]

        return {
            'outputs':outputs,
            'sentences':sentences,
        }

    def getLoRaParams(self):
        return self.generativeModel.parameters()
        # return self.LoRaModel.parameters()

    def setLoRaTrain(self):
        self.generativeModel.train()
        # self.LoRaModel.train()

    def setLoRaEval(self):
        self.generativeModel.train()
        # self.LoRaModel.eval()


class DAModel:
    def __init__(self, PLM, PLM_tokenizer_name, geneativeModelName, clusteredData, clusterCenters, device='cuda'):
        print(f"Initializing DAModel....")
        self.PRNG = random.getrandbits(20)
        self.PLM_Name = PLM_tokenizer_name
        self.clusteredData = clusteredData ## class -> cluster -> content_list
        self.clusterCenters = clusterCenters ## class -> cluster -> tensor
        self.device = device

        self.R_Model = CLSRewardModel(PLM, PLM_tokenizer_name).to(device)
        self.DGModel = DGModel(geneativeModelName,device).to(device)

        self.dataset = DADataset(clusteredData,clusterCenters)
        self.PT_dataset = DA_PT_Dataset(clusteredData,clusterCenters,self.DGModel.generativeModelTokenizer)
        self.optimizer = torch.optim.AdamW(self.DGModel.getLoRaParams(), lr=5e-5, weight_decay=0.03)

    def getDataLoader(self,batch_size):
        return tqdm(self.dataset.getDataLoader(batch_size=batch_size))

    def RLtrain(self,epochs=1):
        self.DGModel.setLoRaTrain()
        dataloader = self.getDataLoader(16)
        for epoch in range(epochs):
            R_sum = 0
            for index,batch in enumerate(dataloader):
                """
                batch {
                    'contents':contents,
                    'labels':labels,
                    'cluster_ids':cluster_ids,
                    'cluster_centers':cluster_centers
                }
                """
                # move data to device
                for k, v in batch.items():
                    try:
                        batch.update({k: v.to(self.device)})
                    except:
                        pass

                generativeOutputs = self.DGModel(batch['contents'])
                genSentences = generativeOutputs['sentences']
                genOutputs = generativeOutputs['outputs']
                genSeqLogProb = self.DGModel.generativeModel.compute_transition_scores(
                    genOutputs.sequences, genOutputs.scores, normalize_logits=True) #(batch,seqLen)
                genSeqLogProb = torch.where(torch.isinf(genSeqLogProb), torch.tensor(0.0), genSeqLogProb)
                genSeqLogProb = genSeqLogProb.sum(-1)

                R_Value = self.R_Model(batch['contents'],batch['cluster_centers'],genSentences,genSeqLogProb)
                R_sum += R_Value

                loss = -R_Value
                loss.backward()
                self.optimizer.step()
                dataloader.set_description(f"R_Mean:{R_sum/(index+1)}, R_Value:{R_Value}, Epoch:{epoch}")

    def doAugmentation(self):
        self.DGModel.setLoRaEval()
        results = {
            'content':[],
            'label':[],
            'genSent':[]
        }
        for index, batch in enumerate(self.getDataLoader(16)):
            """
            batch {
                'contents':contents, (list)
                'labels':labels, (tensor)
                'cluster_ids':cluster_ids, (tensor)
                'cluster_centers':cluster_centers, (tensor)
            }
            """
            # move data to device
            for k, v in batch.items():
                try:
                    batch.update({k: v.to(self.device)})
                except:
                    pass

            generativeOutputs = self.DGModel(batch['contents'])
            genSentences = generativeOutputs['sentences']

            results['content'] += batch['contents']
            results['label'] += batch['labels'].cpu().numpy().tolist()
            results['genSent'] += genSentences

        pd.DataFrame.from_dict(results).to_csv(
            f"results/{self.PLM_Name}-{self.PRNG}-{time.localtime().tm_hour}-{time.localtime().tm_min}-DataAugmentation.csv",
            encoding='utf-8-sig')



























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

class KMeans_Clustering():
    def __init__(self, features, desc="Clustering"):
        """
        :param features: torch.tensor(NumOfSample,H)
        """
        features = features.numpy()
        print(f"Performing KMeans Clustering... Features Shape:{features.shape}, Desc:{desc}")
        self.optimal_clusters = self._find_optimal_clusters(features)
        self.KMeansModel = KMeans(n_clusters=self.optimal_clusters, n_init='auto').fit(features)
        self.clusterCenters = self.__getClusterCenters(features)

    def _find_optimal_clusters(self, features):
        inertia = []
        max_clusters = min(50, len(features))
        print(f"Finding best cluster number...")
        for n_clusters in tqdm(range(1, max_clusters + 1)):
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
            kmeans.fit(features)
            inertia.append(kmeans.inertia_)

        # 找到肘部
        deltas = [inertia[i] - inertia[i - 1] for i in range(1, len(inertia))]
        max_delta_idx = np.argmax(deltas)
        self.optimal_clusters = max_delta_idx + 1
        print(f"Finding best cluster number is {self.optimal_clusters}")
        return self.optimal_clusters

    def forward(self, features):
        """
        进行K均值聚类
        """
        labels = self.KMeansModel.predict(features.numpy())
        return labels

    def __getClusterCenters(self, features):
        sub_labels = self.forward(torch.tensor(features))
        clusterCenters = dict(enumerate([[] for _ in range((self.optimal_clusters))]))

        for index, cluster in enumerate(sub_labels):
            clusterCenters[cluster].append(torch.tensor(features[index]).unsqueeze(0))

        for cluster in clusterCenters.keys():
            clusterCenters[cluster] = torch.cat(clusterCenters[cluster], 0).mean(0)
        return clusterCenters  ## size: (768)

    def getClusterCenters(self):
        return self.clusterCenters



class ClusteringModule:
    def __init__(self, featureOnClass, contentsOnClass):
        self.features = featureOnClass
        self.contents = contentsOnClass

        self.clustering_models = {}
        for label in self.features.keys():
            self.clustering_models.update({label: KMeans_Clustering(self.features[label])})

        self.clusteredData = {}  ## class -> cluster -> content_list
        for label in self.features.keys():
            self.clusteredData.update({
                label: self.getClusterData_(self.features[label], self.contents[label], self.clustering_models[label])
            })

        self.clusterCenters = {} ## class -> cluster -> tensor
        for label in self.features.keys():
            self.clusterCenters.update({
                label:self.clustering_models[label].getClusterCenters()
            })

    def getClusterData_(self, feature, contents, model):
        sub_labels = model.forward(feature)
        datas = list(zip(sub_labels, contents))
        clusteredData = dict(enumerate([[] for _ in range((model.optimal_clusters))]))
        for label, content in datas:
            clusteredData[label] += [content]
        return clusteredData

    def getClusterData(self):
        return self.clusteredData

    def getClusterCenters(self):
        return self.clusterCenters



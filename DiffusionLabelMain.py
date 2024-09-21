from Components.CustomDatasets.CLDataset import *
from Components.CustomDatasets.DADataset import *
from Components.CustomDatasets.DiffusionDataset import *

from Components.CustomModels.CLModel import *
from Components.CustomModels.DiffusionWithLabelBert import *
from Components.CustomModels.DiffusionWithLabelAugmentation import *
from Components.CustomModels.DAModel import *
from Components.CustomModels.ClusteringModel import *

from Components.utils import *
import torch
import numpy as np
import time
import warnings



def Diffusion_Augmentation_main():
    plm = "hfl/chinese-roberta-wwm-ext"
    # plm = "bert-base-chinese"
    # plm = "GKLMIP/roberta-hindi-devanagari"

    max_length = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    virusDataset = get_datasets('raw_data/virusDataset')

    TaskLabel2ID = virusDataset['test'].getLabel2Id()
    print(f"Start Time: {time.localtime().tm_hour}:{time.localtime().tm_min}")
    print(TaskLabel2ID)

    # Initialize batch processor and data loader
    batchProcessor = BatchProcessor(plm, max_length)

    virus_train_loader_TrainMode = getDataLoader(dataset=virusDataset['train'], batch_size=10,
                                                 collate_fn=batchProcessor.get_collate_fn(
                                                     virusDataset['train'].is_train), shuffle=True)
    virus_train_loader_EvalMode = getDataLoader(dataset=virusDataset['train'], batch_size=10,
                                                collate_fn=batchProcessor.get_collate_fn(is_train=False))

    virus_test_loader = getDataLoader(dataset=virusDataset['test'], batch_size=16,
                                      collate_fn=batchProcessor.get_collate_fn(virusDataset['test'].is_train))

    ct_model = CLModel(TaskLabel2ID, PLM=None, PLM_Name=plm, t=0.02, device=device)
    ct_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=1)

    raw_virus_data_for_train = virusDataset['train'].get_rawData()
    raw_virus_data_for_gen = virusDataset['train'].get_rawData()[:1000]

    virusTrainLoaderAttScore = ct_model.getAttScores(virus_train_loader_EvalMode)
    diffusion_train_dataset = DifLabelDataset(data=raw_virus_data_for_train,
                                         AttScore=virusTrainLoaderAttScore,
                                         PLM_Name=plm,
                                         max_step=16,
                                         max_length=max_length)

    diffusion_gen_dataset = DifLabelDataset(data=raw_virus_data_for_gen,
                                       AttScore=virusTrainLoaderAttScore,
                                       PLM_Name=plm,
                                       max_step=1,
                                       max_length=max_length)

    diffusion_dataloader_train = diffusion_train_dataset.getDataLoader(batch_size=32, shuffle=True)
    diffusion_dataloader_test = diffusion_gen_dataset.getDataLoader(batch_size=32, shuffle=False)

    dif_model = DiffusionModel(diffusion_train_dataset.id2LabelToken,ct_model.getPLM().cpu(), PLM_Name=plm, device=device,max_length=max_length)
    del ct_model
    dif_model.train(train_loader=diffusion_dataloader_train, eval_loader=diffusion_dataloader_test, epochs=1)

    diffusionAugmentation_model = DifA_Model(Label2Id=TaskLabel2ID,
                                             diffusionModel=dif_model.model,
                                             PLM_Name=plm,
                                             device=device,
                                             bag_size=8,
                                             t=0.02,
                                             selection_mode='selective')

    diffusionAugmentation_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=15)


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


if __name__ == "__main__":
    setup_seed(42)
    warnings.filterwarnings("ignore")
    Diffusion_Augmentation_main()

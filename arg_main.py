import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Augmentation Main Script")
    parser.add_argument("--plm", type=str, default="hfl/chinese-roberta-wwm-ext", help="Pre-trained language model name or path")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size_train", type=int, default=10, help="Batch size for training data loader")
    parser.add_argument("--batch_size_test", type=int, default=16, help="Batch size for testing data loader")
    parser.add_argument("--eval_gap", type=int, default=1)
    parser.add_argument("--epochs_ct_model", type=int, default=5, help="Number of epochs for training CLModel")
    parser.add_argument("--epochs_dif_model", type=int, default=1, help="Number of epochs for training DiffusionModel")
    parser.add_argument("--epochs_dif_augmentation_model", type=int, default=5, help="Number of epochs for training DiffusionAugmentationModel")
    parser.add_argument("--t", type=int, default=0.02, help="Number of epochs for training DiffusionAugmentationModel")
    parser.add_argument("--bag_size", type=int, default=8)
    parser.add_argument("--step_group_num", type=int, default=4)
    parser.add_argument("--max_diffusion_step", type=int, default=64)
    parser.add_argument("--dataset_name", type=str, default="/virusDataset")
    parser.add_argument("--run_type", type=str, default="train")
    parser.add_argument("--diffusion_train_mode", type=str, default="Adam")
    parser.add_argument("--balance_needed", type=bool, default=False)
    parser.add_argument("--ssl_mode", type=bool, default=False)
    return parser.parse_args()

def Diffusion_Augmentation_main(args):
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    virusDataset = get_datasets(f'raw_data/{args.dataset_name}')
    TaskLabel2ID = virusDataset['test'].getLabel2Id()
    TrainSetBalanceInfo = virusDataset['train'].getBalanceInfo()

    batchProcessor = BatchProcessor(args.plm, args.max_length)

    virus_train_loader_TrainMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                 collate_fn=batchProcessor.get_collate_fn(virusDataset['train'].is_train), shuffle=True)
    virus_train_loader_EvalMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                collate_fn=batchProcessor.get_collate_fn(is_train=False))

    virus_test_loader = getDataLoader(dataset=virusDataset['test'], batch_size=args.batch_size_test,
                                      collate_fn=batchProcessor.get_collate_fn(virusDataset['test'].is_train))


    print(TaskLabel2ID)
    print(TrainSetBalanceInfo)
    ct_model = CLModel(TaskLabel2ID, PLM=None, PLM_Name=args.plm, t=args.t, device=device)
    if not os.path.exists(dif_model_save_path):
        ct_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_ct_model,eval_gap=args.eval_gap)

    cls_report = ct_model.evaluate(virus_test_loader)
    args_dict = vars(args)


    raw_virus_data_for_train = virusDataset['train'].get_rawData()
    raw_virus_data_for_gen = virusDataset['train'].get_rawData()[:1000]

    virusTrainLoaderAttScore = ct_model.getAttScores(virus_train_loader_EvalMode)
    diffusion_train_dataset = DifLabelDataset(data=raw_virus_data_for_train,
                                         AttScore=virusTrainLoaderAttScore,
                                         PLM_Name=args.plm,
                                         max_step=args.max_diffusion_step,
                                         max_length=args.max_length)

    diffusion_gen_dataset = DifLabelDataset(data=raw_virus_data_for_gen,
                                       AttScore=virusTrainLoaderAttScore,
                                       PLM_Name=args.plm,
                                       max_step=1,
                                       max_length=args.max_length)

    diffusion_dataloader_train = diffusion_train_dataset.getDataLoader(batch_size=32, shuffle=True)
    diffusion_dataloader_test = diffusion_gen_dataset.getDataLoader(batch_size=32, shuffle=False)

    dif_model = DiffusionModel(diffusion_train_dataset.id2LabelToken,
                               ct_model.getPLM().cpu(),
                               save_path=dif_model_save_path,
                               PLM_Name=args.plm,
                               device=device,
                               max_length=args.max_length)
    del ct_model

    if os.path.exists(dif_model_save_path):
        dif_model.load(dif_model_save_path)
    else:
        dif_model.train(train_loader=diffusion_dataloader_train, eval_loader=diffusion_dataloader_test,epochs=args.epochs_dif_model,mode=args.diffusion_train_mode)

    if args.balance_needed:
        diffusionAugmentation_model = DifA_Model(Label2Id=TaskLabel2ID,
                                                 balance_info=TrainSetBalanceInfo,
                                                 step_group_num = args.step_group_num,
                                                 max_step= args.max_diffusion_step,
                                                 diffusionModel=dif_model.model,
                                                 PLM_Name=args.plm,
                                                 device=device,
                                                 bag_size=args.bag_size,
                                                 t=args.t,
                                                 selection_mode='selective',
                                                 ssl_mode=args.ssl_mode)
    else:
        diffusionAugmentation_model = DifA_Model(Label2Id=TaskLabel2ID,
                                                 balance_info=None,
                                                 step_group_num=args.step_group_num,
                                                 max_step=args.max_diffusion_step,
                                                 diffusionModel=dif_model.model,
                                                 PLM_Name=args.plm,
                                                 device=device,
                                                 bag_size=args.bag_size,
                                                 t=args.t,
                                                 selection_mode='selective',
                                                 ssl_mode=args.ssl_mode)

    diffusionAugmentation_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_dif_augmentation_model,eval_gap=args.eval_gap)
    dif_cls_report = diffusionAugmentation_model.evaluate(virus_test_loader)

    with open(f'runs/{args.dataset_name}-{random.randint(0,1000)}--{args.run_type}.txt','a',encoding='utf-8')as f:
        f.write(str(args_dict))
        f.write('\n\n')
        f.write('ct_model_eval:')
        f.write(cls_report)
        f.write('\n\n')
        f.write('augmentation_model_eval:')
        f.write(dif_cls_report)
        f.write('\n\n')



def LoopStepGroups(args):
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    virusDataset = get_datasets(f'raw_data/{args.dataset_name}')
    TaskLabel2ID = virusDataset['test'].getLabel2Id()

    batchProcessor = BatchProcessor(args.plm, args.max_length)

    virus_train_loader_TrainMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                 collate_fn=batchProcessor.get_collate_fn(virusDataset['train'].is_train), shuffle=True)
    virus_train_loader_EvalMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                collate_fn=batchProcessor.get_collate_fn(is_train=False))

    virus_test_loader = getDataLoader(dataset=virusDataset['test'], batch_size=args.batch_size_test,
                                      collate_fn=batchProcessor.get_collate_fn(virusDataset['test'].is_train))


    print(TaskLabel2ID)
    ct_model = CLModel(TaskLabel2ID, PLM=None, PLM_Name=args.plm, t=args.t, device=device)
    if not os.path.exists(dif_model_save_path):
        ct_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_ct_model)

    cls_report = ct_model.evaluate(virus_test_loader)
    args_dict = vars(args)


    raw_virus_data_for_train = virusDataset['train'].get_rawData()
    raw_virus_data_for_gen = virusDataset['train'].get_rawData()[:1000]

    virusTrainLoaderAttScore = ct_model.getAttScores(virus_train_loader_EvalMode)
    diffusion_train_dataset = DifLabelDataset(data=raw_virus_data_for_train,
                                         AttScore=virusTrainLoaderAttScore,
                                         PLM_Name=args.plm,
                                          max_step=args.max_diffusion_step,
                                          max_length=args.max_length)

    diffusion_gen_dataset = DifLabelDataset(data=raw_virus_data_for_gen,
                                       AttScore=virusTrainLoaderAttScore,
                                       PLM_Name=args.plm,
                                       max_step=1,
                                       max_length=args.max_length)

    diffusion_dataloader_train = diffusion_train_dataset.getDataLoader(batch_size=32, shuffle=True)
    diffusion_dataloader_test = diffusion_gen_dataset.getDataLoader(batch_size=32, shuffle=False)



    dif_model = DiffusionModel(diffusion_train_dataset.id2LabelToken,
                               ct_model.getPLM().cpu(),
                               save_path=dif_model_save_path,
                               PLM_Name=args.plm,
                               device=device,
                               max_length=args.max_length)
    del ct_model

    if os.path.exists(dif_model_save_path):
        dif_model.load(dif_model_save_path)
    else:
        dif_model.train(train_loader=diffusion_dataloader_train, eval_loader=diffusion_dataloader_test, epochs=args.epochs_dif_model,mode=args.diffusion_train_mode)


    for step_gp_idx in range(args.max_diffusion_step//args.bag_size):

        diffusionAugmentation_model = DifA_Model(Label2Id=TaskLabel2ID,
                                                 step_group_num = step_gp_idx,
                                                 max_step= args.max_diffusion_step,
                                                 diffusionModel=dif_model.model,
                                                 PLM_Name=args.plm,
                                                 device=device,
                                                 bag_size=args.bag_size,
                                                 t=args.t,
                                                 selection_mode='selective')

        diffusionAugmentation_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_dif_augmentation_model)
        dif_cls_report = diffusionAugmentation_model.evaluate(virus_test_loader)

        with open(f'runs/{args.dataset_name}-{random.randint(0,1000)}-{args.run_type}-{step_gp_idx}.txt','a',encoding='utf-8')as f:
            f.write(str(args_dict))
            f.write('\n\n')
            f.write('ct_model_eval:')
            f.write(cls_report)
            f.write('\n\n')
            f.write('augmentation_model_eval:')
            f.write(dif_cls_report)
            f.write('\n\n')




def ct_only_main(args):
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    virusDataset = get_datasets(f'raw_data/{args.dataset_name}')
    TaskLabel2ID = virusDataset['test'].getLabel2Id()

    batchProcessor = BatchProcessor(args.plm, args.max_length)

    virus_train_loader_TrainMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                 collate_fn=batchProcessor.get_collate_fn(virusDataset['train'].is_train), shuffle=True)
    virus_train_loader_EvalMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                collate_fn=batchProcessor.get_collate_fn(is_train=False))

    virus_test_loader = getDataLoader(dataset=virusDataset['test'], batch_size=args.batch_size_test,
                                      collate_fn=batchProcessor.get_collate_fn(virusDataset['test'].is_train))


    print(TaskLabel2ID)
    ct_model = CLModel(TaskLabel2ID, PLM=None, PLM_Name=args.plm, t=args.t, device=device)
    # if not os.path.exists(f"ckpt/DIFMODEL-{args.dataset_name}-{args.plm}-loop.ckpt"):
    ct_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_ct_model)

    cls_report = ct_model.evaluate(virus_test_loader)
    args_dict = vars(args)

    with open(f'runs/{args.dataset_name}-{random.randint(0,1000)}-{args.run_type}.txt','a',encoding='utf-8')as f:
        f.write(str(args_dict))
        f.write('\n\n')
        f.write('ct_model_eval:')
        f.write(cls_report)
        f.write('\n\n')


def gen_data_retriving(args):
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    virusDataset = get_datasets(f'raw_data/{args.dataset_name}')
    TaskLabel2ID = virusDataset['test'].getLabel2Id()
    TrainSetBalanceInfo = virusDataset['train'].getBalanceInfo()

    batchProcessor = BatchProcessor(args.plm, args.max_length)

    virus_train_loader_TrainMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                 collate_fn=batchProcessor.get_collate_fn(
                                                     virusDataset['train'].is_train), shuffle=True)
    virus_train_loader_EvalMode = getDataLoader(dataset=virusDataset['train'], batch_size=args.batch_size_train,
                                                collate_fn=batchProcessor.get_collate_fn(is_train=False))

    virus_test_loader = getDataLoader(dataset=virusDataset['test'], batch_size=args.batch_size_test,
                                      collate_fn=batchProcessor.get_collate_fn(virusDataset['test'].is_train))

    print(TaskLabel2ID)
    print(TrainSetBalanceInfo)
    ct_model = CLModel(TaskLabel2ID, PLM=None, PLM_Name=args.plm, t=args.t, device=device)
    if not os.path.exists(dif_model_save_path):
        ct_model.train(virus_train_loader_TrainMode, virus_test_loader, epochs=args.epochs_ct_model)

    cls_report = ct_model.evaluate(virus_test_loader)
    args_dict = vars(args)

    raw_virus_data_for_train = virusDataset['train'].get_rawData()
    raw_virus_data_for_gen = virusDataset['train'].get_rawData()[:1000]

    virusTrainLoaderAttScore = ct_model.getAttScores(virus_train_loader_EvalMode)
    diffusion_train_dataset = DifLabelDataset(data=raw_virus_data_for_train,
                                              AttScore=virusTrainLoaderAttScore,
                                              PLM_Name=args.plm,
                                              max_step=args.bag_size,
                                              max_length=args.max_length)

    diffusion_gen_dataset = DifLabelDataset(data=raw_virus_data_for_gen,
                                            AttScore=virusTrainLoaderAttScore,
                                            PLM_Name=args.plm,
                                            max_step=1,
                                            max_length=args.max_length)

    diffusion_dataloader_train = diffusion_train_dataset.getDataLoader(batch_size=32, shuffle=True)
    diffusion_dataloader_test = diffusion_gen_dataset.getDataLoader(batch_size=32, shuffle=False)

    dif_model = DiffusionModel(diffusion_train_dataset.id2LabelToken,
                               ct_model.getPLM().cpu(),
                               save_path=dif_model_save_path,
                               PLM_Name=args.plm,
                               device=device,
                               max_length=args.max_length)
    del ct_model

    if os.path.exists(dif_model_save_path):
        dif_model.load(dif_model_save_path)
    else:
        dif_model.train(train_loader=diffusion_dataloader_train, eval_loader=diffusion_dataloader_test,
                        epochs=args.epochs_dif_model, mode=args.diffusion_train_mode)

    dif_model.evaluate(diffusion_dataloader_train,is_save=True)

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
    args = parse_args()


    dif_model_save_path = f"ckpt/DIFMODEL-{args.dataset_name}-{args.plm.replace('/','_')}-{args.diffusion_train_mode}.ckpt"

    if args.run_type =='train':
        Diffusion_Augmentation_main(args)
    elif args.run_type == 'loop':
        LoopStepGroups(args)
    elif args.run_type == 'ct':
        ct_only_main(args)
    elif args.run_type == 'visualization':
        gen_data_retriving(args)


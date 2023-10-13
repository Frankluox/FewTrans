import yaml
import os
import argparse

all_roots = {}
# all_roots["ILSVRC"] = "/dev/shm/ImageNet_train"
# all_roots["ILSVRC"] = "/home/wuhao/data/imagenet/val" #0
all_roots["ILSVRC"] = "/home/luoxu/data2/all_dataset/ImageNet_sketch" #0
all_roots["Quick Draw"] = "/home/luoxu/data/all_datasets/domainnet/quickdraw" #1
all_roots["VGG Flower"] = "../data/all_datasets/vggflowers"#2
all_roots["Aircraft"] = "/home/luoxu/data2/all_dataset/aircraft_nocrop"#3
all_roots["Textures"] = "../data/all_datasets/dtd"#4
all_roots["Fungi"] = "../data/all_datasets/fungi"#5
all_roots["CIFAR100"] = "../data/all_datasets/cifar100"#6
all_roots["euroSAT"] = "../data2/all_dataset/EuroSAT"#7
all_roots["ucf"] = "/home/luoxu/data2/all_dataset/UCF-101-midframes"#8
all_roots["plantD"] = "/home/luoxu/data2/all_dataset/plant_disease/train"#9

Data = {}
# mode = "vpt"
# mode = "visualFT"
# mode = "zeroshot"
# mode = "MaPLe"
# mode = "coOp"
# mode = "textFT"
mode = "allFT"
# mode = "CoCoOp"
# mode = "ProGrad"
# mode = "kgcoop"
Data["DATA"] = {}
Data["DATA"]["BASE2NOVEL"] = True

Data["IS_TRAIN"] = 0

names = list(all_roots.keys())
roots = list(all_roots.values())


Data["DATA"]["TEST"] = {}

Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[0]]
Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[0]]


# 5 way 1 shot example
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
Data["DATA"]["TEST"]["BATCH_SIZE"] = 2

# 5 way 5 shot example
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
# Data["DATA"]["TEST"]["BATCH_SIZE"] = 8

# varied way varied shot example
# Data["DATA"]["TEST"]["BATCH_SIZE"] = 8



Data["OUTPUT"] = "../data2/new_metadataset_result"
Data["MODEL"] = {}

Data["MODEL"]["NAME"] = "evaluation"
Data["GPU_ID"] = 7

# 1 if use sequential sampling in the oroginal biased Meta-Dataset sampling procedure, 0 unbiased.
# 1 can be used to re-implement the results in the ICML 2023 paper (except traffic signs); 0, however, is recommended for unbiased results
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 0

Data["AUG"] = {}

# ImageNet
Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
Data["AUG"]["STD"] = [0.229, 0.224, 0.225]

# miniImageNet
# Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
# Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

# ImageNet
Data["DATA"]["IMG_SIZE"] = 224

# miniImageNet
# Data["DATA"]["IMG_SIZE"] = 84

# Data["MODEL"]["BACKBONE"] = 'resnet12'
# Data["MODEL"]["BACKBONE"] = 'resnet50'
# Data["MODEL"]["BACKBONE"] = 'clip'
# Data["MODEL"]["BACKBONE"] = 'res18_url'
# Data["MODEL"]["BACKBONE"] = 'DINO_ViT'

if mode == "coOp" or mode == "CoCoOp" or mode == "ProGrad" or mode == "kgcoop":
   Data["MODEL"]["BACKBONE"] = "clip_prompt_coop"
elif mode == "MaPLe":
   Data["MODEL"]["BACKBONE"] = "clip_prompt_maple"
elif mode == "textFT" or mode == "allFT":
   Data["MODEL"]["BACKBONE"] = "clip_prompt_text_finetune"
else:
   Data["MODEL"]["BACKBONE"] = "clip_prompt"

if mode == "visualFT":
   Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0, "visualFT"]
elif mode == "zeroshot":
   Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0]
elif mode == "textFT" or mode == "allFT" or mode == "CoCoOp" or mode == "ProGrad" or mode == "kgcoop":
   Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [mode]
# Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0]


# Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = ['small', 16]

# Data["MODEL"]["PRETRAINED"] = '../data/pretrained_models/dino_deitsmall16_pretrain.pth'# for example
# Data["MODEL"]["PRETRAINED"] = '../data2/new_metadataset_result/ImageNet_PNptrainfromDINOsmall16official/task10000lr0.0001warm2000fromE-6/ckpt_epoch_5_top1.pth'# for example

Data["DATA"]["NUM_WORKERS"] = 8


# True for re-implementing the results in the ICML 2023 paper.
# Data["AUG"]["TEST_CROP"] = True

Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
Data["PRINT_FREQ"] = 20
# some examples of gradient-based methods. Hyperparameters need to be tuned by using search_hyperparameter.py
# Data["MODEL"]["TYPE"] = "fewshot_finetune"



Data["MODEL"]["TYPE"] = "fewshot_finetune_ensemble"
# if mode == "visualFT":
# Data["MODEL"]["TYPE"] = "fewshot_finetune_ensemble_visualFT"
# elif mode == "vpt":
# Data["MODEL"]["TYPE"] = "fewshot_finetune_ensemble"
# elif mode == "MaPLe":
#    Data["MODEL"]["TYPE"] = "fewshot_finetune_ensemble_maple"
# elif mode == "coOp":
#    Data["MODEL"]["TYPE"] = "fewshot_finetune_ensemble_coop"
# else:
#    Data["MODEL"]["TYPE"] = "fewshot_finetune"
# Data["MODEL"]["CLASSIFIER"] = "finetune"
# Data["MODEL"]["CLASSIFIER"] = "eTT"
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune_batchsize,query_feedingbatchsize,epoch,backbone_lr,classifer_lr,use_alpha,use_beta, mode
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,True,True,"NCC"]# tsa
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,True,"NCC"]# URL
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"cc"]# CC
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,"eTT"]# eTT

if mode == "zeroshot":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None,"vpt"]
elif mode == "allFT":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,128,None, None, mode]
elif mode == "visualFT":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,256,None, None, mode]
elif mode == "CoCoOp":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [8,16,None, None, mode]
else:
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode]
   # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,128,None, None, mode]

# other adaptation classifiers
# Data["MODEL"]["TYPE"] = "Episodic_Model"
# Data["MODEL"]["CLASSIFIER"] = "LR"
# Data["MODEL"]["CLASSIFIER"] = "metaopt"
# Data["MODEL"]["CLASSIFIER"] = "proto_head"
Data["MODEL"]["CLASSIFIER"] = "prompt_tuning"
# Data["MODEL"]["TYPE"] = "promt_tune"
# Data["MODEL"]["CLASSIFIER"] = "MatchingNet"

parser = argparse.ArgumentParser('write_yaml', add_help=False)
parser.add_argument('--dataset_id', type=int, required=True)
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--cross_dataset', type=int, choices = [0,1], required=True)

parser.add_argument('--vary', type=int, choices = [0,1], required=True)
parser.add_argument('--way', type=int, required=True)
parser.add_argument('--shot', type=int, required=True)
parser.add_argument('--model_name', type=str, required=True)

args, unparsed = parser.parse_known_args()

if args.cross_dataset:
      Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[8], roots[12]]
      Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[8], names[12]]
      Data["DATA"]["CROSS_DATASET"] = True
else:
      Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[args.dataset_id]]
      Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[args.dataset_id]]

Data["GPU_ID"] = args.gpu_id
if not args.vary:
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_NOVEL"] = 15
#    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = args.way
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = args.shot
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = args.shot + Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"]
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
   with open(f'./configs/PN/PN_singledomain_test_{args.way}w_{args.shot}s_{args.dataset_id}_{args.model_name}.yaml', 'w') as f:
      yaml.dump(data=Data, stream=f)
else:

   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_WAYS"] = 2
# maximum ways to sample
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_WAYS_UPPER_BOUND"] = 15
      # maximum total number of images in the support set
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SET_SIZE"] = 150
      # maximum total number of images per class in the support set
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS"] = 10
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = 2
      # randomly decide contribution of each class to the support set
      # see Appendix of the Meta-Dataset paper for detail
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = True
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = True
   with open(f'./configs/PN/PN_singledomain_test_vary_way_vary_shot_{args.dataset_id}_{args.model_name}.yaml', 'w') as f:
      yaml.dump(data=Data, stream=f)
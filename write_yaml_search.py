# only for non-clip loss
import yaml
import os
import argparse

all_roots = {}

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
mode = "finetune"
# mode = "vpt"
# mode = "LoRA"
# mode = "linear"
# mode = "bias"
# mode = "adapter"
# mode = "adaptformer"
# mode = "TSA"
noclip = True

Data["DATA"] = {}
Data["DATA"]["BASE2NOVEL"] = False

Data["IS_TRAIN"] = 0

names = list(all_roots.keys())
roots = list(all_roots.values())


Data["DATA"]["TEST"] = {}

# Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[0]]
# Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[0]]

# Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[10]]
# Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[10]]

# Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[1]]
# Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[1]]

Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}

# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_NOVEL"] = 15
# #    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 1
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] + Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"]
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SAMPLE_ALL"] = 1

# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_WAYS"] = 2
# # maximum ways to sample
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_WAYS_UPPER_BOUND"] = 15
# # maximum total number of images in the support set
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SET_SIZE"] = 150
# # maximum total number of images per class in the support set
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS"] = 10
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = 2
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = True


Data["DATA"]["TEST"]["BATCH_SIZE"] = 1



Data["OUTPUT"] = "../data2/new_metadataset_result"

Data["MODEL"] = {}

Data["MODEL"]["NAME"] = "find_hyperparameters"

# 1 if use sequential sampling in the original false Meta-Dataset sampling
# 1 used to re-implement the results in the ICML 2023 paper; 0, however, is recommended
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
# Data["MODEL"]["BACKBONE"] = "clip_visualonly"
Data["MODEL"]["BACKBONE"] = "DINOv2"
# Data["MODEL"]["BACKBONE"] = "resnet50"
# Data["MODEL"]["BACKBONE"] = "mae_official" #32 128
# Data["MODEL"]["BACKBONE"] = "BiT" # r101: 64
# Data["MODEL"]["BACKBONE"] = "swin_transformer"
# Data["MODEL"]["BACKBONE"] = "convnext" # 32
# Data["MODEL"]["BACKBONE"] = "IBOT_ViT" # 32

# Data["MODEL"]["PRETRAINED"] = '../pretrained_models/ce_miniImageNet_res12.ckpt'# for example
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data2/pretrained_models/dinov2/dinov2_vitl14_pretrain.pth"
Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/dinov2_vits14_pretrain.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/resnet50_official_nofc.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/mae_finetuned_vit_base.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/BiT-M-R101x1.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/swin_base_patch4_window7_224.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/convnext_base_1k_224_ema.pth"
# Data["MODEL"]["PRETRAINED"] = "/home/luoxu/data/pretrained_models/IBOT_ViT_B_clean.pth"

if not (mode == "vpt" or noclip):
   Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0, mode]

# Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = ["large"]
# Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = ["small"]

# Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = ["101"]


Data["DATA"]["NUM_WORKERS"] = 8

Data["AUG"]["TEST_CROP"] = False

Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 50
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 1


# some examples of gradient-based methods.
Data["MODEL"]["TYPE"] = "fewshot_finetune"


if mode == "finetune":
   # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [16,64,None, None, mode]
   # mae base
   # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,128,None, None, mode]
   # vit-l
   # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [8,32,None, None, mode]
   # vit-s or resnet101
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode]
elif mode == "vpt" or mode == "bias" or mode == "SSF":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,128,None, None, mode]
elif mode == "linear":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [512,2048,None, None, mode]
elif mode == "TSA":
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode, "NCC"]
else:
   Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode]

if Data["MODEL"]["BACKBONE"] == "mae_official": 
   Data["MODEL"]["CLASSIFIER"] = "prompt_tuning_visualonly"
else:
   Data["MODEL"]["CLASSIFIER"] = "prompt_tuning_visualonly_cosine"

# Data["MODEL"]["CLASSIFIER"] = "eTT"

# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,100,0.02,0.1,False,False,"fc"]# finetune_batchsize,query_feedingbatchsize,epoch,backbone_lr,classifer_lr,use_alpha,use_beta, mode
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,False,"fc"]# finetune
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,True,True,"NCC"]# tsa
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,True,"NCC"]# URL
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,False,"cc"]# CC
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,"eTT"]# eTT


Data["PRINT_FREQ"] = 2


Data["SEARCH_HYPERPARAMETERS"] = {}
Data["SEARCH_HYPERPARAMETERS"]["BASE_ONLY"] = True
# change this

# Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [0.0000001,0.000001, 0.00001]
# Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0.02,0.2,2]
# Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [40]


# Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [1e-06,2e-06,5e-06,1e-05,2e-05,5e-05,1e-04]
Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [2e-07,1e-06,5e-06]#[1e-06,1e-05,1e-04]
# Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [1e-04,1e-03]
# Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [0]
# Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0.001,0.002,0.005,0.01,0.02,0.05,0.1]
Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0.001,0.005,0.02]#[0.001,0.01,0.1]
# Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [60,70,80]
# Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [30,40,50,60,70]
# Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [50,70,100]
Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [30]
Data["SEED"] = 0

parser = argparse.ArgumentParser('write_yaml', add_help=False)
parser.add_argument('--dataset_id', type=int, required=True)
parser.add_argument('--gpu_id', type=int, required=True)
# parser.add_argument('--cross_dataset', type=int, choices = [0,1], required=True)

parser.add_argument('--vary', type=int, choices = [0,1], required=True)
parser.add_argument('--way', type=int, required=True)
parser.add_argument('--shot', type=int, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--sample_all', type=int, choices = [0,1], required=True)

args, unparsed = parser.parse_known_args()

Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[args.dataset_id]]
Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[args.dataset_id]]
Data["GPU_ID"] = args.gpu_id

if not args.vary:
   # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_NOVEL"] = 15
   #    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = args.way
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = args.shot
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = args.shot + Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"]
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
   Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SAMPLE_ALL"] = args.sample_all
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

# Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [0.0002,0.002,0.02]
# Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0.01,0.1,1]
# Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [5,10,20]
# if not os.path.exists('./configs/search'):
#    os.makedirs('./configs/search')
# with open('./configs/search/finetune_res12_CE.yaml', 'w') as f:
#    yaml.dump(data=Data, stream=f)
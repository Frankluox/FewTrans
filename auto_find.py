import yaml
import os
import json
import sys

DATA_ROOT = "./data" 
OUTPUT_ROOT = "./results"


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() #每次写入后刷新到文件中，防止程序意外结束
    def flush(self):
        self.log.flush()
 
 
all_roots = {}
all_roots["ILSVRC"] = os.path.join(DATA_ROOT, "imagenet/val")
all_roots["Quick Draw"] = os.path.join(DATA_ROOT, "domainnet/quickdraw")
all_roots["VGG Flower"] = os.path.join(DATA_ROOT, "vggflowers")
all_roots["Aircraft"] = os.path.join(DATA_ROOT, "aircraft")
all_roots["Textures"] = os.path.join(DATA_ROOT, "dtd")
all_roots["Fungi"] = os.path.join(DATA_ROOT, "fungi")
all_roots["CIFAR100"] = os.path.join(DATA_ROOT, "cifar100")
all_roots["euroSAT"] = os.path.join(DATA_ROOT, "EuroSAT")
all_roots["ucf"] = os.path.join(DATA_ROOT, "UCF-101-midframes")
all_roots["plantD"] = os.path.join(DATA_ROOT, "plant_disease/train")
all_roots["CUB"] = os.path.join(DATA_ROOT, "CUB_200_2011")





def determine_test_config_clip(way, shot, epoch_range, lr_backbone_range, dataset_idx, model_name, dataset_name, gpuid, imgsz):

    Data = {}
    Data["GPU_ID"] = 0
    Data["DATA"] = {}

    # mode = "vpt"
    # mode = "visualFT"
    # mode = "zeroshot"
    # mode = "coOp"
    mode = "MaPLe"
    # mode = "textFT"
    # mode = "allFT"
    # mode = "CoCoOp"
    # mode = "ProGrad"
    # mode = "kgcoop"
    # Data["DATA"]["ALL_TEST"] = True
    

    Data["DATA"]["TEST"] = {}
    Data["DATA"]["BASE2NOVEL"] = False

    names = list(all_roots.keys())
    roots = list(all_roots.values())

    Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[dataset_idx]]
    Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[dataset_idx]]


    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}

    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = way
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = shot
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]

    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_WAYS"] = 2
    # maximum ways to sample
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_WAYS_UPPER_BOUND"] = 15
      # maximum total number of images in the support set
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SET_SIZE"] = 150
      # maximum total number of images per class in the support set
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS"] = 10
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = 2
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = True


    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 50
    Data["DATA"]["TEST"]["BATCH_SIZE"] = 2

    Data["OUTPUT"] = OUTPUT_ROOT
    Data["MODEL"] = {}
    # Data["MODEL"]["NAME"] = "Multi-domain_Res12"
    # Data["MODEL"]["NAME"] = "Single-domain_Res50_official"
    Data["MODEL"]["NAME"] = "find_hyperparameters"
    # Data["MODEL"]["NAME"] = f"{model_name}_alltest"
    Data["GPU_ID"] = gpuid
    # Data["SEED"] = 20
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 0
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 1
    Data["AUG"] = {}

    Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
    Data["AUG"]["STD"] = [0.229, 0.224, 0.225]
    # Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
    # Data["AUG"]["MEAN"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

    Data["DATA"]["IMG_SIZE"] = imgsz
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

    Data["DATA"]["NUM_WORKERS"] = 8
    Data["AUG"]["TEST_CROP"] = False
    # Data["DATA"]["REAL_CLASS_IDS"] = True

    # Data["MODEL"]["TYPE"] = "Episodic_Model"
    # Data["MODEL"]["TYPE"] = "promt_tune"
    Data["MODEL"]["TYPE"] = "fewshot_finetune"

    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,False,False,"fc"]# batchsize,test_batchsize,epoch,base_lr,classifer_lr,alpha,beta
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None,"visualFT"]# finetune
    
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
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,256,None, None,"visualFT"]# finetune
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,True,True,"NCC"]# tsa
    # print(epoch, lr_backbone, lr_head)
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,epoch, lr_backbone, lr_head,False,True,"NCC"]# URL

    Data["MODEL"]["CLASSIFIER"] = "prompt_tuning"

    Data["SEARCH_HYPERPARAMETERS"] = {}
    Data["SEARCH_HYPERPARAMETERS"]["BASE_ONLY"] = True
    Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = lr_backbone_range
    # Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = lr_head_range
    Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0]

    Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = epoch_range

    # with open('./configs/PN/PN_singledomain_test.yaml', 'w') as f:
    # with open(f'./configs/test_{model_name}_{dataset_name}.yaml', 'w') as f:
    with open(f'./configs/find_hyperparameter_{model_name}_{dataset_name}.yaml', 'w') as f:
        yaml.dump(data=Data, stream=f)


def determine_test_config_visual_only(way, shot, epoch_range, lr_backbone_range, lr_head_range, dataset_idx, model_name, dataset_name, gpuid, imgsz):

    Data = {}
    Data["GPU_ID"] = 0
    Data["DATA"] = {}

    # mode = "vpt"
    # mode = "finetune"
    mode = "LoRA"
    # mode = "visualFT"
    # mode = "zeroshot"
    # mode = "coOp"
    # mode = "MaPLe"
    # mode = "textFT"
    # mode = "allFT"
    # mode = "CoCoOp"
    # Data["DATA"]["ALL_TEST"] = True
    # noclip = True
    noclip = False
    

    Data["DATA"]["TEST"] = {}
    Data["DATA"]["BASE2NOVEL"] = False

    names = list(all_roots.keys())
    roots = list(all_roots.values())

    Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[dataset_idx]]
    Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[dataset_idx]]


    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}

    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 1
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"] = 15
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY_BASE"]

    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_WAYS"] = 2
    # # maximum ways to sample
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_WAYS_UPPER_BOUND"] = 15
    #   # maximum total number of images in the support set
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SET_SIZE"] = 150
    #   # maximum total number of images per class in the support set
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS"] = 10
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = 2
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = True


    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 50
    Data["DATA"]["TEST"]["BATCH_SIZE"] = 2

    Data["OUTPUT"] = OUTPUT_ROOT
    Data["MODEL"] = {}
    # Data["MODEL"]["NAME"] = "Multi-domain_Res12"
    # Data["MODEL"]["NAME"] = "Single-domain_Res50_official"
    Data["MODEL"]["NAME"] = "find_hyperparameters"
    # Data["MODEL"]["NAME"] = f"{model_name}_alltest"
    Data["GPU_ID"] = gpuid
    # Data["SEED"] = 20
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 0
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 1
    Data["AUG"] = {}

    Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
    Data["AUG"]["STD"] = [0.229, 0.224, 0.225]
    # Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
    # Data["AUG"]["MEAN"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

    Data["DATA"]["IMG_SIZE"] = imgsz

    
    Data["MODEL"]["BACKBONE"] = "clip_visualonly"
    # Data["MODEL"]["BACKBONE"] = "mae_official"
    # Data["MODEL"]["BACKBONE"] = "BiT" # r101: 64
    # Data["MODEL"]["BACKBONE"] = "swin_transformer" # 32
    # Data["MODEL"]["BACKBONE"] = "convnext" # 32
    # Data["MODEL"]["BACKBONE"] = "DINOv2"
    # Data["MODEL"]["BACKBONE"] = "IBOT_ViT" # 32
    
    if not (mode == "vpt" or noclip):
        Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0, mode]
    # if mode == "finetune":
    #     Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0, "finetune"]
    # Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = ["101"]
    Data["DATA"]["NUM_WORKERS"] = 8
    Data["AUG"]["TEST_CROP"] = False
    # Data["DATA"]["REAL_CLASS_IDS"] = True

    # Data["MODEL"]["TYPE"] = "Episodic_Model"
    # Data["MODEL"]["TYPE"] = "promt_tune"
    
    Data["MODEL"]["TYPE"] = "fewshot_finetune"

    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,False,False,"fc"]# batchsize,test_batchsize,epoch,base_lr,classifer_lr,alpha,beta
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None,"visualFT"]# finetune
    
    if mode == "finetune":
        # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,128,None, None, mode]
        Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode]
    else:
        Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None, mode]
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,256,None, None,"visualFT"]# finetune
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,True,True,"NCC"]# tsa
    # print(epoch, lr_backbone, lr_head)
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,epoch, lr_backbone, lr_head,False,True,"NCC"]# URL

    # Data["MODEL"]["CLASSIFIER"] = "prompt_tuning_visualonly"
    Data["MODEL"]["CLASSIFIER"] = "prompt_tuning_visualonly_cosine"

    Data["SEARCH_HYPERPARAMETERS"] = {}
    Data["SEARCH_HYPERPARAMETERS"]["BASE_ONLY"] = True
    Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = lr_backbone_range
    Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = lr_head_range
    # Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0]

    Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = epoch_range

    # with open('./configs/PN/PN_singledomain_test.yaml', 'w') as f:
    # with open(f'./configs/test_{model_name}_{dataset_name}.yaml', 'w') as f:
    with open(f'./configs/find_hyperparameter_{model_name}_{dataset_name}.yaml', 'w') as f:
        yaml.dump(data=Data, stream=f)


def determine_test_config_linear(way, shot, epoch_range, lr_head_range, dataset_idx, model_name, dataset_name, gpuid, imgsz):

    Data = {}
    Data["GPU_ID"] = 0
    Data["DATA"] = {}


    # Data["DATA"]["ALL_TEST"] = True
    

    Data["DATA"]["TEST"] = {}
    Data["DATA"]["BASE2NOVEL"] = False

    names = list(all_roots.keys())
    roots = list(all_roots.values())

    Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[dataset_idx]]
    Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[dataset_idx]]


    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}

    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = way
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = shot
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]

    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_WAYS"] = 2
    # maximum ways to sample
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_WAYS_UPPER_BOUND"] = 15
      # maximum total number of images in the support set
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SET_SIZE"] = 150
      # maximum total number of images per class in the support set
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS"] = 10
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = 2
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = True


    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 50
    Data["DATA"]["TEST"]["BATCH_SIZE"] = 2

    Data["OUTPUT"] = OUTPUT_ROOT
    Data["MODEL"] = {}
    # Data["MODEL"]["NAME"] = "Multi-domain_Res12"
    # Data["MODEL"]["NAME"] = "Single-domain_Res50_official"
    Data["MODEL"]["NAME"] = "find_hyperparameters"
    # Data["MODEL"]["NAME"] = f"{model_name}_alltest"
    Data["GPU_ID"] = gpuid
    # Data["SEED"] = 20
    Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 0
    # Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 1
    Data["AUG"] = {}

    Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
    Data["AUG"]["STD"] = [0.229, 0.224, 0.225]
    # Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
    # Data["AUG"]["MEAN"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.5,0.5,0.5]
    # Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

    Data["DATA"]["IMG_SIZE"] = imgsz
    Data["MODEL"]["BACKBONE"] = "clip_visualonly"
    

    Data["MODEL"]["BACKBONE_HYPERPARAMETERS"] = [0, "visualFT"]


    Data["DATA"]["NUM_WORKERS"] = 8
    Data["AUG"]["TEST_CROP"] = False
    # Data["DATA"]["REAL_CLASS_IDS"] = True

    # Data["MODEL"]["TYPE"] = "Episodic_Model"
    # Data["MODEL"]["TYPE"] = "promt_tune"
    Data["MODEL"]["TYPE"] = "fewshot_finetune"

    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,False,False,"fc"]# batchsize,test_batchsize,epoch,base_lr,classifer_lr,alpha,beta
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [64,256,None, None,"visualFT"]# finetune
    

    Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [512,2048,None, None,"linear"]

    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [32,256,None, None,"visualFT"]# finetune
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [25,100,100,0.02,0.1,True,True,"NCC"]# tsa
    # print(epoch, lr_backbone, lr_head)
    # Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,epoch, lr_backbone, lr_head,False,True,"NCC"]# URL

    Data["MODEL"]["CLASSIFIER"] = "prompt_tuning_visualonly"

    Data["SEARCH_HYPERPARAMETERS"] = {}
    Data["SEARCH_HYPERPARAMETERS"]["BASE_ONLY"] = True
    Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [0]
    # Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = lr_head_range
    Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = lr_head_range

    Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = epoch_range

    # with open('./configs/PN/PN_singledomain_test.yaml', 'w') as f:
    # with open(f'./configs/test_{model_name}_{dataset_name}.yaml', 'w') as f:
    with open(f'./configs/find_hyperparameter_{model_name}_{dataset_name}.yaml', 'w') as f:
        yaml.dump(data=Data, stream=f)


def decide_epochs(epoch, shot1 = False, in_loop = True, up = True, still = False):

    


    #in_loop: whether not the first try
    #up: in_loop时，是否需要往上找
    if still:
        return [epoch]
    list_ = [1,2,5,10,15,20,30,40,50,60,70,80,90,100]
    assert epoch in list_
    for i, element in enumerate(list_):
        if epoch == element:
            specified_index = i
            break

    if specified_index == 0:
        if in_loop:
            #不是first try时，是降下来的，所以无脑5
            return list_[:1]
        else:
            return list_[:2]
    elif specified_index == len(list_)-1:
        if shot1 and not in_loop:
            #shot 1时，可能way变大，epoch会变小
            return list_[-2:]
        else:
            #其他情况无脑100
            return list_[-1:]
    else:
        if in_loop:
            if up:
                return list_[specified_index:specified_index+2]
            else:
                return list_[specified_index-1:specified_index+1]
        else:
            return list_[specified_index-1:specified_index+2]
    

def sishewuru(a):
    if a == 0:
        return a
    times = 1
    while True:
        if a>=1:
            break
        a *= 10
        times *= 10
    a = round(a,1)
    a /= times
    return a   

def decide_lr(lr, in_loop = True, up = True, still = False):
    # lower_bound = 1e-5
    lower_bound = 1e-7
    # lr = sishewuru(lr)
    if still:
        return [lr]
    if lr == 0:
        if in_loop:
            return [0]
        else:
            return [0, lower_bound]
    if lr == lower_bound:
        if in_loop:
            if up:
                return [lower_bound, 0.000002]
            else:
                return [0, lower_bound]
        else:                  
            return [0, lower_bound, 0.000002]
    
    if "1" in str(lr) or "5" in str(lr):
        upper = lr*2
    else:
        upper = lr*2.5
    
    if "2" in str(lr) or "1" in str(lr):
        lower = lr/2
    else:
        lower = lr/2.5

    if in_loop:
        if up:
            return [lr, upper]
        else:
            return [lower, lr]
    else:
        return[lower,lr,upper]




def critical_point(is_shot,shot_list, dataset_idx_list, model_name, gpuid, imgsz=224):



    names = list(all_roots.keys())
    # dataset_idx = 0
    # model_name = "url_mdl"


    for dataset_idx in dataset_idx_list:
        


        total_dic = {}
        
        if is_shot:
            way = 5
        else:
            shot = 5 
        shot1 = True


        for id_, num in enumerate(shot_list):
            # finetune
            # previous_ft_epoch = 70
            # previous_ft_lr_1 = 0.00001
            # previous_ft_lr_2 = 0.002

            # lora
            previous_ft_epoch = 40
            previous_ft_lr_1 = 0.002
            previous_ft_lr_2 = 0.01

            previous_shot1_epoch = previous_ft_epoch
            previous_shot1_ft_lr_1 = previous_ft_lr_1
            previous_shot1_ft_lr_2 = previous_ft_lr_2
            if is_shot:
                shot = num
            else:
                way = num
        # for shot in ["vary"]:
            dic_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], "final_results.json")
            if os.path.exists(dic_path):
                with open(dic_path, "r") as file_:
                    dic = json.load(file_)
                # if len(dic)>id_:
                if f"{way}_{shot}" in dic:
                    continue

            print(f"way: {way}, shot: {shot}")

            first_loop_flag = True

            status = ["","",""]
            runs = 0
            while True:  
                runs+=1        
                parameters = []
                if first_loop_flag:
                    if shot1:
                        parameters.append(decide_epochs(previous_shot1_epoch, True, False))
                        if previous_ft_lr_1 == 0:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False, still=True))
                        else:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False))
                        if previous_ft_lr_2 == 0:
                            parameters.append(decide_lr(previous_shot1_ft_lr_2, False, still=True))
                        else:
                            parameters.append(decide_lr(previous_shot1_ft_lr_2, False))
                    else:
                        parameters.append(decide_epochs(previous_ft_epoch, False, False))
                        parameters.append(decide_lr(previous_ft_lr_1, False))
                        parameters.append(decide_lr(previous_ft_lr_2, False))
                    
                else:
                    #假设已经有了上一轮的status，以及parameters
                    assert len(status) == 3
                    ups = []
                    stills = []
                    print(status)
                    for sta in status:
                        if sta == "still":
                            ups.append(True)
                            stills.append(True)
                        elif sta == "up":
                            ups.append(True)
                            stills.append(False)
                        else:
                            ups.append(False)
                            stills.append(False)
                    for i in range(3):
                        # print(current_parameters[i])
                        # print(ups[i])
                        # print(stills[i])
                        if i == 0:
                            parameters.append(decide_epochs(current_parameters[i], False, True,ups[i], stills[i]))
                        else:
                            parameters.append(decide_lr(current_parameters[i], True,ups[i], stills[i]))
                        # print(parameters[i])
                # print(parameters)

                determine_test_config_visual_only(way, shot, parameters[0], parameters[1], parameters[2], dataset_idx, model_name, names[dataset_idx], gpuid, imgsz)
                

                
                tag = f"{model_name}/{names[dataset_idx]}/{way}w{shot}s/iteration_{runs}"
                
                try:
                    a = os.system(f"python search_hyperparameter.py --cfg configs/find_hyperparameter_{model_name}_{names[dataset_idx]}.yaml --is_train 0 --tag {tag}")
                except ValueError:
                    valueerror = True
                    print("meet value error, escape this dataset")
                    break
                else:
                    if a==256:
                        valueerror = True
                        print("meet value error, escape this dataset")
                        break
                valueerror = False

                json_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], f"{way}w{shot}s", f"iteration_{runs}", "results.json")

                with open(json_path) as _file:
                    no_U = json.load(_file)


                max_value = -1.
                position = [0,0,0]
                i = 0
                j = 0
                k = 0

                for value in no_U:
                    if value[3]>max_value:
                        max_value = value[3]
                        max_hyper = value[:3]
                        position = [i,j,k]
                    k += 1
                    if k == len(parameters[2]):
                        k=0
                        j += 1
                    if j == len(parameters[1]):
                        j=0
                        i += 1

                largest_index = position
                current_parameters = max_hyper
                

                
                if first_loop_flag:
                    for x in range(3):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            status[x] = "down"
                        elif index == len(parameters[x])-1:
                            status[x] = "up"
                        else:
                            status[x] = "still"
                else:
                    for x in range(3):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            if status[x] == "up":
                                status[x] = "still"
                            else:
                                status[x] = "down"
                        elif index == len(parameters[x])-1:
                            if status[x] == "up":
                                status[x] = "up"
                            else:
                                status[x] = "still"
                        else:
                            status[x] ="still"
                first_loop_flag = False
                # print(status)
                if "down" not in status and "up" not in status:
                    break
            
            if valueerror:
                break
            previous_ft_epoch = current_parameters[0]
            previous_ft_lr_1 = current_parameters[1]
            previous_ft_lr_2 = current_parameters[2]

            if os.path.exists(dic_path):
                with open(dic_path, 'r') as f:
                    total_dic = json.load(f)


            total_dic[f"{way}_{shot}"] = current_parameters

            with open(dic_path, 'w') as f:
                json.dump(total_dic, f)

            if shot1:
                previous_shot1_epoch = current_parameters[0]
                previous_shot1_ft_lr_1 = current_parameters[1]
                previous_shot1_ft_lr_2 = current_parameters[2]
            shot1 = False
def critical_point_clip(is_shot,shot_list, dataset_idx_list, model_name, gpuid, imgsz=224):


    #value 无用

    names = list(all_roots.keys())


    for dataset_idx in dataset_idx_list:
        
        


        total_dic = {}
        
        if is_shot:
            way = 5
        else:
            shot = 5 
        shot1 = True


        for id_, num in enumerate(shot_list):
            previous_ft_epoch = 20
            # previous_ft_lr_1 = 0.001
            previous_ft_lr_1 = 0.0002
            # previous_ft_lr_1 = 2e-6
            

            # previous_ft_epoch = 60
            # previous_ft_lr_1 = 0.01
            # previous_ft_lr_2 = 0.5

            previous_shot1_epoch = previous_ft_epoch
            previous_shot1_ft_lr_1 = previous_ft_lr_1
            if is_shot:
                shot = num
            else:
                way = num
        # for shot in ["vary"]:
            dic_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], "final_results.json")
            if os.path.exists(dic_path):
                with open(dic_path, "r") as file_:
                    dic = json.load(file_)
                # if len(dic)>id_:
                if f"{way}_{shot}" in dic:
                    continue

            print(f"way: {way}, shot: {shot}")

            first_loop_flag = True

            status = ["",""]
            runs = 0
            while True:  
                runs+=1        
                parameters = []
                if first_loop_flag:
                    if shot1:
                        parameters.append(decide_epochs(previous_shot1_epoch, True, False))
                        if previous_ft_lr_1 == 0:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False, still=True))
                        else:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False))
                    else:
                        parameters.append(decide_epochs(previous_ft_epoch, False, False))
                        parameters.append(decide_lr(previous_ft_lr_1, False))
                    
                else:
                    #假设已经有了上一轮的status，以及parameters
                    assert len(status) == 2
                    ups = []
                    stills = []
                    print(status)
                    for sta in status:
                        if sta == "still":
                            ups.append(True)
                            stills.append(True)
                        elif sta == "up":
                            ups.append(True)
                            stills.append(False)
                        else:
                            ups.append(False)
                            stills.append(False)

                    parameters.append(decide_epochs(current_parameters[0], False, True,ups[0], stills[0]))

                    parameters.append(decide_lr(current_parameters[1], True,ups[1], stills[1]))
                        # print(parameters[i])
                # print(parameters)

                determine_test_config_clip(way, shot, parameters[0], parameters[1], dataset_idx, model_name, names[dataset_idx], gpuid, imgsz)
                
                tag = f"{model_name}/{names[dataset_idx]}/{way}w{shot}s/iteration_{runs}"
                
                try:
                    a = os.system(f"python search_hyperparameter.py --cfg configs/find_hyperparameter_{model_name}_{names[dataset_idx]}.yaml --is_train 0 --tag {tag}")
                except ValueError:
                    valueerror = True
                    print("meet value error, escape this dataset")
                    break
                else:
                    if a==256:
                        valueerror = True
                        print("meet value error, escape this dataset")
                        break
                valueerror = False

                json_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], f"{way}w{shot}s", f"iteration_{runs}", "results.json")

                with open(json_path) as _file:
                    no_U = json.load(_file)


                max_value = -1.
                position = [0,0]
                i = 0
                j = 0

                for value in no_U:
                    if value[3]>max_value:
                        max_value = value[3]
                        max_hyper = value[:3]
                        position = [i,j]
                    j += 1
                    if j == len(parameters[1]):
                        j=0
                        i += 1

                largest_index = position
                current_parameters = max_hyper
                

                
                if first_loop_flag:
                    for x in range(2):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            status[x] = "down"
                        elif index == len(parameters[x])-1:
                            status[x] = "up"
                        else:
                            status[x] = "still"
                else:
                    for x in range(2):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            if status[x] == "up":
                                status[x] = "still"
                            else:
                                status[x] = "down"
                        elif index == len(parameters[x])-1:
                            if status[x] == "up":
                                status[x] = "up"
                            else:
                                status[x] = "still"
                        else:
                            status[x] ="still"
                first_loop_flag = False
                # print(status)
                if "down" not in status and "up" not in status:
                    break
            
            if valueerror:
                break
            previous_ft_epoch = current_parameters[0]
            previous_ft_lr_1 = current_parameters[1]

            if os.path.exists(dic_path):
                with open(dic_path, 'r') as f:
                    total_dic = json.load(f)


            total_dic[f"{way}_{shot}"] = current_parameters

            with open(dic_path, 'w') as f:
                json.dump(total_dic, f)

            if shot1:
                previous_shot1_epoch = current_parameters[0]
                previous_shot1_ft_lr_1 = current_parameters[1]
            shot1 = False


def critical_point_linear(is_shot,shot_list, dataset_idx_list, model_name, gpuid, imgsz=224):

    #value 无用

    names = list(all_roots.keys())


    for dataset_idx in dataset_idx_list:
        
        


        total_dic = {}
        
        if is_shot:
            way = 5
        else:
            shot = 5 
        shot1 = True


        for id_, num in enumerate(shot_list):
            previous_ft_epoch = 50
            # previous_ft_lr_1 = 0.001
            # previous_ft_lr_1 = 0.0002
            previous_ft_lr_1 = 0.2
            # previous_ft_lr_1 = 2e-6
            

            # previous_ft_epoch = 60
            # previous_ft_lr_1 = 0.01
            # previous_ft_lr_2 = 0.5

            previous_shot1_epoch = previous_ft_epoch
            previous_shot1_ft_lr_1 = previous_ft_lr_1
            if is_shot:
                shot = num
            else:
                way = num
        # for shot in ["vary"]:
            dic_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], "final_results.json")
            if os.path.exists(dic_path):
                with open(dic_path, "r") as file_:
                    dic = json.load(file_)
                # if len(dic)>id_:
                if f"{way}_{shot}" in dic:
                    continue

            print(f"way: {way}, shot: {shot}")

            first_loop_flag = True

            status = ["",""]
            runs = 0
            while True:  
                runs+=1        
                parameters = []
                if first_loop_flag:
                    if shot1:
                        parameters.append(decide_epochs(previous_shot1_epoch, True, False))
                        if previous_ft_lr_1 == 0:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False, still=True))
                        else:
                            parameters.append(decide_lr(previous_shot1_ft_lr_1, False))
                    else:
                        parameters.append(decide_epochs(previous_ft_epoch, False, False))
                        parameters.append(decide_lr(previous_ft_lr_1, False))
                    
                else:
                    #假设已经有了上一轮的status，以及parameters
                    assert len(status) == 2
                    ups = []
                    stills = []
                    print(status)
                    for sta in status:
                        if sta == "still":
                            ups.append(True)
                            stills.append(True)
                        elif sta == "up":
                            ups.append(True)
                            stills.append(False)
                        else:
                            ups.append(False)
                            stills.append(False)

                    parameters.append(decide_epochs(current_parameters[0], False, True,ups[0], stills[0]))

                    parameters.append(decide_lr(current_parameters[1], True,ups[1], stills[1]))
                        # print(parameters[i])
                # print(parameters)

                determine_test_config_linear(way, shot, parameters[0], parameters[1], dataset_idx, model_name, names[dataset_idx], gpuid, imgsz)
                
                tag = f"{model_name}/{names[dataset_idx]}/{way}w{shot}s/iteration_{runs}"
                
                try:
                    a = os.system(f"python search_hyperparameter.py --cfg configs/find_hyperparameter_{model_name}_{names[dataset_idx]}.yaml --is_train 0 --tag {tag}")
                except ValueError:
                    valueerror = True
                    print("meet value error, escape this dataset")
                    break
                else:
                    if a==256:
                        valueerror = True
                        print("meet value error, escape this dataset")
                        break
                valueerror = False

                json_path = os.path.join(OUTPUT_ROOT, "find_hyperparameters", model_name, names[dataset_idx], f"{way}w{shot}s", f"iteration_{runs}", "results.json")

                with open(json_path) as _file:
                    no_U = json.load(_file)


                max_value = -1.
                position = [0,0]
                i = 0
                j = 0

                for value in no_U:
                    if value[3]>max_value:
                        max_value = value[3]
                        max_hyper = value[:3]
                        position = [i,j]
                    j += 1
                    if j == len(parameters[1]):
                        j=0
                        i += 1

                largest_index = position
                current_parameters = max_hyper
                

                
                if first_loop_flag:
                    for x in range(2):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            status[x] = "down"
                        elif index == len(parameters[x])-1:
                            status[x] = "up"
                        else:
                            status[x] = "still"
                else:
                    for x in range(2):
                        index = largest_index[x]
                        if len(parameters[x]) == 1:
                            status[x] = "still"
                        elif index == 0:
                            if status[x] == "up":
                                status[x] = "still"
                            else:
                                status[x] = "down"
                        elif index == len(parameters[x])-1:
                            if status[x] == "up":
                                status[x] = "up"
                            else:
                                status[x] = "still"
                        else:
                            status[x] ="still"
                first_loop_flag = False
                # print(status)
                if "down" not in status and "up" not in status:
                    break
            
            if valueerror:
                break
            previous_ft_epoch = current_parameters[0]
            previous_ft_lr_1 = current_parameters[2]

            if os.path.exists(dic_path):
                with open(dic_path, 'r') as f:
                    total_dic = json.load(f)


            total_dic[f"{way}_{shot}"] = current_parameters

            with open(dic_path, 'w') as f:
                json.dump(total_dic, f)

            if shot1:
                previous_shot1_epoch = current_parameters[0]
                previous_shot1_ft_lr_1 = current_parameters[2]
            shot1 = False




def find_all():

    # shot_list = [1,2,5,10,20,50]
    # shot_list = [0]
    shot_list = [1]
    # shot_list = [10,20,50]
    # way_list = [2,10,20,50]
    # way_list = [20]
    # dataset_idx_list = list(range(12))
    # dataset_idx_list = [0]
    dataset_idx_list = [10]


    # clip
    # dataset_idx_list = [0,2,3,4,5,6,8,9,10,11,12]

    # dataset_idx_list = [3]
    # dataset_idx_list = list(range(2,12))
    # critical_point(True,shot_list, dataset_idx_list, "hyparameter_dataset_5way_find", 6)
    # critical_point(True,shot_list, dataset_idx_list, "clip_visualonly_finetune_5way", 0)
    # critical_point(True,shot_list, dataset_idx_list, "clip_visualonly_LoRA_5way", 1)
    # critical_point(True,shot_list, dataset_idx_list, "clip_visualonly_finetune_5way_CUB", 0)
    critical_point(True,shot_list, dataset_idx_list, "clip_visualonly_LoRA_5way_CUB", 1)
    # critical_point_clip(True,shot_list, dataset_idx_list, "clip_maple_find_again", 0)
    # critical_point_clip(True,shot_list, dataset_idx_list, "clip_CoCoOp", 7)
    # critical_point_clip(False,way_list, dataset_idx_list, "clip", 3)
    # critical_point_linear(True,shot_list, dataset_idx_list, "clip_visualonly_linear", 1)
    


    # critical_point()

# sys.stdout = Logger(f"../CE_MDImageNet_shot.txt")
find_all()
# critical_point()
# test_all()
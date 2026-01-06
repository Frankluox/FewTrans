"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import collections
import torch
import time
import numpy as np

def accuracy_(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(output.shape)
    maxk = min(max(topk), output.size()[2])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 2, True, True)
    # print(pred)
    pred = torch.transpose(pred, 1, 2)
    # print('hh')
    # print(pred.shape)
    # pred = pred.t()
    pred, _ = torch.mode(pred, 0)
    # print(pred.shape)
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class FinetuneModule(nn.Module):
    def __init__(self, config, data_specs = None):
        super().__init__()
        # self.ft_lr_1s = [0.001,0.01,0.1]
        # self.ft_lr_1s = [0.0005,0.002,0.01,0.05,0.2]
        # self.ft_lr_1s = [0.00002,0.0001,0.0005,0.002,0.01]

        # finetune emsemble
        self.ft_lr_1s = [0.000002,0.00001,0.00005]
        self.ft_lr_2s = [0.0004,0.002,0.01]
        self.ft_epochs = 70

        # finetune ensemble *3
        # self.ft_lr_1s = [0.0000033333,0.00001,0.00003]
        # self.ft_lr_2s = [0.000666666,0.002,0.006]
        # self.ft_epochs = 70

        # finetune ensemble *10
        # self.ft_lr_1s = [0.000001,0.00001,0.0001]
        # self.ft_lr_2s = [0.0002,0.002,0.02]
        # self.ft_epochs = 70

        # finetune best
        # self.ft_lr_1s = [0.00001]
        # self.ft_lr_2s = [0.002]
        # self.ft_epochs = 70

        # LoRA emsemble
        # self.ft_lr_1s = [0.0004,0.002,0.01]
        # self.ft_lr_2s = [0.002,0.01,0.05]
        # self.ft_epochs = 40

        # LoRA best
        # self.ft_lr_1s = [0.002]
        # self.ft_lr_2s = [0.01]
        # self.ft_epochs = 40


        #finetune_vo
        # self.ft_lr_1s = [0.0001,0.0005,0.002]
        # 70	1.00E-05	0.02
        # self.ft_lr_1s = [5e-06,2e-05,1e-04]
        # self.ft_lr_2s = [0.01,0.05,0.2]




        # self.ft_lr_1s = [1e-07,1e-06,1e-05]
        # self.ft_lr_2s = [0.0005,0.005,0.05]

        # finetune afterwards
        # self.ft_lr_1s = [1e-07,1e-06,1e-05]
        # self.ft_lr_2s = [0.0005,0.005,0.05]
        # self.ft_lr_1s = [1e-05]
        # self.ft_lr_2s = [0.01]

        # self.ft_epochs = [80]
        # self.ft_epochs = 30
        # self.epoch_ensemble = True
        self.epoch_ensemble = False
        #vpt_vo
        # self.ft_lr_1s = [0.0005,0.002,0.01]
        # self.ft_lr_2s = [0.02,0.1,0.5]
        # self.ft_epochs = 15

        # self.ft_epochs = [80]

        
        if data_specs is not None:
            # print('spec')
            all_class_names = []
            for data_spec in data_specs:
                class_names = [data_spec["id2name"][i] for i in range(len(data_spec["id2name"]))]
                # print(class_names)
                all_class_names.append(class_names)
            self.backbone = get_backbone(config.MODEL.BACKBONE, all_class_names, *config.MODEL.BACKBONE_HYPERPARAMETERS)
            classifier_hyperparameters = [config, self.backbone]+config.MODEL.CLASSIFIER_PARAMETERS
            self.classifier = get_classifier(config.MODEL.CLASSIFIER, *classifier_hyperparameters)
        else:
            # print('no spec')
            self.config = config
            self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)

            # The last hyperparameter is the head mode
            self.mode = config.MODEL.CLASSIFIER_PARAMETERS[-1]
            classifier_hyperparameters = [config, self.backbone]+config.MODEL.CLASSIFIER_PARAMETERS
            self.classifier = get_classifier(config.MODEL.CLASSIFIER, *classifier_hyperparameters)
    
    def append_adapter(self):
        # append adapter to the backbone
        self.backbone = get_backbone("resnet_tsa",backbone=self.backbone)
        classifier_hyperparameters = [self.backbone]+self.config.MODEL.CLASSIFIER_PARAMETERS
        self.classifier = get_classifier(self.config.MODEL.CLASSIFIER, *classifier_hyperparameters)

    # def test_forward(self, tasks):
        
    #     all_accs = collections.defaultdict(list)
    #     self.classifier.ft_epoch = self.ft_epochs
    #     for task in tasks:
    #         all_scores = collections.defaultdict(list)
    #         for lr_backbone in self.ft_lr_1s:
    #             for lr_head in self.ft_lr_2s:
    #                 # print(f"lr_backbone: {lr_backbone}, lr_head: {lr_head}")
    #                 self.classifier.ft_lr_1 = lr_backbone
    #                 self.classifier.ft_lr_2 = lr_head
    #                 scores = self.classifier(task, self.epoch_ensemble)
    #                 # print(scores.keys())
    #                 for key, value in scores.items():
    #                     all_scores[key].extend(value)
            
    #         count = 1
    #         for key, value in all_scores.items():
    #             # print(key)
    #             # for values in value:
    #             #     print(values.shape)
    #             # print("hh")
    #             scores = torch.stack(value)
    #             # print(scores.shape)
    #             if key == "base":
    #                 all_accs[key].append(accuracy_(scores, task[0][3].squeeze_().cuda())[0])
    #             elif key == "novel":
    #                 all_accs[key].append(accuracy_(scores, task[0][6].squeeze_().cuda())[0])
    #             else:
    #                 all_accs[key].append(accuracy_(scores, task[count][1].squeeze_().cuda())[0])
    #                 count+=1

    #         # for key, value in acc.items():
    #         #     accs[key].append(value)
    #     return all_accs


    # 增加每个超参数组合的独立准确率记录
    def test_forward(self, tasks):
        all_accs = collections.defaultdict(list)
        self.classifier.ft_epoch = self.ft_epochs

        # [新增] 用于存储所有适配过程的耗时
        adaptation_times = []
        peak_memories = []

        for task in tasks:
            all_scores = collections.defaultdict(list)
            for lr_backbone in self.ft_lr_1s:
                for lr_head in self.ft_lr_2s:
                    self.classifier.ft_lr_1 = lr_backbone
                    self.classifier.ft_lr_2 = lr_head

                    # --- 计时开始 ---
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats() # 重置峰值计数器
                    start_time = time.perf_counter()

                    # 获取当前 LR 组合下的 scores (dict: {key: [score_tensors]})
                    scores_dict = self.classifier(task, self.epoch_ensemble) 

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    # --- 计时结束 ---

                    adaptation_times.append(end_time - start_time)

                    if torch.cuda.is_available():
                        # 获取显存峰值并转换为 MB
                        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                        peak_memories.append(peak_mem)
                    
                    # 为当前超参数组合生成后缀
                    hp_suffix = f"_lr1_{lr_backbone}_lr2_{lr_head}"
                    
                    # 记录该超参数组合的独立准确率
                    temp_count = 1
                    for key, value in scores_dict.items():
                        # 计算当前组合的独立 Acc (使用与 ensemble 相同的 accuracy_ 函数)
                        # 虽然这里只有一个组合，但 accuracy_ 兼容 stack 后的维度
                        individual_scores = torch.stack(value)
                        
                        # 确定对应的标签 (Label)
                        if key == "base":
                            labels = task[0][3].squeeze().cuda()
                        elif key == "novel":
                            labels = task[0][6].squeeze().cuda()
                        else:
                            labels = task[temp_count][1].squeeze().cuda()
                            temp_count += 1
                        
                        # 计算 Acc 并存入对应的独立 Key 中
                        acc_val = accuracy_(individual_scores, labels)[0]
                        all_accs[key + hp_suffix].append(acc_val)
                        
                        # 同时将 score 加入 all_scores 列表，用于最后的 Ensemble
                        all_scores[key].extend(value)
            
            # 计算最终的 Ensemble 结果 (原始逻辑)
            count = 1
            for key, value in all_scores.items():
                scores = torch.stack(value)
                if key == "base":
                    labels = task[0][3].squeeze().cuda()
                elif key == "novel":
                    labels = task[0][6].squeeze().cuda()
                else:
                    labels = task[count][1].squeeze().cuda()
                    count += 1
                # 存入原始 Key (如 "base" 或 "novel")，代表 Ensemble 结果
                all_accs[key].append(accuracy_(scores, labels)[0])

        # [新增] 在返回前打印资源消耗报告
        avg_time = np.mean(adaptation_times)
        avg_mem = np.mean(peak_memories) if peak_memories else 0
        max_mem = np.max(peak_memories) if peak_memories else 0
        
        print(f"\n" + "="*50)
        print(f"RESOURCE ANALYSIS REPORT")
        print(f"-"*50)
        print(f"Single Adaptation Time: {avg_time:.4f}s")
        print(f"Peak GPU Memory Usage: {avg_mem:.2f} MB (Max: {max_mem:.2f} MB)")
        print(f"Estimated 600-task Total (3x3 grid): {(avg_time * 9 * 600 / 3600):.2f} hours")
        print(f"="*50 + "\n")
                
        return all_accs

def get_model(config, data_specs):
    return FinetuneModule(config, data_specs)
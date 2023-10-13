"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import collections
import torch

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

        #finetune_vo
        # self.ft_lr_1s = [0.0001,0.0005,0.002]
        # 70	1.00E-05	0.02
        # self.ft_lr_1s = [5e-06,2e-05,1e-04]
        # self.ft_lr_2s = [0.01,0.05,0.2]


        # self.ft_lr_1s = [1e-07,1e-06,1e-05]
        # self.ft_lr_2s = [0.0005,0.005,0.05]

        self.ft_lr_1s = [1e-07,1e-06,1e-05]
        self.ft_lr_2s = [0.0005,0.005,0.05]
        # self.ft_lr_1s = [1e-05]
        # self.ft_lr_2s = [0.01]

        # self.ft_epochs = [80]
        self.ft_epochs = 30
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

    def test_forward(self, tasks):
        
        all_accs = collections.defaultdict(list)
        self.classifier.ft_epoch = self.ft_epochs
        for task in tasks:
            all_scores = collections.defaultdict(list)
            for lr_backbone in self.ft_lr_1s:
                for lr_head in self.ft_lr_2s:
                    # print(f"lr_backbone: {lr_backbone}, lr_head: {lr_head}")
                    self.classifier.ft_lr_1 = lr_backbone
                    self.classifier.ft_lr_2 = lr_head
                    scores = self.classifier(task, self.epoch_ensemble)
                    # print(scores.keys())
                    for key, value in scores.items():
                        all_scores[key].extend(value)
            
            count = 1
            for key, value in all_scores.items():
                # print(key)
                # for values in value:
                #     print(values.shape)
                # print("hh")
                scores = torch.stack(value)
                # print(scores.shape)
                if key == "base":
                    all_accs[key].append(accuracy_(scores, task[0][3].squeeze_().cuda())[0])
                elif key == "novel":
                    all_accs[key].append(accuracy_(scores, task[0][6].squeeze_().cuda())[0])
                else:
                    all_accs[key].append(accuracy_(scores, task[count][1].squeeze_().cuda())[0])
                    count+=1

            # for key, value in acc.items():
            #     accs[key].append(value)
        return all_accs

def get_model(config, data_specs):
    return FinetuneModule(config, data_specs)
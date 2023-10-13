"""
A unified implementation of gradient-based adaptation-time classifiers,
including finetune, URL and cosine classifer.
"""
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
import collections
import torch.nn.functional as F

def compute_prototypes(features_support, labels_support):
    #[n_way,num_support]
    labels_support_transposed = labels_support.transpose(0, 1)

    #[n_way, dim]
    prototypes = torch.mm(labels_support_transposed, features_support)
    labels_support_transposed = (labels_support_transposed.sum(dim=1, keepdim=True)+1e-12).expand_as(prototypes)
    prototypes = prototypes.div(
        labels_support_transposed
    )
    #[n_way,dim]
    return prototypes

def prototype_scores(support_embeddings, support_labels,
                   query_embeddings):
    one_hot_label = F.one_hot(support_labels,num_classes = torch.max(support_labels).item()+1).float()
    support_embeddings = F.normalize(support_embeddings, p=2, dim=1, eps=1e-12)
    # [n_way,dim]
    prots = compute_prototypes(support_embeddings, one_hot_label)
    prots = F.normalize(prots, p=2, dim=1, eps=1e-12)

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1, eps=1e-12)
    classification_scores = torch.mm(query_embeddings, prots.transpose(0, 1))*10

    return classification_scores

class FinetuneModel(torch.nn.Module):
    """
    the overall finetune module that incorporates a backbone and a head.
    """
    def __init__(self, backbone, way, device, ncc = False, use_linear_head = True):
        super().__init__()
        '''
        backbone: the pre-trained backbone
        way: number of classes
        device: GPU ID
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: Use fc head or PN head to finetune.
        '''
        self.backbone = deepcopy(backbone).to(device)

        self.use_linear_head = use_linear_head
        self.ncc = ncc
        assert ncc or use_linear_head
        if use_linear_head:
            if ncc:
                self.Linear_head = nn.Linear(backbone.outdim, backbone.outdim, bias = False).to(device)
                self.Linear_head.weight.data = torch.eye(backbone.outdim, backbone.outdim).to(device)
            else:
                self.Linear_head = nn.Linear(backbone.outdim, way).to(device)
                self.Linear_head.weight.data.fill_(1)
                self.Linear_head.bias.data.fill_(0)

    def forward(self, x, labels=None, backbone_grad = True, training = True):
        # print('hh')
        # turn backbone_grad off if backbone is not to be finetuned
        if backbone_grad:
            # print("heihei")
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)
        if x.dim() == 4:
            # print('hh')
            x = F.adaptive_avg_pool2d(x, 1).squeeze_(-1).squeeze_(-1)
            x = F.normalize(x, dim=1)
        else:
            # pass
            x = F.normalize(x, dim=1)
        if self.use_linear_head:
            x = self.Linear_head(x)
            if self.ncc:
                x = F.normalize(x, dim=1)
        if training:
            if self.ncc:
                x = prototype_scores(x, labels,
                                        x)
            loss = F.cross_entropy(x, labels)
            return loss
        return x


class Finetuner(nn.Module):
    def __init__(self, config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1, mode = "vpt", head = "fc", use_linear = True):
        '''
        backbone: the pre-trained backbone
        ft_batchsize: batch size for finetune
        feed_query_batchsize: max number of query images processed once (avoid memory issues)
        ft_epoch: epoch of finetune
        ft_lr_1: backbone learning rate
        ft_lr_2: head learning rate
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: the classification head--"fc", "NCC" or "cc"
        '''
        super().__init__()
        self.ft_batchsize = ft_batchsize
        self.feed_query_batchsize = feed_query_batchsize
        self.ft_epoch = ft_epoch
        self.ft_lr_1 = ft_lr_1
        self.ft_lr_2 = 0.
        self.backbone = backbone
        self.config = config
        self.mode = mode
        self.head = head
        self.use_linear = use_linear

    def forward(self, tasks, epoch_ensemble = True) -> Tensor:
        """Take one task of few-shot support examples and query examples as input,
            output the logits of each query examples.

        Args:
            query_images: query examples. size: [num_query, c, h, w]
            support_images: support examples. size: [num_support, c, h, w]
            support_labels: labels of support examples. size: [num_support, way]
        Output:
            classification_scores: The calculated logits of query examples.
                                   size: [num_query, way]
        """
        # tasks元素是一个数据集，第0个就是训练用的数据集。
        # 每个数据集元素顺序：(均是optional) base_support_image, base_support_label, base_query_image, base_query_label,base_class_id, novel_query_image, novel_query_label, novel_class_id
        support_images = tasks[0][0].squeeze_().cuda()
        query_images = tasks[0][2].squeeze_().cuda()
        support_labels = tasks[0][1].squeeze_().cuda()
        # query_labels = tasks[0][3].squeeze_().cuda()
        # base_class_ids = torch.cat(tasks[0][4])

        # print(len(support_labels))
        # print(support_labels)
        # print(query_images.shape)
        # print(len(tasks))



        # labels = support_labels
        # # print(labels.shape)
        # class_id = 0
        # count = 0
        # shot_num = 0

        # for i in range(len(labels)):
        #     # print(labels)
        #     if labels[i].item() == class_id:
        #         count += 1
        #     else:
        #         if labels[i].item() == 0:
        #             shot_num += count
        #             print(f"class {class_id}: {count} support images")
        #             break
        #         else:
        #             print(f"class {class_id}: {count} support images")
        #             shot_num += count
        #             count = 1
        #             class_id += 1
        # shot_num += count
        # # print(shot_num)
        # # print(len(labels))
        # # print(labels)
        # assert shot_num==len(labels)
        # # shot.append(shot_num/(class_id+1))
        # # support_size.append(shot_num)
        # print(f"class {class_id}: {count} support images")
        # way.append(class_id+1)
        # print(idx)
        


        support_size = support_images.size(0)

        device = support_images.device

        way = torch.max(support_labels).item()+1

        ncc = self.head == "NCC"

        model = FinetuneModel(self.backbone, way, device, ncc, self.use_linear)
        # model = deepcopy(self.backbone).to(device)

        # print("lr1", self.ft_lr_1)
        # print("lr2", self.ft_lr_2)



        # By default, SGD is adopted as the optimizer. Other optimizers like Adam can be used as well.

        # set_optimizer_1 = torch.optim.SGD(model.parameters(), lr = self.ft_lr_1, momentum=0.9)
        # set_optimizer_1 = torch.optim.Adam(model.parameters(), lr = self.ft_lr_1)


        if not self.mode == "linear":
            set_optimizer_1 = torch.optim.Adam(model.backbone.parameters(), lr = self.ft_lr_1)
            # set_optimizer_1 = torch.optim.SGD(model.backbone.parameters(), lr = self.ft_lr_1, momentum=0.9)

        set_optimizer_2 = torch.optim.Adam(model.Linear_head.parameters(), lr = self.ft_lr_2)

        # print(self.ft_lr_1)
        # print(self.ft_lr_2)
        # print(self.ft_epoch)

        model.eval()

        name_to_update = "prompt_learner"

        if self.mode == "vpt" or self.mode == "MaPLe":
            for name, param in model.named_parameters():
                if name_to_update not in name:
                    # Make sure that VPT prompts are updated
                    if "VPT" in name or "Linear_head" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        elif self.mode == "linear":
            for name, param in model.named_parameters():
                if "Linear_head" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif self.mode == "bias":
            # name_to_update = "prompt_learner"
            for name, param in model.named_parameters():
                if "bias" in name or "Linear_head" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif self.mode == "adapter" or self.mode == "adaptformer" or self.mode == "TSA":
            for name, param in model.named_parameters():
                if "adapter" in name or "Linear_head" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif self.mode == "SSF":
            for name, param in model.named_parameters():
                if "ssf" in name or "Linear_head" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        

        # enabled = set()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if "Linear" in name:
        #             print(name)
        #         enabled.add(name)
        # print(f"Parameters to be updated: {enabled}")


        # for name, param in model.named_parameters():
        #     # if name_to_update not in name:
        #         # Make sure that VPT prompts are updated
        #     if "VPT" not in name:
        #         print(param.requires_grad)
                # _(True)
            # else:
                # param.requires_grad_(False)
                
        # total finetuning steps
        global_steps = self.ft_epoch*((support_size+self.ft_batchsize-1)//self.ft_batchsize)
        _rng = np.random.RandomState(11)

        step = 0
        all_scores = collections.defaultdict(list)
        # total_classification_scores = 0.
        # print(self.ft_epoch)
        # print(self.ft_epoch)
        # self.ft_epoch = 0
        for epoch in range(self.ft_epoch):
            # print(epoch)
            # print("epoch ", epoch)
            with torch.enable_grad():
                # randomly suffule support set
                rand_id = _rng.permutation(support_size)
                # print(f"epoch:{epoch}")
                for i in range(0, support_size , self.ft_batchsize):
                    # print("iteration ", i)
                    # print(f"iteration:{i}")
                    # by default, cosine LR shedule is used.
                    lr_1 = 0.5 * self.ft_lr_1* (1. + math.cos(math.pi * step / global_steps))
                    lr_2 = 0.5 * self.ft_lr_2* (1. + math.cos(math.pi * step / global_steps))
                    if not self.mode == "linear":
                        for param_group in set_optimizer_1.param_groups:
                            param_group["lr"] = lr_1
                    for param_group in set_optimizer_2.param_groups:
                        param_group["lr"] = lr_2
                    if not self.mode == "linear":
                        set_optimizer_1.zero_grad()
                    set_optimizer_2.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = support_images[selected_id]
                    label_batch = support_labels[selected_id] 
                    # loss = model(train_batch, label_batch, dataset_idx = 0,training=True,  class_ids=base_class_ids)
                    if not self.mode == "linear":
                        loss = model(train_batch, label_batch)
                    else:
                        loss = model(train_batch, label_batch, backbone_grad = False)
                    loss.backward()

                    # for name, param in model.named_parameters():
                    #     if name_to_update not in name:
                    #         # Make sure that VPT prompts are updated
                    #         if "VPT" in name:
                    #             print(param.grad)
                    if not self.mode == "linear":
                        # print("update")
                        set_optimizer_1.step()
                    set_optimizer_2.step()
                    step += 1
            
            relevant = self.ft_epoch//3+1


            if epoch_ensemble:
                if (epoch+1)%relevant == 0 or epoch+1 == self.ft_epoch:
                    # print("hh")
                    model.eval()          
                    query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize  
                    if not self.head == "NCC":
                        # number of feed-forward calculations to calculate all query embeddings
                        
                        
                        base_scores = []
                        for run in range(query_runs):
                            # for non-NCC head, the model directly ouputs score
                            base_scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False))
                        base_classification_scores = torch.cat(base_scores, dim=0)
                    else:
                        support_features = []
                        query_features = []
                        support_runs = (support_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
                        for run in range(support_runs):
                            support_features.append(model(support_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,support_images.size(0))], training=False))
                        for run in range(query_runs):
                            query_features.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False))
                        support_features = torch.cat(support_features, dim=0)
                        query_features = torch.cat(query_features, dim=0)
                        base_classification_scores = prototype_scores(support_features, support_labels,
                            query_features)
                    all_scores["base"].append(base_classification_scores)

        if self.ft_epoch == 0 or not epoch_ensemble:
            model.eval()          
            query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize  
            if not self.head == "NCC":
                base_scores = []
                for run in range(query_runs):
                    # for non-NCC head, the model directly ouputs score
                    base_scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False))
                base_classification_scores = torch.cat(base_scores, dim=0)
            else:
                support_features = []
                query_features = []
                support_runs = (support_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
                for run in range(support_runs):
                    support_features.append(model(support_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,support_images.size(0))], training=False))
                for run in range(query_runs):
                    query_features.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False))
                support_features = torch.cat(support_features, dim=0)
                query_features = torch.cat(query_features, dim=0)
                base_classification_scores = prototype_scores(support_features, support_labels,
                    query_features)
            all_scores["base"].append(base_classification_scores)



        return all_scores

def create_model(config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,mode = "vpt", head = "fc", use_linear = True):
    return Finetuner(config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1, mode,head, use_linear)
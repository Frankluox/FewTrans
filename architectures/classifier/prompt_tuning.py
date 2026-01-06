"""
A unified implementation of gradient-based adaptation-time classifiers,
including finetune, URL and cosine classifer.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from .utils import CC_head, prototype_scores
from utils import accuracy
import collections
from torch.nn.modules.loss import _Loss



class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss



def prograd_backward_and_update(loss_a, loss_b, model, optimizer,lambda_=1):
    # loss_b not increase is okay
    # loss_a has to decline
    # self.model_zero_grad(names)
    optimizer.zero_grad()

    # get name of the model parameters
    # names = self.get_model_names(names)
    # backward loss_a

    if not torch.isfinite(loss_b).all():
        raise FloatingPointError("Loss is infinite or NaN!")

    loss_b.backward(retain_graph=True)

    # normalize gradient
    b_grads = []
    for p in model.parameters():
        if p.requires_grad:
            b_grads.append(p.grad.clone())
        else:
            b_grads.append(None)   

    optimizer.zero_grad()   

    # optimizer don't step
    # for name in names:
    #     self._optims[name].zero_grad()

    # backward loss_a
    if not torch.isfinite(loss_a).all():
        raise FloatingPointError("Loss is infinite or NaN!")

    loss_a.backward()


    for p, b_grad in zip(model.parameters(), b_grads):
        # calculate cosine distance
        if b_grad is None:
            assert not p.requires_grad
            continue
        b_grad_norm = b_grad / torch.linalg.norm(b_grad)
        a_grad = p.grad.clone()
        a_grad_norm = a_grad / torch.linalg.norm(a_grad)

        if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
            p.grad = a_grad - lambda_ * torch.dot(
                a_grad.flatten(), b_grad_norm.flatten()
            ) * b_grad_norm

    # optimizer
    optimizer.step()



class Finetuner(nn.Module):
    def __init__(self, config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1, mode = "vpt"):
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
        self.criterion = ProGradLoss(1.0)

    def forward(self, tasks, epoch_ensemble) -> Tensor:
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
        query_labels = tasks[0][3].squeeze_().cuda()
        base_class_ids = torch.cat(tasks[0][4])

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

        
        model = deepcopy(self.backbone).to(device)



        # By default, SGD is adopted as the optimizer. Other optimizers like Adam can be used as well.

        # set_optimizer_1 = torch.optim.SGD(model.parameters(), lr = self.ft_lr_1, momentum=0.9)
        set_optimizer_1 = torch.optim.Adam(model.parameters(), lr = self.ft_lr_1)

        model.eval()

        name_to_update = "prompt_learner"

        if self.mode == "vpt" or self.mode == "MaPLe":
            for name, param in model.named_parameters():
                if name_to_update not in name:
                    # Make sure that VPT prompts are updated
                    if "VPT" in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        elif self.mode == "visualFT":
            # print(self.mode)
            for name, param in model.named_parameters():
                # Make sure that VPT prompts are updated
                if "text_encoder" in name or "scale" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        elif self.mode == "coOp" or self.mode == "CoCoOp" or self.mode == "ProGrad" or self.mode == "kgcoop":
            # print("coOp")
            # print(self.mode)
            for name, param in model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        elif self.mode == "textFT":
            for name, param in model.named_parameters():
                # Make sure that VPT prompts are updated
                if "image_encoder" in name or "scale" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        elif self.mode == "allFT":
            for name, param in model.named_parameters():
                # Make sure that VPT prompts are updated
                if "scale" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

        # elif self.mode == "ProGrad":



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
            with torch.enable_grad():
                # randomly suffule support set
                rand_id = _rng.permutation(support_size)
                # print(f"epoch:{epoch}")
                for i in range(0, support_size , self.ft_batchsize):
                    # print(f"iteration:{i}")
                    # by default, cosine LR shedule is used.
                    lr_1 = 0.5 * self.ft_lr_1* (1. + math.cos(math.pi * step / global_steps))
                    for param_group in set_optimizer_1.param_groups:
                        param_group["lr"] = lr_1

                    

                    set_optimizer_1.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = support_images[selected_id]
                    label_batch = support_labels[selected_id] 

                    if self.mode == "ProGrad":
                        # print("ProGrad")
                        with torch.no_grad():
                            logits_zs = self.backbone(train_batch, label_batch, dataset_idx = 0,training=False,  class_ids=base_class_ids, prograd = True)
                        logits = model(train_batch, label_batch, dataset_idx = 0,training=False,  class_ids=base_class_ids)
                        xe_loss, kl_loss = self.criterion(logits,
                                            logits_zs.detach(),
                                            label_batch)
                        prograd_backward_and_update(xe_loss, kl_loss, model, set_optimizer_1)
                    else:
                        if self.mode == "kgcoop":
                            loss = model(train_batch, label_batch, dataset_idx = 0,training=True,  class_ids=base_class_ids, kgcoop = True)
                        else:
                            loss = model(train_batch, label_batch, dataset_idx = 0,training=True,  class_ids=base_class_ids)
                        loss.backward()
                        set_optimizer_1.step()
                    step += 1
            if epoch_ensemble:
                relevant = self.ft_epoch//3+1
                # if relevant == 0:
                #     relevant = 1

                if (epoch+1)%relevant == 0 or epoch+1 == self.ft_epoch:
                    # print("hh")
                    model.eval()            

                    # number of feed-forward calculations to calculate all query embeddings
                    query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize

                    
                    
                    base_scores = []
                    for run in range(query_runs):
                        # for non-NCC head, the model directly ouputs score
                        base_scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False, dataset_idx = 0, class_ids=base_class_ids))
                    base_classification_scores = torch.cat(base_scores, dim=0)

                    all_scores["base"].append(base_classification_scores)
                # loss = F.cross_entropy(classification_scores, query_labels)
                    for i, task in enumerate(tasks):
                        # if "novel_images" in task:
                            # print(i)
                        if i == 0:
                            if not self.config.DATA.BASE2NOVEL:
                                continue
                            # print(i)
                            novel_images = task[5].squeeze_().cuda()
                            # print(novel_images.shape)
                            
                            novel_labels = task[6].squeeze_().cuda()
                            novel_class_ids = torch.cat(task[7])
                        else:
                            novel_images = task[0].squeeze_().cuda()
                            # print(novel_images.shape)
                            
                            novel_labels = task[1].squeeze_().cuda()
                            novel_class_ids = torch.cat(task[2])
                            # print(novel_class_ids)
                        novel_scores = []
                        novel_runs = (novel_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
                        for run in range(novel_runs):
                            novel_scores.append(model(novel_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,novel_images.size(0))], training=False, dataset_idx = i, class_ids=novel_class_ids))
                        
                        novel_classification_scores = torch.cat(novel_scores, dim=0)
                        if i==0:
                            all_scores["novel"].append(novel_classification_scores)
                        else:
                            all_scores[i].append(novel_classification_scores)

        if self.ft_epoch == 0 or not epoch_ensemble:
            model.eval()            

            # number of feed-forward calculations to calculate all query embeddings
            query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize

            
            
            base_scores = []
            for run in range(query_runs):
                # for non-NCC head, the model directly ouputs score
                base_scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], training=False, dataset_idx = 0, class_ids=base_class_ids))
            base_classification_scores = torch.cat(base_scores, dim=0)

            all_scores["base"].append(base_classification_scores)
        # loss = F.cross_entropy(classification_scores, query_labels)
            for i, task in enumerate(tasks):
                if i == 0:
                    if not self.config.DATA.BASE2NOVEL:
                        continue
                    # print(i)
                    novel_images = task[5].squeeze_().cuda()
                    # print(novel_images.shape)
                    
                    novel_labels = task[6].squeeze_().cuda()
                    novel_class_ids = torch.cat(task[7])
                else:
                    novel_images = task[0].squeeze_().cuda()
                    # print(novel_images.shape)
                    
                    novel_labels = task[1].squeeze_().cuda()
                    novel_class_ids = torch.cat(task[2])
                    
                # print(novel_class_ids)
                novel_scores = []
                novel_runs = (novel_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
                for run in range(novel_runs):
                    novel_scores.append(model(novel_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,novel_images.size(0))], training=False, dataset_idx = i, class_ids=novel_class_ids))
                
                novel_classification_scores = torch.cat(novel_scores, dim=0)
                if i==0:
                    all_scores["novel"].append(novel_classification_scores)
                else:
                    all_scores[i].append(novel_classification_scores)



        return all_scores

def create_model(config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,mode = "vpt"):
    return Finetuner(config, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1, mode)
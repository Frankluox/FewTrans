"""
Searching hyperparameters for gradient-based adaptation algorithms
such as finetune, eTT, TSA, URL and Cosine Classifier.
"""

import argparse
from config import get_config
import os
from logger import create_logger
from data import create_torch_dataloader
from data.dataset_spec import Split
import torch
import numpy as np
import random
import json
from utils import accuracy, AverageMeter, load_pretrained
from models import get_model
import math
import collections


def setup_seed(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_option():
    parser = argparse.ArgumentParser('Searching hyperparameters', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--test-batch-size', type=int, help="test batch size for single GPU")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--pretrained', type=str, help="pretrained path") 
    parser.add_argument('--tag', help='tag of experiment')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

@torch.no_grad()
def testing(config, dataset, data_loader, model):
    model.eval()

    # loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    accs = collections.defaultdict(list)
    # dataset.set_epoch()
    for idx, batches in enumerate(data_loader):

        acc = model.test_forward(batches)

        for key, value in acc.items():
            accs[key].extend(value)
        acc = torch.mean(torch.stack(acc["base"]))

        # loss_meter.update(loss.item())
        acc_meter.update(acc.item())


    # accs = torch.stack(accs)
    for key, value in accs.items():
        value = torch.stack(value)
        accs[key] = [torch.mean(value).item(), (1.96*torch.std(value)/math.sqrt(value.shape[0])).item()]
    # ci = (1.96*torch.std(accs)/math.sqrt(accs.shape[0])).item()
    return accs


def search_hyperparameter(config):
    valid_dataloader, valid_dataset = create_torch_dataloader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    if "clip_prompt" in config.MODEL.BACKBONE:
        dataset_specs = valid_dataset.dataset_specs
    else:
        dataset_specs = None
    model = get_model(config, dataset_specs).cuda()

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)

    if hasattr(model, 'mode') and model.mode == "NCC":
        model.append_adapter()


    logger.info("Start searching for hyperparameters.")

    epoch_range = config.SEARCH_HYPERPARAMETERS.EPOCH_RANGE
    lr_backbone_range = config.SEARCH_HYPERPARAMETERS.LR_BACKBONE_RANGE
    lr_head_range = config.SEARCH_HYPERPARAMETERS.LR_HEAD_RANGE
    base_only = config.SEARCH_HYPERPARAMETERS.BASE_ONLY

    # print(lr_backbone_range)
    if lr_backbone_range is None:
        lr_backbone_range = [0]
    if lr_head_range is None:
        lr_head_range = [0]
    
    path = os.path.join(config.OUTPUT, "results.json")
    with open(path, 'w') as f:
        dic = []
        json.dump(dic, f)

    # [accuracy, confidence interval]
    max_accuracy = [0.,0.]
    logger.info(f"epoch range: {epoch_range}, backbone lr range: {lr_backbone_range}, head lr range: {lr_head_range}")
    for epoch in epoch_range:
        for lr_backbone in lr_backbone_range:
            for lr_head in lr_head_range:
                model.classifier.ft_epoch = epoch
                model.classifier.ft_lr_1 = lr_backbone
                model.classifier.ft_lr_2 = lr_head
                accs = testing(config, valid_dataset, valid_dataloader, model)
                acc1 = accs["base"][0]
                ci1 = accs["base"][1]       
                if base_only:
                    logger.info(f"Test Accuracy with epoch: {epoch}, backbone lr: {lr_backbone}, head lr: {lr_head} is base: {acc1:.2f}%+-{ci1:.2f}")
                else:
                    acc2 = accs["novel"][0]
                    ci2 = accs["novel"][1]
                    logger.info(f"Test Accuracy with epoch: {epoch}, backbone lr: {lr_backbone}, head lr: {lr_head} is base: {acc1:.2f}%+-{ci1:.2f}, novel: {acc2:.2f}%+-{ci2:.2f}")

                
                if acc1>max_accuracy[0]:
                    if base_only:
                        max_accuracy = [acc1, ci1]
                    else:
                        max_accuracy = [acc1, ci1, acc2, ci2]
                    max_hyperparameters = (epoch, lr_backbone, lr_head)
                    logger.info("achieve new best.")

                
                with open(path, 'r') as f:
                    dic = json.load(f)
                if base_only:
                    dic.append([epoch, lr_backbone, lr_head, acc1, ci1])
                else:
                    dic.append([epoch, lr_backbone, lr_head, acc1, ci1, acc2, ci2])
                with open(path, 'w') as f:
                    json.dump(dic, f)
    if base_only:
        logger.info(f"base best accuracy base: {max_accuracy[0]:.2f}%+-{max_accuracy[1]:.2f} is achieved when epoch is {max_hyperparameters[0]}, backbone lr is {max_hyperparameters[1]}, head lr is {max_hyperparameters[2]}.")
    else:
        logger.info(f"base best accuracy base: {max_accuracy[0]:.2f}%+-{max_accuracy[1]:.2f}, novel: {max_accuracy[2]:.2f}%+-{max_accuracy[3]:.2f} is achieved when epoch is {max_hyperparameters[0]}, backbone lr is {max_hyperparameters[1]}, head lr is {max_hyperparameters[2]}.")



if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(config.GPU_ID)
    
    setup_seed(config.SEED)
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    
    assert isinstance(config.SEARCH_HYPERPARAMETERS.EPOCH_RANGE, list)
    assert isinstance(config.SEARCH_HYPERPARAMETERS.LR_HEAD_RANGE, list) or isinstance(config.SEARCH_HYPERPARAMETERS.LR_BACKBONE_RANGE, list)
    
    search_hyperparameter(config)

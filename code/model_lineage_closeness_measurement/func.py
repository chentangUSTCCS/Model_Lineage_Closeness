# author: TangChen
# email: chen_tang1999@163.com
# date: 

'''
GitHub: https://github.com/Corgiperson, welcome to star my project!
description: 
'''
import torch
import torchvision
import torch.nn as nn
import numpy as np
import logging
import random
from scipy import spatial
import sys
sys.path.append('./data')
# from utils import Utils
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
# from benchmark import ImageBenchmark
from dataset.flower102 import *
from dataset.stanford_dog import SDog120
from dataset.mit67 import MIT67
from model.fe_resnet import resnet18_dropout
from model.fe_mobilenet import MobileNetV2 as mbnetv2
from tqdm import trange
import yaml


def get_dataloader(dataset_id, datasets_dir='../data', split='train', batch_size=64, shuffle=True, seed=98, shot=-1):
    """
    Get the torch Dataset object
    :param dataset_id: the name of the dataset, should also be the dir name and the class name
    :param split: train or test
    :param batch_size: batch size
    :param shot: number of training samples per class for the training dataset. -1 indicates using the full dataset
    :return: torch.utils.data.DataLoader instance
    """
    datapath = os.path.join(datasets_dir, dataset_id)
    # print(datapath)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    from torchvision import transforms
    if split == 'train':
        dataset = eval(dataset_id)(
            datapath, True, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            shot, seed, preload=False
        )
    else:
        dataset = eval(dataset_id)(
            datapath, False, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            shot, seed, preload=False
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return data_loader

def get_model(pth, classes):
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'quantize' in pth:
        device_name = 'cpu'
    device = torch.device(device_name)
    print("Using {}".format(device))

    # set the test params path
    # test_pth_path = os.path.abspath(get_sys_config_yaml(get_program_config_path())['test_model_pth_path'])
    test_pth_path = pth
    print(test_pth_path)

    model_archs = ['resnet18', 'mbnetv2']

    # set steal models
    if 'steal' in test_pth_path:
        steal_model_str = test_pth_path.split('-')[-2]
        for model_arch in model_archs:
            if model_arch in steal_model_str:
                if 'res' in model_arch:
                    # logging.info("Loading model:{} …………".format(model_arch))
                    model = eval(f'{model_arch}_dropout')(
                        pretrained=False,
                        num_classes=classes
                    )
                elif 'mbnet' in model_arch:
                    # logging.info("Loading model:{} …………".format(model_arch))
                    model = eval(f'{model_arch}')(
                        num_classes=classes
                    )
    else:
        # set normal models
        for model_arch in model_archs:
            # set steal model
            if model_arch in test_pth_path:
                if 'res' in model_arch:
                    # logging.info("Loading model:{} …………".format(model_arch))
                    model = eval(f'{model_arch}_dropout')(
                        pretrained=False,
                        num_classes=classes
                    )
                elif 'mbnet' in model_arch:
                    # logging.info("Loading model:{} …………".format(model_arch))
                    model = eval(f'{model_arch}')(
                        num_classes=classes
                    )
                break

    # after loading models, set quantize models
    if 'quantize' in test_pth_path:
        dtype = torch.qint8 if 'qint' in test_pth_path else torch.float16
        model = torch.quantization.quantize_dynamic(model, dtype=dtype)

    print(f'Loading model {model_arch} done.')
    # logging.info("Load model : Done.")

    # load model parameters
    # logging.info("Loading the model params dict …………")
    checkpoint = torch.load(test_pth_path)
    model.load_state_dict(checkpoint['state_dict'])
    # logging.info("Load model params: Done.")
    model = model.to(device)
    model.eval()
    return model

# get_model('F:/application\git\myProject\method3\ModelDiff\models\pretrain(mbnetv2,ImageNet)-/'+'final_ckpt.pth', 1000)

# dataset_ids = ['SDog120', 'Flower102', 'MIT67']
# num_classes_dict = {
#     'SDog120' : 120,
#     'Flower102': 102,
#     'MIT67' : 67
# }

# # set num classes
# for dataset_id in num_classes_dict:
#     test_dataloader = get_dataloader(dataset_id=dataset_id,split='train',batch_size=32)
#     print(len(test_dataloader))
#     print(f'{dataset_id} loading done.')
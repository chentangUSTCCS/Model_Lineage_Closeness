'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from func import get_model, get_dataloader, get_two_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, produce_plot
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from torch.utils.data import DataLoader, Dataset


args = options().parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net

# Data/other training stuff`
torch.manual_seed(args.set_data_seed)
# ------HERE to specify the dataset --------
trainloader, testloader = get_dataloader(dataset_id='Flower102', split='train'),get_dataloader(dataset_id='Flower102', split='test')
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
# ------HERE to specify the model path --------
fix_path = 'F:/application\git\myProject\method3\ModelDiff/models/'
net = get_model(fix_path + 'train(mbnetv2,Flower102)-'+'/final_ckpt.pth', 102)


criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)
elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# Train or load base network
print("Loading the network")
best_acc = 0  # best test accuracy
best_epoch = 0


image_ids = args.imgs
images = [trainloader.dataset[i][0] for i in image_ids]
labels = [trainloader.dataset[i][1] for i in image_ids]

sampleids = '_'.join(list(map(str,image_ids)))
planeloader = make_planeloader(images, args)
preds = decision_boundary(args, net, planeloader, device)
from utils import produce_plot_sepleg
net_name = args.net
# ------HERE to specify the save path --------
plot_path = os.path.join('F:/application\git\myProject\dbViz-main/result/add_experiemtn/',f'cnn_cifar10_result')
os.makedirs(f'{args.plot_path}', exist_ok=True)

produce_plot_sepleg(plot_path, preds, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)

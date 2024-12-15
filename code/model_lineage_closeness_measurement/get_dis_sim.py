import os
import torch, copy, argparse
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from func import get_model, get_dataloader

def evalu(model, testloader):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return accuracy


def fast_gradient_signed(x, y, model, eps):
    device = 'cuda'
    criterion = nn.CrossEntropyLoss().to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y.long())
    model.zero_grad()
    loss.backward()
    sign = torch.sign(x.grad)
    while 1:
        y_pred = model(x + sign * eps)
        _, pred_labels = torch.max(y_pred, 1)
        if not torch.eq(pred_labels, y).item():
            break
        x = (x + sign * eps).detach_()
    return x.detach_()


def gen_adversaries(model, l, dataset, eps):
    # 可以先找出适合成为对抗样本的样本，再精修到决策边界附近
    device = 'cuda'
    advs = []
    count = 0
    for x, y in dataset:
        # generate adversaries
        print('trigger sets setting...   [{:0>3d}/{:0>3d}]'.format(count, l), end='\r')
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        x_advs = fast_gradient_signed(x, y, model, eps)
        advs.append((x_advs,y))
        if len(advs) == l: break

    return advs


def calculate_distance(model, sur_model, trainloader):
    result1, result2 = [], []  # adv, distance, label, label'
    device = 'cuda'
    model.eval()
    sur_model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    for (img, label) in tqdm.tqdm(trainloader):
        img, label = img.to(device), label.to(device)
        img.requires_grad = True
        model.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, label.long())
        loss.backward()
        sign = torch.sign(img.grad)

        # model-calculate
        # cot = 0
        dis1 = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        d1 = 0
        x = img.detach()
        for distance1 in dis1:
            while 1:
                y_pred = model(x + sign * distance1)
                _, pred_labels1 = torch.max(y_pred, 1)
                if not torch.eq(pred_labels1, label).item():
                    break

                x = (x + sign * distance1).detach()
                d1 += distance1
                # cot += 1
                if d1 > 10: break
            if d1 > 10: break
        if d1 > 10: continue
        # sur_model-calculate
        # 先检查输出label是否一样，是否为有效样本
        y_pr = sur_model(img)
        _, pr = torch.max(y_pr, 1)
        if not torch.eq(pr, label).item():
            continue

        # count = 0
        dis = [0.100, 0.010, 0.001, 0.0001, 0.00001]
        d = 0
        x = img.detach()
        for distance in dis:
            while 1:
                y_pred = sur_model(x + sign * distance)
                _, pred_labels2 = torch.max(y_pred, 1)
                if not torch.eq(pred_labels2, label).item():
                    break

                x = (x + sign * distance).detach_()
                d += distance
                # count += 1
                if d > 10: break
            if d > 10: break
        if d > 10: continue
        if pred_labels1 == pred_labels2:
            result1.append(round(d1, 5))
            result2.append(round(d, 5))
    return result1, result2

def cal_similarity(result1 , result2):
    return (sum(result1)-sum(result2))/len(result1)



def main(dataset, SourceModelPath, SuspiciousModelPath, outputSize):
    device = 'cuda'
    trainloader = get_dataloader(dataset_id = dataset, batch_size = 1)
    fix_path = './models/'
    model = get_model(fix_path + SourceModelPath+'/final_ckpt.pth', outputSize)
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=102, bias=True)

    model.to(device)
    model.eval()
    surrogate_model = get_model(
        fix_path + SuspiciousModelPath + '/final_ckpt.pth', outputSize)
    surrogate_model.eval()
    result1, result2 = calculate_distance(model, surrogate_model, trainloader)
    matchRate1 = len(result2) / len(trainloader)
    sim = cal_similarity(result1, result2)

    # import matplotlib.pyplot as plt
    # x = range(len(result1))
    # plt.plot(x, result1,label='Source Model', color='#87CEFA', linestyle='-')
    # plt.plot(x, result2,label='Surrogate Model', color='#FFA500', linestyle='--')
    # plt.xlabel('Adverasary Images ID',fontsize=18)
    # plt.ylabel('Distance', fontsize=18)
    # plt.title('dicision boundary distance between different model')
    # plt.legend(fontsize = 11)
    # plt.show()

    return sim, matchRate1

if __name__ == '__main__':
    # SMP = 'train(resnet18,Flower102)-'
    # SusMP = ['train(resnet18,Flower102)-advTrain-']
    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-distill()-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.2)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-steal(mbnetv2)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-steal(resnet18)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-distill()-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(mbnetv2)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-']
    #
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
        # print([main('Flower102', item, SMP, 102), item, SMP])
    #
    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,1)-distill()-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.2)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.5)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(mbnetv2)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(resnet18)-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-distill()-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(mbnetv2)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-']
    #
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    #
    # SMP = 'train(mbnetv2,Flower102)-'
    # SusMP = ['train(mbnetv2,Flower102)-distill()-',
    #          'train(mbnetv2,Flower102)-prune(0.2)-',
    #          'train(mbnetv2,Flower102)-prune(0.5)-',
    #          'train(mbnetv2,Flower102)-steal(mbnetv2)-',
    #          'train(mbnetv2,Flower102)-steal(resnet18)-',
    #          'train(resnet18,Flower102)-',
    #          'train(resnet18,Flower102)-distill()-',
    #          'train(resnet18,Flower102)-prune(0.2)-',
    #          'train(resnet18,Flower102)-prune(0.5)-',
    #          'train(resnet18,Flower102)-steal(mbnetv2)-',
    #          'train(resnet18,Flower102)-steal(resnet18)-']
    #
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-'
    item = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-ReTrainLL-'
    print([main('SDog120', SMP, item, 120), SMP, item])

    # SMP = 'train(mbnetv2,Flower102)-'
    # SusMP = ['train(mbnetv2,Flower102)-advTrain-',
    #          'train(mbnetv2,Flower102)-FineTuneLL-',
    #          'train(mbnetv2,Flower102)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])

    # SMP = 'train(resnet18,Flower102)-'
    # SusMP = ['train(resnet18,Flower102)-advTrain-',
    #          'train(resnet18,Flower102)-FineTuneLL-',
    #          'train(resnet18,Flower102)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    
    # SMP = 'train(mbnetv2,SDog120)-'
    # SusMP = ['train(mbnetv2,SDog120)-advTrain-',
    #          'train(mbnetv2,SDog120)-FineTuneLL-',
    #          'train(mbnetv2,SDog120)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])
    
    # SMP = 'train(resnet18,SDog120)-'
    # SusMP = ['train(resnet18,SDog120)-advTrain-',
    #          'train(resnet18,SDog120)-FineTuneLL-',
    #          'train(resnet18,SDog120)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])

    
    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    
    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,1)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(Flower102,1)-advTrain-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,1)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    
    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])
    
    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])

    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,1)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('Flower102', SMP, item, 102), SMP, item])

    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-advTrain-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])
    
    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-advTrain-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])

    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-advTrain-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(SDog120,1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])

    # SMP = 'pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,0.1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])

    # SMP = 'pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])
    
    # SMP = 'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(SDog120,1)-advTrain-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-FineTuneLL-',
    #          'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-ReTrainLL-']
    # for item in SusMP:
    #     print([main('SDog120', SMP, item, 120), SMP, item])
    
        
    
    


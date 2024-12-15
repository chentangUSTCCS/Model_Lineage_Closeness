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


def fast_gradient_signed(x, y, model, eps, iter=10):
    device = 'cuda'
    criterion = nn.CrossEntropyLoss().to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y.long())
    model.zero_grad()
    loss.backward()
    sign = torch.sign(x.grad)
    while iter:
        y_pred = model(x + eps * sign)
        _, pred_labels = torch.max(y_pred, 1)
        if not torch.eq(pred_labels, y).item():
            break
        x = x + eps * sign
        iter -= 1
    if iter == 0: return (False,0)
    else:
        return (True,x.detach_())


def gen_adversaries(model, l, dataset, eps):
    # 可以先找出适合成为对抗样本的样本，再精修到决策边界附近
    device = 'cuda'
    advs = []
    count = 0
    for x, y in dataset:
        # generate adversaries
        print('trigger sets setting...   [{:0>3d}/{:0>3d}]'.format(count+1, l), end='\r')
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        flag, x_advs = fast_gradient_signed(x, y, model, eps)
        if flag: 
            advs.append((x_advs,y))
            count += 1
        if len(advs) == l: break
    print('\n')

    return advs


def calculate_distance(model, sur_model, adv):
    result1, result2 = [], []  # adv, distance, label, label'
    device = 'cuda'
    model.eval()
    sur_model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    # advloader = DataLoader(adv, batch_size=1, shuffle=False)
    count = 0
    for (img, label) in adv:
        img, label = img.to(device), label.to(device)
        img.requires_grad = True
        model.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, label.long())
        loss.backward()
        sign = torch.sign(img.grad)
        # print('distance calculating...   [{:0>3d}/{:0>3d}]'.format(len(result), 50), end='\r')

        # model-calculate
        dis1 = [0.001,0.0001,0.00001]
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
        result1.append([round(d1, 5), label.item(), pred_labels1.item()])
        # sur_model-calculate
        dis = [0.100, 0.010, 0.001,0.0001,0.00001]
        d = 0
        x = img.detach()
        for distance in dis:
            while 1:
                y_pred = sur_model(x + sign * distance)
                _, pred_labels = torch.max(y_pred, 1)
                if not torch.eq(pred_labels, label).item():
                    break

                x = (x + sign * distance).detach_()
                d += distance
        # d2 = 0.001
        # x = (img + sign * 0.001).detach_()
        # while 1:
        #     y_pred = sur_model(x)
        #     _, pred_labels = torch.max(y_pred, 1)
        #     if not torch.eq(pred_labels, label).item():
        #         break
        #
        #     x = (x + sign * 0.001).detach_()
        #     d2 += 0.001
        result2.append([round(d, 5), label.item(), pred_labels.item()])
    print(f'|-----bias by dicision boundary is: {count}')
    return result1, result2

def cal_dis_between_samples(x,y):
    pass
    
def cal_similarity(result1 , result2, distance = 'cosine'):
    # 考虑匹配率
    result = [[], []]
    count = 0
    for i in range(len(result1)):
        # d = abs(result1[i][0] - result2[i][0])
        label = (result1[i][1] == result2[i][1])
        label1 = (result1[i][2] == result2[i][2])
        if label == True and label1 == True:
            count += 1
        result[0].append(result1[i][0])
        result[1].append(result2[i][0])
    matchingRate = count/len(result1)
    if distance == 'cosine':
        similarity = torch.cosine_similarity(torch.tensor(result[0]), torch.tensor(result[1]), dim=0)
        return similarity, matchingRate
    elif distance == 'KL':
        similarity = F.kl_div(F.log_softmax(torch.tensor(result[0]),dim=-1), F.softmax(torch.tensor(result[1]), dim=-1), reduction='sum')
        return similarity, matchingRate
    elif distance == 'both':
        return torch.cosine_similarity(torch.tensor(result[0]), torch.tensor(result[1]), dim=0), F.kl_div(F.log_softmax(torch.tensor(result[0]),dim=-1), F.softmax(torch.tensor(result[1]), dim=-1), reduction='sum'), matchingRate

def main(dataset, SourceModelPath, SuspiciousModelPath, triggernum, outputSize):
    device = 'cuda'
    trainloader = get_dataloader(dataset_id = dataset, batch_size = 1)
    fix_path = './models/'
    model = get_model(fix_path + SourceModelPath+'/final_ckpt.pth', outputSize)
    model.to(device)
    # print(cs)

    trigger_num = triggernum
    model.eval()
    advs = gen_adversaries(model, trigger_num, trainloader, 0.02)
    print(len(advs))
    # assert (len(advs) == trigger_num)
    print('------trigger sets setting done!------')
    surrogate_model = get_model(
        fix_path + SuspiciousModelPath + '/final_ckpt.pth', outputSize)
    # homo_model = get_model(
    #     fix_path + 'models\pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-' + '/final_ckpt.pth', 102)
    result1, result2 = calculate_distance(model, surrogate_model, advs)
    # result3 = calculate_distance(homo_model, advs)

    sim11,sim12, matchRate1 = cal_similarity(result1, result2, distance = 'both')
    # sim21,sim22, matchRate2 = cal_similarity(result1, result3, distance = 'both')
    # print(sim11.item(),sim12.item(), matchRate1)
    return sim11.item(),sim12.item(), matchRate1
    #
    # import matplotlib.pyplot as plt
    # x = range(len(result[0]))
    # plt.plot(x, result[0],label='Source Model', color='#87CEFA', linestyle='-')
    # plt.plot(x, result[1],label='Surrogate Model', color='#FFA500', linestyle='--')
    # plt.xlabel('Adverasary Images ID',fontsize=18)
    # plt.ylabel('Distance', fontsize=18)
    # plt.title('dicision boundary distance between different model')
    # plt.legend(fontsize = 11)
    # plt.show()
    # plt.savefig('watermark_detect_rate.eps', dpi=600)
    

import argparse
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', help='type of dataset', type=str, default='Flower102')
    # parser.add_argument('--OutputSize', help='size of output', type=int, default=102)
    # parser.add_argument('--SMP', help='Source model path', type=str, default='')
    # parser.add_argument('--SusMP', help='Suspicious model path', type=str, default='')
    # parser.add_argument('--triggernum', help='number of trigger set', type=int, default=100)
    # args = parser.parse_args()
    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-distill()-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.5)-',
    # 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.8)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-steal(mbnetv2)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-steal(resnet18)-','pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-']
    
    # for item in SusMP:
    #     print(main('SDog120', SMP, item, 100, 120), SMP, item)
    answer = []

    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-', 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-', 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.5)-',
    # 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.8)-', 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-steal(mbnetv2)-', 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-','pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-']
    # for item in SusMP:
    #     answer.append([main('Flower102', SMP, item, 500, 102), SMP, item])
    # for ans in answer:
    #     print(ans)

    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-'
    # SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-distill()-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.5)-',
    # 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.8)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-steal(mbnetv2)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-steal(resnet18)-', 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.5)-']
    # for item in SusMP:
    #     if 'steal(mb' in item:
    #         answer.append([main('SDog120', SMP, item, 500, 120), SMP, item])
    # for ans in answer:
    #     print(ans)

    SMP = 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-'
    # SusMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-','pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-distill()-','pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-',
    # 'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(resnet18)-','pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(mbnetv2)-']
    # for item in SusMP:
    #     answer.append([main('Flower102', SMP, item, 2040, 102), SMP, item])
    #     break
    # for ans in answer:
    #     print(ans)

    # SMP = 'pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-'
    SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-','pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-distill()-',
    'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(mbnetv2)-','pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-']
    for item in SusMP:
        answer.append([main('Flower102', SMP, item, 2040, 102), SMP, item])
        break
    for ans in answer:
        print(ans)

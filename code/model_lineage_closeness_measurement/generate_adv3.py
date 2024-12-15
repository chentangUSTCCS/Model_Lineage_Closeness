import os
import torch, copy, argparse
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from func import get_model, get_dataloader
from generate_adv2 import calculate_distance, cal_similarity

def evalu(model, testloader):
    model.eval()
    device = 'cuda'
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

def isMatch(model, sur_model, data):
    device = 'cuda'
    model.eval()
    sur_model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    img, label = data[0].to(device), data[1].to(device)
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
            if d1 > 10: return False
    # sur_model-calculate
    # 先检查输出label是否一样，是否为有效样本
    y_pr = sur_model(img)
    _, pr = torch.max(y_pr, 1)
    if not torch.eq(pr, label).item():
        return False

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
            if d > 10: return False
    if pred_labels1 == pred_labels2:
        return True
    else: return False

def cal_similarity(result1 , result2):
    return (sum(result1)-sum(result2))/(len(result1)+0.0001)

def testset_generate(model ,blood_models,no_blood_models, dataset):
    testset = []
    for image, label in tqdm.tqdm(dataset):
        flag = True
        for m in blood_models:
            if not isMatch(model, m, (image, label)):
                flag = False
                break
        if not flag: continue
        for m in no_blood_models:
            if isMatch(model, m, (image,label)):
                flag = False
                break
        if flag: testset.append((image, label))
    return testset

def main(dataset, SourceModelPath, BloodModelPath, NoBloodModelPath, SusModelPath, outputSize):
    device = 'cuda'
    trainloader = get_dataloader(dataset_id = dataset, batch_size = 1)
    test_dataset = get_dataloader(dataset_id=dataset, split='test', batch_size=50, shuffle=False)
    fix_path = '../models/'
    model = get_model(fix_path + SourceModelPath+'/final_ckpt.pth', outputSize)
    model.to(device)
    model.eval()
    bloodmodels = []
    for mp in BloodModelPath:
        bloodmodel = get_model(fix_path + mp +'/final_ckpt.pth', outputSize)
        bloodmodel.to(device)
        bloodmodel.eval()
        bloodmodels.append(bloodmodel)
    nobloodmodels = []
    for mp in NoBloodModelPath:
        nobloodmodel = get_model(fix_path + mp + '/final_ckpt.pth', outputSize)
        nobloodmodel.to(device)
        nobloodmodel.eval()
        nobloodmodels.append(nobloodmodel)

    import time
    t = time.time()
    testdata = testset_generate(model, bloodmodels, nobloodmodels, trainloader)
    print(f'total dataset is {len(testdata)}, about {round(len(testdata)/len(trainloader)*100,2)}%, total time is {time.time()-t}.')

    for item in SusMP:
        sus_model = get_model(fix_path + item + '/final_ckpt.pth', outputSize)
        sus_model.to(device)
        sus_model.eval()
        result1, result2 = calculate_distance(model, sus_model, testdata)
        matchRate = len(result2) / len(testdata)
        sim = cal_similarity(result1, result2)
        print(matchRate, sim, item, SMP)


if __name__ == '__main__':
    # # Souce model position
    # SMP = 'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-'
    # # Blood model position
    # BMP = ['pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-prune(0.2)-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-FineTuneLL-',
    #          'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-ReTrainLL-']
    # # No Blood model position
    # NBMP = ['pretrain(resnet18,ImageNet)-transfer(Flower102,1)-',
    #         'pretrain(resnet18,ImageNet)-transfer(Flower102,1)-distill()-']
    # # Suscipious model position
    # SusMP= ['pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-advTrain-',
    #         'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-distill()-',
    #         'pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-']
    # main('Flower102', SMP, BMP, NBMP, SusMP, 102)

    
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    # Souce model position
    SMP = 'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-'
    # Blood model position
    BMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-',
           'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-FineTuneLL-',
           'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-ReTrainLL-']
    # No Blood model position
    NBMP = ['pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-',
            'pretrain(resnet18,ImageNet)-transfer(SDog120,0.5)-distill()-']
    # Suscipious model position
    SusMP = ['pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-FineTuneLL-',
             'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-ReTrainLL-',
             'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.2)-',
             'pretrain(mbnetv2,ImageNet)-transfer(SDog120,0.1)-prune(0.5)-',
             'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-prune(0.5)-',
             'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-steal(mbnetv2)-',
             'pretrain(resnet18,ImageNet)-transfer(SDog120,1)-steal(resnet18)-']
    main('SDog120', SMP, BMP, NBMP, SusMP, 120)

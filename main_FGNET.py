import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from FGNET_data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from residue_loss_adaptive_K import MeanResidueLossAdaptive  # for K value searching , replace this with mean residue loss
import cv2

START_AGE = 0
END_AGE = 69
VALIDATION_RATE= 0

def VGG16(num_classes):

    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def ResNet50(num_classes):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    return model

def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory):

    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_variance_loss = 0.
    running_softmax_loss = 0.
    interval = 1
    adaptive_Ks = []
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        output = model(images)
        mean_loss, variance_loss, adaptive_K = criterion1(output, labels)
        adaptive_Ks.append(adaptive_K)
        softmax_loss = criterion2(output, labels)
        loss = mean_loss + variance_loss + softmax_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_mean_loss += mean_loss.data
        running_variance_loss += variance_loss.data
        running_softmax_loss += softmax_loss.data
        if (i + 1) % interval == 0:
            log_info = f'[{epoch}, {i:5d}] mean_loss: {running_mean_loss / interval:.3f}, residual_loss: {running_variance_loss / interval:.3f}, softmax_loss: {running_softmax_loss / interval:.3f}, loss: {running_loss / interval:.3f}, K:{int(adaptive_K)}'
            print(log_info)
            with open(os.path.join(result_directory, 'log'), 'a') as f:
                f.write(log_info + '\n')
            running_loss = 0.
            running_mean_loss = 0.
            running_variance_loss = 0.
            running_softmax_loss = 0.
    return adaptive_Ks

def test(test_loader, model):
    model.cuda()
    model.eval()
    mae = 0.
    num_correct = [0] * 11
    num_wrong = [0] * 11
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            output = model(image)
            m = nn.Softmax(dim=1)
            output = m(output)
            for j in range(11):
                v, ind = torch.max(output[0], 0)
                if abs(ind.data - label) <= j:
                    num_correct[j] += 1
                else:
                    num_wrong[j] += 1
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    cs_list = []
    for i in range(11):
        cs_list.append(num_correct[i]/(num_correct[i] + num_wrong[i]))
    return mae / len(test_loader), cs_list

def predict(model, image):

    model.eval()
    with torch.no_grad():
        image = image.astype(np.float32) / 255.
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        output = model(img[None])
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred

def get_image_list(image_directory, leave_sub, validation_rate):
    
    train_val_list = []
    test_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)
        subject = int(fn[:3])
        if subject == leave_sub:
            test_list.append(filepath)
        else:
            train_val_list.append(filepath)
    num = len(train_val_list)
    index_val = np.random.choice(num, int(num * validation_rate), replace=False)
    train_list = []
    val_list = []
    for i, fp in enumerate(train_val_list):
        if i in index_val:
            val_list.append(fp)
        else:
            train_list.append(fp)

    return train_list, val_list, test_list


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-i', '--image_directory', type=str, default='FGNET/images')
    parser.add_argument('-ls', '--leave_subject', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-K', '--K', type=int, default=6)
    parser.add_argument('-L1', '--LAMBDA1', type=float, default=0.2)
    parser.add_argument('-L2', '--LAMBDA2', type=float, default=0.25)
    parser.add_argument('--net', type=str, default='VGG', help='VGG/ResNet')
    parser.add_argument('-s', '--seed', type=int, default=2019)
    parser.add_argument('-m','--milestones', nargs='+', required=True)
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    parser.add_argument('-pi', '--pred_image', type=str, default=None)
    parser.add_argument('-pm', '--pred_model', type=str, default=None)
    return parser.parse_args()


def main():
    #----deterministic----
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    K = args.K
    LAMBDA_1 = args.LAMBDA1
    LAMBDA_2 = args.LAMBDA2 # Tuned
    milestones = args.milestones
    milestones = [int(i) for i in milestones]
    if args.epoch > 0:
        batch_size = args.batch_size
        if args.result_directory is not None:
            if not os.path.exists(args.result_directory):
                os.mkdir(args.result_directory)

        train_filepath_list, val_filepath_list, test_filepath_list\
            = get_image_list(args.image_directory, args.leave_subject, VALIDATION_RATE)
        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply(
                # [torchvision.transforms.RandomAffine(degrees=10, shear=16),
                 [torchvision.transforms.RandomHorizontalFlip(p=1.0),
                ], p=0.5),

            # not in the original paper
            torchvision.transforms.Resize((256, 256)),
            # torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.RandomCrop((227, 227)), # VGG16

            torchvision.transforms.ToTensor()
        ])
        train_gen = FaceDataset(train_filepath_list, transforms_train, seed)
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((227, 227)),# 224, 224 for resnet

            torchvision.transforms.ToTensor()
        ])
        val_gen = FaceDataset(val_filepath_list, transforms)
        val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

        test_gen = FaceDataset(test_filepath_list, transforms)
        test_loader = DataLoader(test_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
        
        if args.net == 'VGG':
            model = VGG16(END_AGE - START_AGE + 1)
        elif args.net == 'ResNet':
            model = ResNet50(END_AGE - START_AGE + 1)


        model.cuda()
        try:
            for param in model.classifier.parameters(): # VGG16
                param.requires_grad = True
            for param in model.features.parameters(): # VGG16
                param.requires_grad = True
        except:
            for param in model.parameters(): # Resnet
                param.requires_grad = True

        optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
        criterion1 = MeanResidueLossAdaptive(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE, K).cuda()

        criterion2 = torch.nn.CrossEntropyLoss().cuda()

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1) # ResNet 50

        best_val_mae = np.inf
        best_val_loss = np.inf
        best_mae_epoch = -1
        best_loss_epoch = -1
        mae_tests = []
        with open(os.path.join(args.result_directory, 'log'), 'a') as f:
            f.write(f'Starting learning rate: {args.learning_rate}\n')
            f.write(f'Milestones: {milestones}\n')
            f.write(f'K={K}\n')
            f.write(f'LAMBDA1={LAMBDA_1}\n')
            f.write(f'LAMBDA2={LAMBDA_2}\n')
            f.write(f'seed={seed}\n')
        K_values = []
        for epoch in range(args.epoch):
            adaptive_Ks = train(train_loader, model, criterion1, criterion2, optimizer, epoch, args.result_directory)
            K_values.append(adaptive_Ks)
            # mean_loss, variance_loss, softmax_loss, loss_val, mae = 0,0,0,0,0 #evaluate(val_loader, model, criterion1, criterion2)
            mae_test, cs_list = test(test_loader, model)
            test_mae_info = f'epoch: {int(epoch)}, test_mae: {mae_test[0][0]:.3f}'
            print(test_mae_info)
            mae_tests.append(mae_test)
            min_mae_tests = min(mae_tests)[0][0]
            best_test_msg = f'best test mae {min_mae_tests:.6f} at epoch {mae_tests.index(min_mae_tests)}'  # for monitoring
            print(best_test_msg)
            for idx, i in enumerate(cs_list):
                print(f'theta = {idx}, acc = {i:.6f}')
            with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                f.write(test_mae_info + '\n')
                f.write(best_test_msg + '\n')
                f.write('CS accuracy:\n')
                for idx, i in enumerate(cs_list):
                    f.write(f'theta = {idx}, acc = {i:.6f}\n')

            scheduler.step(epoch) # after PyTorch 1.1.0, scheduler.step should be after optimizer.step, inside train
        K_sum = 0

        torch.save(model.state_dict(), os.path.join(args.result_directory, "last_checkpoint"))  
    if args.pred_image and args.pred_model:
        model = VGG16(END_AGE - START_AGE + 1)
        model.cuda()
        img = cv2.imread(args.pred_image)
        resized_img = cv2.resize(img, (227, 227)) # 227 for vgg

        model.load_state_dict(torch.load(args.pred_model))
        pred = predict(model, resized_img)
        print('Age: ' + str(int(pred)))
        cv2.putText(img, 'Age: ' + str(int(pred)), (int(img.shape[1]*0.1), int(img.shape[0]*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        name, ext = os.path.splitext(args.pred_image)
        cv2.imwrite(name + '_result.jpg', img)
        
if __name__ == "__main__":
    main()

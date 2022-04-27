import os
import time
import json
import sys

sys.path.insert(0, '../')
import argparse
import torch
import torchvision
import random
import numpy as np
from CLAP_data import CLAPFaceDataset
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet50
from residue_loss_adaptive_K import \
    MeanResidueLossAdaptive  # for K value searching , replace this with mean residue loss
sys.path.append('/home/ziyuan/age_estimation/AgeEstimation-K')
from AgeEstimation.mean_variance_loss import MeanVarianceLoss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pytz
import logging


START_AGE = 0
END_AGE = 100
VALIDATION_RATE = 0


def checkpoint_save(model, is_best,name):
    if is_best:
        torch.save(model.state_dict(), os.path.join(name, 'checkpoint.pth'))

        print('Saved checkpoint:', os.path.join(name, 'checkpoint.pth'))


basedir = ''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=64)  # 32 * 2
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)  # from 0.0001
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-K', '--K', type=int, default=6)
    parser.add_argument('-L1', '--LAMBDA1', type=float, default=0.2)
    parser.add_argument('-L2', '--LAMBDA2', type=float, default=0.25)
    parser.add_argument('-s', '--seed', type=int, default=2019)
    parser.add_argument('-m', '--milestones', nargs='+'
                        ,default=50)
    parser.add_argument('-rd', '--result-directory', type=str, default=None)
    parser.add_argument('-pi', '--pred-image', type=str, default=None)
    parser.add_argument('-pm', '--pred-model', type=str, default=None)
    
    parser.add_argument('--net', type=str, default='VGG', help='VGG/ResNet')
    parser.add_argument('--loss', type=str, default='mrloss', help = 'mean_softmax/residual_softmax/softmax/mrloss/mvloss')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument('--Adam', action='store_true')
    


    parser.add_argument('--gpu', type=int, default=2, help='GPU to use')
    return parser.parse_args()

def get_cur_time():
    return datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), '%Y-%m-%d_%H-%M-%S')



def VGG16(num_classes):
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def ResNet50(num_classes):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    return model


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory, loss_mode):
    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_residual_loss = 0.
    running_softmax_loss = 0.
    interval = 1
    adaptive_Ks = []
    adaptive_K = 0
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda(non_blocking=True)
        labels = sample['label']  # .cuda()
        for idx, label in enumerate(labels):
            labels[idx] = np.around(labels[idx].item())

        labels = labels.cuda(non_blocking=True)
        labels = labels.long()
        output = model(images)

        if loss_mode == 'mvloss':
            mean_loss, residual_loss = criterion1(output, labels)      
        else:
            mean_loss, residual_loss, adaptive_K = criterion1(output, labels)
            adaptive_Ks.append(adaptive_K)
        softmax_loss = criterion2(output, labels)
        loss = mean_loss + residual_loss + softmax_loss        
        optimizer.zero_grad()
        if loss_mode == 'mean_softmax':
            (mean_loss + softmax_loss).backward()
        elif loss_mode == 'residual_softmax':
            (residual_loss + softmax_loss).backward()
        elif loss_mode == 'softmax':
            softmax_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        running_loss = loss.data
        running_mean_loss = mean_loss.data
        running_residual_loss = residual_loss.data
        running_softmax_loss = softmax_loss.data
        
        logging.info(f'[{epoch}, {i:5d}] mean_loss: {running_mean_loss:.3f}, residual_loss: {running_residual_loss:.3f}, softmax_loss: {running_softmax_loss:3f}, loss: {running_loss:.3f}, K:{int(adaptive_K)}')

    # tb.flush()
    return adaptive_Ks, running_mean_loss, running_residual_loss, running_softmax_loss, running_loss


# change labels
def evaluate(val_loader, model, criterion1, criterion2, loss_mode):
    # model.cuda()
    model.eval()
    loss_val = 0.
    mean_loss_val = 0.
    residual_loss_val = 0.
    softmax_loss_val = 0.
    eps = 0.
    k=0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            labels = sample['label']  # .cuda()
            labels = np.around(labels)
            labels = labels.cuda()
            labels = labels.long()
            output = model(image)
            var_s = sample['var']
            if loss_mode == 'mvloss':
                mean_loss, residual_loss = criterion1(output, labels)
            else:
                mean_loss, residual_loss, k = criterion1(output, labels)
            # print(f'k {k:.3f}',)
            softmax_loss = criterion2(output, labels)
            loss = mean_loss + residual_loss + softmax_loss

            loss_val += loss.data
            mean_loss_val += mean_loss.data
            residual_loss_val += residual_loss.data
            softmax_loss_val += softmax_loss.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)

            eps += np.e ** (-(pred - np.around(sample['label'].cpu().data.numpy().item())) ** 2 / (
                        2 * (var_s + 0.0001) ** 2).item())  # in case var_s == 0

    eps /= len(val_loader)
    eps = 1 - eps

    return mean_loss_val / len(val_loader), \
           residual_loss_val / len(val_loader), \
           softmax_loss_val / len(val_loader), \
           loss_val / len(val_loader), \
           eps


def test(test_loader, model):
    model.eval()
    eps = 0.
    num_correct = [0] * 11
    num_wrong = [0] * 11
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['image'].cuda()
            labels = np.around(sample['label'].item())  # .cuda()
            var_s = sample['var']
            output = model(image)
            m = nn.Softmax(dim=1)
            output = m(output)
            for j in range(11):
                v, ind = torch.max(output[0], 0)
                if abs(ind.data - labels) <= j:
                    num_correct[j] += 1
                else:
                    num_wrong[j] += 1
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            # pred = mean
            eps += np.e ** (-(pred - np.around(sample['label'].cpu().data.numpy().item())) ** 2 / (
                        2 * (var_s + 0.0001) ** 2).item())  # in case var_s == 0

    eps /= len(test_loader)
    eps = 1 - eps
    cs_list = []
    for i in range(11):
        cs = num_correct[i] / (num_correct[i] + num_wrong[i])
        cs_list.append(cs)

    return eps, cs_list


def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.astype(np.float32) / 255.
        image = np.transpose(image, (2, 0, 1))
        img = torch.from_numpy(image).cuda()
        output = model(img[None])
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred



def pred_labels():
    model = VGG16(END_AGE - START_AGE + 1)
    checkpoint = torch.load('../result/epoch_43_checkpoint.pth')
    model.load_state_dict(checkpoint)
    test_filepath_list = '../CLAP/test_cv'
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((227, 227)),  # 224, 224 for alex, resnet34
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.4816, 0.4199, 0.3884], [0.2568, 0.2408, 0.2323]),
    ])
    test_gen = CLAPFaceDataset(test_filepath_list, transforms)
    print (test_gen)

    
    
def main():
    # ----deterministic----
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_args()
    gpu = args.gpu
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    K = args.K
    LAMBDA_1 = args.LAMBDA1
    LAMBDA_2 = args.LAMBDA2  # Tuned
    milestones = args.milestones
    milestones = [int(i) for i in milestones]
    
    logdir = os.path.join(basedir, 'logs', str(args.net), get_cur_time())
    print(logdir)
    savedir = os.path.join(basedir, 'checkpoints', str(args.net), get_cur_time())
    print(savedir)
    shotdir = os.path.join(basedir, 'snapshot', str(args.net), get_cur_time())
    print(shotdir)
    
    os.makedirs(logdir, exist_ok=False)
    os.makedirs(savedir, exist_ok=False)
    os.makedirs(shotdir, exist_ok=False)
    
    tb = SummaryWriter(logdir)
    
    logging.basicConfig(filename=shotdir+"/"+"snapshot.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    logging.info(str(args))
    
    
    #time
    start_time = time.time()
    print('----------------------------------Training Started-----------------------------------------------------')
    # setup dist


    torch.cuda.set_device(gpu)

    if args.epoch > 0:
        batch_size = args.batch_size
        if args.result_directory is not None:
            if not os.path.exists(args.result_directory):
                os.mkdir(args.result_directory)

        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # imagenet mean and std
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomAffine(degrees=10, shear=16),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomGrayscale(p=0.1),
            # not in the original paper
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((227, 227)),  # VGG16
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.4816, 0.4199, 0.3884], [0.2568, 0.2408, 0.2323]),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
        # hard code
        train_filepath_list = '../CLAP/train_cv'
        val_filepath_list = '../CLAP/valid_cv'
        test_filepath_list = '../CLAP/test_cv'

        train_gen = CLAPFaceDataset(train_filepath_list, transforms_train, seed)

        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((227, 227)),  # 224, 224 for alex, resnet34
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.4816, 0.4199, 0.3884], [0.2568, 0.2408, 0.2323]),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
        val_gen = CLAPFaceDataset(val_filepath_list, transforms)

        val_loader = DataLoader(val_gen, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)  # 3 to speed up

        test_gen = CLAPFaceDataset(test_filepath_list, transforms)

        test_loader = DataLoader(test_gen, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)  # 3 to speed up
        if args.net == 'VGG':
            model = VGG16(END_AGE - START_AGE + 1)
        elif args.net == 'ResNet':
            model = ResNet50(END_AGE - START_AGE + 1)

        model.to(gpu)

        
        if args.SGD:
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9, weight_decay=1e-4)
        elif args.Adam:
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        if args.loss == 'mvloss':
            criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda(gpu)
        else: 
            criterion1 = MeanResidueLossAdaptive(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE, K).cuda(gpu)
        

        criterion2 = torch.nn.CrossEntropyLoss().cuda(gpu)

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1) # 10 for AlexBN; for VGG16 step size should be 15;
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # ResNet 34

        best_val_eps = np.inf
        best_val_loss = np.inf
        best_eps_epoch = -1
        best_loss_epoch = -1
        eps_tests = []

        K_values = []
        for epoch in range(args.epoch):

            adaptive_Ks, running_mean_loss, running_residual_loss, running_softmax_loss, running_loss = train(train_loader, model, criterion1, criterion2, optimizer, epoch, args.result_directory, args.loss)
          
            tb.add_scalar('info/train_mean_loss', running_mean_loss, epoch)
            tb.add_scalar('info/train_residual_loss', running_residual_loss, epoch)
            tb.add_scalar('info/train_softmax_loss', running_softmax_loss, epoch)
            tb.add_scalar('info/train_total_loss', running_loss, epoch)

            K_values.append(adaptive_Ks)
            mean_loss, residual_loss, softmax_loss, loss_val, eps = evaluate(val_loader, model, criterion1, criterion2, args.loss)
                     
            
            
            
            tb.add_scalar('info/val_mean_loss', mean_loss, epoch)
            tb.add_scalar('info/val_mean_loss', residual_loss, epoch)
            tb.add_scalar('info/val_softmax_loss', softmax_loss, epoch)
            tb.add_scalar('info/val_total_loss', loss_val, epoch)
            

            tb.add_scalar('info/val_eps',eps, epoch)

            eps_test, cs_list = test(test_loader, model)
            
            
            tb.add_scalar('info/test_eps',eps_test, epoch)
            logging.info(f'epoch: {int(epoch)}, val_eps: {eps[0][0]:.3f}')
            logging.info(f'epoch: {int(epoch)}, test_eps: {eps_test[0][0]:.3f}')
            eps_tests.append(eps_test)
            min_eps_tests = float(min(eps_tests))
            logging.info(f'best test eps {min_eps_tests:.3f} at epoch {eps_tests.index(min_eps_tests)}')    # for monitoring
            
            if eps < best_val_eps:
                
                checkpoint_save(model, eps < best_val_eps, savedir)
                logging.info("save model to {}".format(savedir))
                best_val_eps = min(eps, best_val_eps)
             
            scheduler.step(epoch)  # after PyTorch 1.1.0, scheduler.step should be after optimizer.step, inside train
    end_time = time.time()
    print('----------------------------------Training Ended-----------------------------------------------------')
    print(f'--------------------------------Time elapsed for {args.epoch} epochs is {end_time-start_time}')

    
    if args.Adam:
        tb.add_hparams({'log_dir':logdir, 'arch/model':args.net, 'loss_func': args.loss,'optimizer': 'Adam', 'lr': args.learning_rate, 'batch_size': args.batch_size, 'num_epoch':args.epoch, 'lambda_1':args.LAMBDA1, 'lambda_2':args.LAMBDA2}, {'val_eps':best_val_eps, 'test_eps': min_eps_tests})
    elif args.SGD:
        tb.add_hparams({'log_dir':logdir, 'arch/model':args.net, 'loss_func': args.loss,'optimizer': 'SGD', 'lr': args.learning_rate, 'batch_size': args.batch_size, 'num_epoch':args.epoch, 'lambda_1':args.LAMBDA1, 'lambda_2':args.LAMBDA2}, {'val_eps':best_val_eps, 'test_eps': min_eps_tests})
    tb.close()

if __name__ == "__main__":

    main()
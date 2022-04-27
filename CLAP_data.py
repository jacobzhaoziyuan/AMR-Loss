import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import csv


class CLAPFaceDataset(Dataset):

    def __init__(self, filepath, transform=None, seed=2019):
        # fix dataset and loading
        np.random.seed(seed) # original code does not have this

        # hardcode gt_path
        self.train_gt_file = '../CLAP/gt_avg_train.csv'
        self.validation_gt_file = '../CLAP/gt_avg_valid.csv'
        self.test_gt_file = '../CLAP/gt_avg_test.csv'
        self.gt_file = ''
        # print(os.getcwd())
        # self.ignore
        self.ignore = '../CLAP/ignore_list.txt'

        print(self.ignore)
        self.images = []
        self.labels = []
        self.vars = []
        self.label_dict = {}
        with open(self.ignore) as f:
            self.ignore_images = f.readlines()
        self.ignore_images = [i[:-1] for i in self.ignore_images]

        if 'train' in filepath:
            self.gt_file = self.train_gt_file
        elif 'valid' in filepath:
            self.gt_file = self.validation_gt_file
        else:
            self.gt_file = self.test_gt_file
        with open(self.gt_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                row = row[0].split(',')
                try:
                    self.label_dict[row[0]] = {'mean':float(row[2]), 'var':float(row[3])}
                except:
                    # this is the heading line
                    continue   

        for _, _, files in os.walk(filepath):
            for f in files:
                img = np.array(Image.open(os.path.join(filepath, f)).convert('RGB'))
                if img.size < 3072: # 3 * 32 * 32
                    print(f, 'is too small')
                    continue

                # if False:
                if f in self.ignore_images:
                    print(f)
                    continue
                else:
                    self.labels.append(self.label_dict[f]['mean'])
                    self.vars.append(self.label_dict[f]['var'])
                    img = np.array(Image.open(os.path.join(filepath, f)).convert('RGB'))
                    self.images.append(img)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.vars = np.array(self.vars)
        self.transform = transform

    def __len__(self):

        return self.images.shape[0]

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        var = self.vars[index]
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': label, 'var': var}
        return sample       

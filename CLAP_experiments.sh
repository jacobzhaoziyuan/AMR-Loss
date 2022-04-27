#!/usr/bin/bash

batch_size=64
learning_rate=0.001
epoch=100
K=6
LAMBDA1=0.2
LAMBDA2=0.05
Net=ResNet
loss=mvloss
gpu=0


python main_CLAP.py --batch-size $batch_size --learning-rate $learning_rate --epoch $epoch --K $K --LAMBDA1 $LAMBDA1 --LAMBDA2 $LAMBDA2 --net $Net --loss mrloss --SGD --gpu $gpu -m 15 30 45 60 75 90
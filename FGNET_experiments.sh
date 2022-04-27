#!/usr/bin/bash

mkdir -p FGNET_experiments

batch_size=64
learning_rate=0.001
epoch=100
K=6
LAMBDA1=0.2
LAMBDA2=0.05
Net=ResNet

for i in {1..82}
do
   CUDA_VISIBLE_DEVICES=0 python3 main_FGNET.py --leave_subject $i -b $batch_size --learning_rate $learning_rate -K $K --LAMBDA1 $LAMBDA1 --LAMBDA2 $LAMBDA2 --net $Net --epoch $epoch --seed 42 -m 15 30 45 60 75 90 --result_directory FGNET_experiments/subj_$i
done
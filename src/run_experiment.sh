#!/bin/sh

mkdir -p logs

for nn in AlphaGoZero DeepSpeech2 FasterRCNN NCF_recommendation \
  NCF_recommendation_short Resnet50 Sentimental_seqCNN Sentimental_seqLSTM \
  Sentimental_seqLSTM_short Transformer Transformer_short
do
  for allreduce in ring multitree mxnettree
  do
    screen -dmS ${nn}_${allreduce}
    screen -S $nn -X stuff $'python simulate.py \
      --network SCALE-Sim/topologies/mlperf/'$nn'.csv \
      --run-name '$nn' \
      --booksim-config booksim2/runfiles/ctorus44multitree.cfg \
      --allreduce '$allreduce' \
      --only-allreduce &\n'
  done
done

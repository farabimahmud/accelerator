#!/bin/sh

for nn in AlphaGoZero DeepSpeech2 FasterRCNN NCF_recommendation \
  NCF_recommendation_short Resnet50 Sentimental_seqCNN Sentimental_seqLSTM \
  Sentimental_seqLSTM_short Transformer Transformer_short
do
  for allreduce in multitree ring mxnettree
  do
    python simulate.py \
      --network $SIMHOME/src/SCALE-Sim/topologies/mlperf/$nn.csv \
      --run-name ${nn} \
      --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
      --allreduce $allreduce \
      --message-buffer-size 32 \
      > $SIMHOME/results/logs/${nn}_${allreduce}_error.log 2>&1 &
  done
done

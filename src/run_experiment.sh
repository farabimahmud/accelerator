#!/bin/sh

mkdir -p logs

for nn in AlphaGoZero DeepSpeech2 FasterRCNN NCF_recommendation \
  NCF_recommendation_short Resnet50 Sentimental_seqCNN Sentimental_seqLSTM \
  Sentimental_seqLSTM_short Transformer Transformer_short
do
  screen -dmS $nn
  screen -S $nn -X stuff $'python simulate.py \
    --network SCALE-Sim/topologies/mlperf/'$nn'.csv \
    --run-name '$nn' \
    --booksim-config booksim2/runfiles/mesh44express.cfg \
    > logs/'$nn'.log &\n'
done

#!/bin/sh

outdir=$SIMHOME/results/logs_strict_schedule_nov_23

mkdir -p $outdir

# MLPerf
for nn in AlphaGoZero DeepSpeech2 FasterRCNN NCF_recommendation \
  NCF_recommendation_short Resnet50 Sentimental_seqCNN Sentimental_seqLSTM \
  Sentimental_seqLSTM_short Transformer Transformer_short
do
  for allreduce in multitree ring mxnettree
  do
    python $SIMHOME/src/simulate.py \
      --network $SIMHOME/src/SCALE-Sim/topologies/mlperf/$nn.csv \
      --run-name ${nn} \
      --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
      --allreduce $allreduce \
      --outdir $outdir \
      --message-buffer-size 32 \
      > $SIMHOME/results/logs_strict_schedule_nov_23/${nn}_${allreduce}_error.log 2>&1 &
  done
done

# CNN
for nn in alexnet Googlenet
do
  for allreduce in multitree ring mxnettree
  do
    python $SIMHOME/src/simulate.py \
      --network $SIMHOME/src/SCALE-Sim/topologies/conv_nets/$nn.csv \
      --run-name ${nn} \
      --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
      --allreduce $allreduce \
      --outdir $outdir \
      --message-buffer-size 32 \
      > $SIMHOME/results/logs_strict_schedule_nov_23/${nn}_${allreduce}_error.log 2>&1 &
  done
done

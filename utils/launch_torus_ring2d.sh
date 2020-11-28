#!/bin/sh

outdir=$SIMHOME/results/logs

mkdir -p $outdir

mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

# dtree
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=ring2d
  python $SIMHOME/src/simulate.py \
    --num-hmcs 16 \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus4x4multitree.cfg \
    --booksim-network torus \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    > $outdir/${nn}_${allreduce}_error.log 2>&1 &
done

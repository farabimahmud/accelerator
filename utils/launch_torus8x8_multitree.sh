#!/bin/sh

outdir=$SIMHOME/results/layerwise/dnn_torus8x8_logs

mkdir -p $outdir

mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

# multitree and msg
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=multitree

  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --num-hmcs 64 \
    --booksim-network torus \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus8x8multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --strict-schedule \
    --prioritize-schedule \
    --estimate-lockstep \
    > $outdir/${nn}_${allreduce}_beta_error.log 2>&1 &

  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --num-hmcs 64 \
    --booksim-network torus \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus8x8multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 0 \
    --sub-message-size 256 \
    --only-allreduce \
    --strict-schedule \
    --prioritize-schedule \
    --estimate-lockstep \
    > $outdir/${nn}_${allreduce}_delta_error.log 2>&1 &
done

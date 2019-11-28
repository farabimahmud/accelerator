#!/bin/sh

outdir=$SIMHOME/results/mesh_logs

mkdir -p $outdir

mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

# ring
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=ring
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/mesh44express.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_error.log 2>&1 &
done

# mxnettree-alpha
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=mxnettree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/mesh44express.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    > $outdir/${nn}_${allreduce}_alpha_error.log 2>&1 &
done

# mxnettree-beta
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=mxnettree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/cmesh44multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 2 \
    --radix 4 \
    --only-allreduce \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_beta_error.log 2>&1 &
done

# multitree-gamma
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=mxnettree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/cmesh44multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 2 \
    --radix 4 \
    --only-allreduce \
    --message-buffer-size 32 \
    --message-size 0 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_gamma_error.log 2>&1 &
done

# multitree-alpha
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=multitree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/mesh44express.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 1 \
    --only-allreduce \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_alpha_error.log 2>&1 &
done

# multitree-beta
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=multitree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/cmesh44multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --only-allreduce \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_beta_error.log 2>&1 &
done

# multitree-gamma
for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet50 $mlperfdir/Transformer $mlperfdir/Transformer_short \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  allreduce=multitree
  python $SIMHOME/src/simulate.py \
    --network $nnpath.csv \
    --run-name ${nn} \
    --booksim-network mesh \
    --booksim-config $SIMHOME/src/booksim2/runfiles/cmesh44multitree.cfg \
    --allreduce $allreduce \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --only-allreduce \
    --message-buffer-size 32 \
    --message-size 0 \
    --sub-message-size 256 \
    > $outdir/${nn}_${allreduce}_gamma_error.log 2>&1 &
done

#!/bin/sh

outdir=$SIMHOME/results/strong_scalability_logs

mkdir -p $outdir

mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

data=8388608 # parameters, * 4bytes = 32MB

## ring
for nodes in 16 36 64 100 144 196 256
do
  python $SIMHOME/src/simulate.py \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/torus44express.cfg \
    --allreduce ring \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_ring_error.log 2>&1 &
done

# mxnettree-alpah
for nodes in 16 36 64 100 144 196 256
do
  python $SIMHOME/src/simulate.py \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/torus44express.cfg \
    --allreduce mxnettree \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_mxnettree_alpha_error.log 2>&1 &
done

# mxnettree-beta
for nodes in 16 36 64 100 144 196 256
do
  python $SIMHOME/src/simulate.py \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
    --allreduce mxnettree \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_mxnettree_beta_error.log 2>&1 &
done

# multitree-beta
for nodes in 16 36 64 100 144 196 256
do
  python $SIMHOME/src/simulate.py \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_multitree_beta_error.log 2>&1 &
done

# multitree-gamma
for nodes in 16 36 64 100 144 196 256
do
  python $SIMHOME/src/simulate.py \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus44multitree.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 0 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_multitree_gamma_error.log 2>&1 &
done

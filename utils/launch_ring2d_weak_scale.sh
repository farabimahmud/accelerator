#!/bin/sh

outdir=$SIMHOME/results/weak_scalability_logs

mkdir -p $outdir

datachunk=98304 # parameters, * 4bytes = 375KB

# 2d-ring
#for nodes in 16 36 64 100 144 196 256
for dim in `seq 4 2 16`
do
  nodes=$(($dim*$dim))
  data=$(($datachunk*$nodes))
  #echo "dim: $dim, nodes: $nodes"

  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/ctorus${dim}x${dim}multitree.cfg \
    --allreduce ring2d \
    --outdir $outdir \
    --kary 5 \
    --radix 4 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_ring2d_error.log 2>&1 &
done


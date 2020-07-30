#!/bin/sh

# ring
for nodes in 32 #16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/bigraph.cfg \
    --allreduce ring \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network bigraph \
    --bigraph-m 4 \
    --bigraph-n 8 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_ring_error.log 2>&1 &
done

# mxnettree
#for nodes in 32 #16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/bigraph44express.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 1 \
#    --booksim-network bigraph \
#    --message-buffer-size 32 \
#    --message-size 256 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_alpha_error.log 2>&1 &
#done

# mxnettree-beta
#for nodes in 32 #16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/cbigraph44multitree.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 4 \
#    --booksim-network bigraph \
#    --message-buffer-size 32 \
#    --message-size 256 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_beta_error.log 2>&1 &
#done

# mxnettree-gamma
#for nodes in 32 #16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/cbigraph44multitree.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 4 \
#    --booksim-network bigraph \
#    --message-buffer-size 32 \
#    --message-size 0 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_gamma_error.log 2>&1 &
#done

# multitree-alpha
for nodes in 32 #16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/bigraph.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network bigraph \
    --bigraph-m 4 \
    --bigraph-n 8 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --strict-schedule \
    --prioritize-schedule \
    --estimate-lockstep \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_multitree_alpha_error.log 2>&1 &
done

# multitree-gamma
for nodes in 32 #16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/bigraph.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network bigraph \
    --bigraph-m 4 \
    --bigraph-n 8 \
    --message-buffer-size 32 \
    --message-size 0 \
    --sub-message-size 256 \
    --strict-schedule \
    --prioritize-schedule \
    --estimate-lockstep \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_multitree_gamma_error.log 2>&1 &
done

# hdrm
for nodes in 32 #16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/bigraph.cfg \
    --allreduce hdrm \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network bigraph \
    --bigraph-m 4 \
    --bigraph-n 8 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_hdrm_error.log 2>&1 &
done

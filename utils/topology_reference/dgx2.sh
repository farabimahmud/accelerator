#!/bin/sh

# data: 32KB 64KB 128KB 256KB 512KB 1MB 2MB 4MB 8MB 16MB ... 64MB
# example: 32KB -> synthetic-data-size 32*1024/4

# for dgx2:
#     run this data size
#     and also another set, data / 6

#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=16nodes_8388608data_multitree_gamma
#SBATCH --ntasks=1
#SBATCH --output=/home/grads/j/jyhuang/express-net/hpca2021/strong_scalability/16nodes_8388608data_multitree_gamma_error.log


# ring
for nodes in 16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/dgx2.cfg \
    --allreduce ring \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network dgx2 \
    --bigraph-m 8 \
    --bigraph-n 2 \
    --message-buffer-size 32 \
    --message-size 256 \
    --sub-message-size 256 \
    --only-allreduce \
    --synthetic-data-size $data \
    > $outdir/${nodes}nodes_${data}data_ring_error.log 2>&1 &
done

# mxnettree
#for nodes in 16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/dgx244express.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 1 \
#    --booksim-network dgx2 \
#    --message-buffer-size 32 \
#    --message-size 256 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_alpha_error.log 2>&1 &
#done

# mxnettree-beta
#for nodes in 16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/cdgx244multitree.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 4 \
#    --booksim-network dgx2 \
#    --message-buffer-size 32 \
#    --message-size 256 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_beta_error.log 2>&1 &
#done

# mxnettree-gamma
#for nodes in 16 #36 64 100 144 196 256
#do
#  data=$(($datachunk*$nodes))
#  python $SIMHOME/src/simulate.py \
#    --num-hmcs ${nodes} \
#    --run-name ${nodes}nodes_${data}data \
#    --booksim-config $SIMHOME/src/booksim2/runfiles/cdgx244multitree.cfg \
#    --allreduce mxnettree \
#    --outdir $outdir \
#    --kary 2 \
#    --radix 4 \
#    --booksim-network dgx2 \
#    --message-buffer-size 32 \
#    --message-size 0 \
#    --sub-message-size 256 \
#    --strict-schedule \
#    $option \
#    --synthetic-data-size $data \
#    > $outdir/${nodes}nodes_${data}data_mxnettree_gamma_error.log 2>&1 &
#done

# multitree-alpha
for nodes in 16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/dgx2.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network dgx2 \
    --bigraph-m 8 \
    --bigraph-n 2 \
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
for nodes in 16 #36 64 100 144 196 256
do
  data=$(($datachunk*$nodes))
  python $SIMHOME/src/simulate.py \
    --num-hmcs ${nodes} \
    --run-name ${nodes}nodes_${data}data \
    --booksim-config $SIMHOME/src/booksim2/runfiles/dgx2.cfg \
    --allreduce multitree \
    --outdir $outdir \
    --kary 2 \
    --radix 1 \
    --booksim-network dgx2 \
    --bigraph-m 8 \
    --bigraph-n 2 \
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

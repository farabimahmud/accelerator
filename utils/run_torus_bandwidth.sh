#!/bin/sh

booksim_net=torus

nodes_array=(16 64)
scale_array=("4x4" "8x8")
allreduces=(ring dtree ring2d multitree multitree multitree)
names=(ring dtree ring2d multitree_alpha multitree_beta multitree_delta)
booksim_configs_first=(torus torus ctorus torus ctorus ctorus)
booksim_configs_second=(express express multitree express multitree multitree)
baseline_options="--kary 2 --radix 1 --message-size 256"
ring2d_options="--kary 5 --radix 4 --message-size 256"
multitree_alpha_options="--kary 2 --radix 1 --message-size 256 --strict-schedule --prioritize-schedule --estimate-lockstep"
multitree_beta_options="--kary 5 --radix 4 --message-size 256 --strict-schedule --prioritize-schedule --estimate-lockstep"
multitree_delta_options="--kary 5 --radix 4 --message-size 0 --strict-schedule --prioritize-schedule --estimate-lockstep"

for i in ${!nodes_array[@]};
do
    nodes=${nodes_array[$i]}
    scale=${scale_array[$i]}

    outdir=$SIMHOME/results/isca2021/bandwidth/${booksim_net}${scale}_logs

    rm -rf $outdir
    mkdir -p $outdir

    for datasize in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536
    do
        for s in ${!names[@]};
        do
            name=${names[$s]}
            allreduce=${allreduces[$s]}
            booksim_config=$SIMHOME/src/booksim2/runfiles/${booksim_configs_first[$s]}${scale}${booksim_configs_second[$s]}.cfg
            case ${name} in
                ring|dtree)
                    options=${baseline_options}
                    ;;

                ring2d)
                    options=${ring2d_options}
                    ;;

                multitree_alpha)
                    options=${multitree_alpha_options}
                    ;;

                multitree_beta)
                    options=${multitree_beta_options}
                    ;;

                multitree_delta)
                    options=${multitree_delta_options}
                    ;;

                *)
                    echo "no option found for ${name}"
                    exit
                    ;;
            esac
            for element_size in 4
            do
                count=`ps aux | grep "only-allreduce" | wc -l`
                while [ $count -ge 40 ]; do
                    count=`ps aux | grep "only-allreduce" | wc -l`
                done
                num_elements=$((($datasize*1024)/$element_size))

                #jobname=${nodes}nodes_${datasize}kB_${element_size}elementsize_${booksim_net}_dtree
                logfile=$outdir/${nodes}nodes_${datasize}kB_${element_size}elementsize_${name}_error.log

                cmd="python $SIMHOME/src/simulate.py
                --num-hmcs ${nodes}
                --run-name ${nodes}nodes_${datasize}kB_${element_size}elementsize
                --booksim-config ${booksim_config}
                --allreduce ${allreduce}
                --outdir $outdir
                --booksim-network ${booksim_net}
                --message-buffer-size 32
                --sub-message-size 256
                --only-allreduce
                --synthetic-data-size $num_elements
                ${options}
                > $logfile 2>&1 &"

                #echo $cmd

                python $SIMHOME/src/simulate.py \
                --num-hmcs ${nodes} \
                --run-name ${nodes}nodes_${datasize}kB_${element_size}elementsize \
                --booksim-config ${booksim_config} \
                --allreduce ${allreduce} \
                --outdir $outdir \
                --booksim-network ${booksim_net} \
                --message-buffer-size 32 \
                --sub-message-size 256 \
                --only-allreduce \
                --synthetic-data-size $num_elements \
                ${options} \
                > $logfile 2>&1 &
            done
        done
    done
done

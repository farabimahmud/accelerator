import argparse
import configparser as cp
import os
import time
import sys
import numpy as np
import math

sys.path.append('SCALE-Sim')
sys.path.append('booksim2/src')
sys.path.append('allreduce')

from model import Model
from hmc import HMC
from booksim import BookSim
from allreduce import construct_allreduce
from eventq import EventQueue
from message_buffer import MessageBuffer


def cleanup(args):
    cmd = 'mkdir ' + args.outdir + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + args.outdir + '/*sram* ' + args.outdir + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + args.outdir + '/*dram* ' + args.outdir + '/layer_wise'
    os.system(cmd)

    if args.dump == False:
        cmd = 'rm -rf ' + args.outdir + '/layer_wise'
        os.system(cmd)


def init():

    parser = argparse.ArgumentParser()

    parser.add_argument('--arch-config', default='./SCALE-Sim/configs/express.cfg',
                        help='accelerator architecture file, '
                             'default=SCALE-Sim/configs/express.cfg')
    parser.add_argument('--num-hmcs', default=16, type=int,
                        help='number of hybrid memory cubes, default=16')
    parser.add_argument('--num-vaults', default=16, type=int,
                        help='number of vaults per hybrid memory cube')
    parser.add_argument('--mini-batch-size', default=16, type=int,
                        help='number of mini batch size per hmc accelerator, distributed to all vault npu')
    parser.add_argument('--network', default='SCALE-Sim/topologies/conv_nets/alexnet.csv',
                        help='neural network architecture topology file, '
                             'default=SCALE-Sim/topologies/conv_nets/alexnet.csv')
    parser.add_argument('--run-name', default='',
                        help='naming for this experiment run, default is empty')
    parser.add_argument('--outdir', default='',
                        help='naming for the output directory, default is empty')
    parser.add_argument('--dump', default=False, action='store_true',
                        help='dump memory traces, default=False')
    parser.add_argument('--allreduce', default='multitree',
                        help='allreduce shedule (multitree or mxnettree or ring), default=multitree')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary allreduce trees, default is 2 (binary)')
    parser.add_argument('--booksim-config', default='', required=True,
                        help='required config file for booksim')

    args = parser.parse_args()

    config = cp.ConfigParser()
    config.read(args.arch_config)

    if not args.run_name:
        args.run_name = config.get('general', 'run_name')

    if not args.run_name:
        net_name = args.network.split('/')[-1].split('.')[0]
        args.run_name = net_name + args.data_flow

    path = './outputs/' + args.run_name
    args.outdir = path

    arch_sec = 'architecture_presets'

    args.pe_array_height= int(config.get(arch_sec, 'ArrayHeight'))
    args.pe_array_width = int(config.get(arch_sec, 'ArrayWidth'))

    args.ifmap_sram_size  = int(config.get(arch_sec, 'IfmapSramSz')) << 10 #* 1024
    args.filter_sram_size = int(config.get(arch_sec, 'FilterSramSz')) << 10 #* 1024
    args.ofmap_sram_size  = int(config.get(arch_sec, 'OfmapSramSz')) << 10 #* 1024

    args.ifmap_offset  = int(config.get(arch_sec, 'IfmapOffset'))
    args.filter_offset = int(config.get(arch_sec, 'FilterOffset'))
    args.ofmap_offset  = int(config.get(arch_sec, 'OfmapOffset'))
    args.ifmap_grad_offset  = int(config.get(arch_sec, 'IfmapGradOffset'))
    args.filter_grad_offset = int(config.get(arch_sec, 'FilterGradOffset'))
    args.ofmap_grad_offset  = int(config.get(arch_sec, 'OfmapGradOffset'))

    args.data_flow = config.get(arch_sec, 'Dataflow')

    # Create output directory
    if not os.path.exists("./outputs/"):
        os.system("mkdir ./outputs")

    if os.path.exists(args.outdir):
        t = time.time()
        old_path = args.outdir + '_' + str(t)
        os.system('mv ' + args.outdir + ' ' + old_path)
    os.system('mkdir ' + args.outdir)

    print("====================================================")
    print("******************* SCALE SIM **********************")
    print("====================================================")
    print("Array Size: \t", args.pe_array_height, "x", args.pe_array_width)
    print("SRAM IFMAP: \t", args.ifmap_sram_size)
    print("SRAM Filter: \t", args.filter_sram_size)
    print("SRAM OFMAP: \t", args.ofmap_sram_size)
    print("CSV file path: \t" + args.network)
    print("Dataflow: \t", args.data_flow)
    print("====================================================")

    global_eventq = EventQueue()

    model = Model(args)
    print('NN model size: {} hyperparameters'.format(model.size))

    network = BookSim(args, global_eventq)

    allreduce = construct_allreduce(args)
    allreduce.compute_schedule(args.kary)

    hmcs = []
    from_network_message_buffers = []
    to_network_message_buffers = []
    for i in range(args.num_hmcs):
        hmcs.append(HMC(i, args, global_eventq))
        hmcs[i].load_model(model)
        hmcs[i].startup()
        # connect with network
        from_network_message_buffers.append(MessageBuffer())
        to_network_message_buffers.append(MessageBuffer())
        from_network_message_buffers[i].set_consumer(hmcs[i])
        to_network_message_buffers[i].set_consumer(network)
        hmcs[i].set_message_buffers(from_network_message_buffers[i],
                to_network_message_buffers[i])
        hmcs[i].set_allreduce(allreduce)

    network.set_message_buffers(to_network_message_buffers,
            from_network_message_buffers)

    return args, global_eventq, model, hmcs, network


def do_sim_loop(eventq):

    while not eventq.empty():
        cur_cycle, events = eventq.next_events()

        for event in events:
            event.process(cur_cycle)


def main():

    args, global_eventq, model, hmcs, network = init()

    do_sim_loop(global_eventq)

    compute_cycles = hmcs[0].compute_cycles
    cycles = global_eventq.cycles
    communication_cycles = cycles - compute_cycles

    compute_percentile = compute_cycles / cycles * 100
    communication_percentile = communication_cycles / cycles * 100

    print('\nTraining epoch runtime: {} cycles'.format(cycles))
    print(' - computation: {} cycles ({:.2f}%)'.format(compute_cycles, compute_percentile))
    print(' - communication: {} cycles ({:.2f}%)\n'.format(communication_cycles, communication_percentile))

    cleanup(args)


if __name__ == '__main__':
    main()

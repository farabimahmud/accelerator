import argparse
import configparser as cp
import os
import time
import sys
import numpy as np

sys.path.append('SCALE-Sim')
sys.path.append('booksim2/src')

from collective_comm import *
from model import Model
from hmc import HMC
import pybooksim


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


def main():

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
    parser.add_argument('--collective', default='tree',
                        help='collective communication shedule (tree or ring), default=tree')
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

    args.ifmap_sram_size  = int(config.get(arch_sec, 'IfmapSramSz')) *1024
    args.filter_sram_size = int(config.get(arch_sec, 'FilterSramSz')) *1024
    args.ofmap_sram_size  = int(config.get(arch_sec, 'OfmapSramSz')) *1024

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

    cycles = 0

    model = Model(args)
    hmc = HMC(args)
    booksim = pybooksim.BookSim(args.booksim_config)
    if args.collective == 'tree':
        cc = TreeCC(args)
    elif args.collective == 'ring':
        cc = RingCC(args)
    else:
        raise RuntimeError('Unknow collective communication schedule: ' + args.collective)


    compute_cycles = hmc.train(model)
    cycles += compute_cycles
    print('training compute cycles: ', compute_cycles)

    compute_cycles = hmc.aggregate(model)
    cycles += compute_cycles
    print('in-hmc weight aggregate cycles: ', compute_cycles)

    booksim.SetSimTime(int(cycles))

    num_messages = model.size * 4 / 64; # message size assumed 64 bytes for now

    iteration = 0
    future_comm = 0
    future_cycles = np.zeros(args.num_hmcs, dtype=int)
    levels = np.zeros(args.num_hmcs, dtype=int)
    num_messages_remained = 0
    num_messages_to_send = np.zeros(args.num_hmcs, dtype=int)
    num_messages_received = np.zeros(args.num_hmcs, dtype=int)

    for src, dest in cc.get_reduce_pairs(iteration).items():
        num_messages_to_send[src] = num_messages
        num_messages_remained += num_messages

    # Reduce phase for weight delta aggragation
    while num_messages_remained or booksim.Idle() == False or future_comm:
        # send messages
        for src in range(args.num_hmcs):
            if num_messages_to_send[src]:
                dest = cc.get_reduce_dest(levels[src], src)
                booksim.IssueMessage(src, dest, -1, pybooksim.Message.WriteRequest)
                num_messages_to_send[src] -= 1
                num_messages_remained -= 1

        # run interconnect for 1 cycle
        booksim.WakeUp()

        # peek and receive messages
        for i in range(args.num_hmcs):
            mid = booksim.PeekMessage(i, 0)
            if mid != -1:
                num_messages_received[i] += 1
                #print('HMC ', i, ' receives a message (id:', mid, ')')

                if num_messages_received[i] == model.size * 4 / 64:
                    #print('schedule-level', levels[i], 'HMC', i, 'received all messages at cycle:', booksim.GetSimTime())
                    num_messages_received[i] = 0
                    future_cycles[i] = booksim.GetSimTime() + hmc.aggregate(model)
                    levels[i] += 1
                    if levels[i] < cc.get_iterations() and cc.reduce_sender_in_iteration(levels[i], i):
                        future_comm += 1

            if booksim.GetSimTime() == future_cycles[i] and \
                    levels[i] < cc.get_iterations() and \
                    cc.reduce_sender_in_iteration(levels[i], i):
                future_comm -= 1
                num_messages_to_send[i] = num_messages
                num_messages_remained += num_messages

    print('max future_cycles: {}, booksim time: {}'.format(max(future_cycles), booksim.GetSimTime()))
    reduce_comm_cycles = max(booksim.GetSimTime(), max(future_cycles)) - cycles
    cycles += reduce_comm_cycles
    print('reduce communication cycles: {}'.format(reduce_comm_cycles))

    # Broadcast phase after weight delta reduction
    levels = np.zeros(args.num_hmcs, dtype=int)
    iteration = 0

    for src, dest in cc.get_broadcast_pairs(iteration).items():
        num_messages_to_send[src] = num_messages
        num_messages_remained += num_messages

    booksim.SetSimTime(cycles)
    while num_messages_remained or booksim.Idle() == False:
        # send messages
        for src in range(args.num_hmcs):
            if num_messages_to_send[src]:
                dest = cc.get_broadcast_dest(levels[src], src)
                mid = booksim.IssueMessage(src, dest, -1, pybooksim.Message.WriteRequest)
                if mid != -1:
                    num_messages_to_send[src] -= 1
                    num_messages_remained -= 1
                    if num_messages_to_send[src] == 0:
                        levels[dest] = levels[src]
                        levels[src] += 1
                        if levels[src] < cc.get_iterations():
                            assert cc.broadcast_sender_in_iteration(levels[src], src)
                            num_messages_to_send[src] = num_messages
                            num_messages_remained += num_messages

        # run interconnect for 1 cycle
        booksim.WakeUp()

        # peek and receive messages
        for i in range(args.num_hmcs):
            mid = booksim.PeekMessage(i, 0)
            if mid != -1:
                num_messages_received[i] += 1
                #print('HMC ', i, ' receives a message (id:', mid, ')')

                if num_messages_received[i] == num_messages:
                    #print('schedule-level', levels[i], 'HMC', i, 'received all messages at cycle:', booksim.GetSimTime())
                    num_messages_received[i] = 0
                    levels[i] += 1
                    if levels[i] < cc.get_iterations():
                        assert cc.broadcast_sender_in_iteration(levels[i], i)
                        num_messages_to_send[i] = num_messages
                        num_messages_remained += num_messages

    broadcast_comm_cycles = booksim.GetSimTime() - cycles
    print('broadcast communication cycles: {}'.format(broadcast_comm_cycles))
    cycles += broadcast_comm_cycles
    reduce_cycle_percent = reduce_comm_cycles / cycles * 100
    broadcast_cycle_percent = broadcast_comm_cycles / cycles * 100
    print('reduce cycles fraction: {:.2f} %, broadcast cycles fraction: {:.2f} %'.format(reduce_cycle_percent, broadcast_cycle_percent))

    cleanup(args)

    print('Training epoch cycles: ', cycles)

if __name__ == '__main__':
    main()

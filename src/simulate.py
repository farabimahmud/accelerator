import argparse
import configparser as cp
import os
import time
import sys
import numpy as np
import math
import logging
import json

sys.path.append('{}/src/SCALE-Sim'.format(os.environ['SIMHOME']))
sys.path.append('{}/src/booksim2/src'.format(os.environ['SIMHOME']))
sys.path.append('{}/src/allreduce'.format(os.environ['SIMHOME']))

from model import Model
from hmc import HMC
from booksim import BookSim
from allreduce import construct_allreduce
from eventq import EventQueue
from message_buffer import MessageBuffer

logger = logging.getLogger(__name__)

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

    parser.add_argument('--arch-config', default='{}/src/SCALE-Sim/configs/express.cfg'.format(os.environ['SIMHOME']),
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
    parser.add_argument('-d', '--outdir', default='',
                        help='naming for the output directory, default is empty')
    parser.add_argument('--dump', default=False, action='store_true',
                        help='dump memory traces, default=False')
    parser.add_argument('--allreduce', default='multitree',
                        help='allreduce shedule (multitree or mxnettree or ring), default=multitree')
    parser.add_argument('-k', '--kary', default=2, type=int,
                        help='generay kary allreduce trees, default is 2 (binary)')
    parser.add_argument('--radix', default=4, type=int,
                        help='node radix connected to router (end node NIs), default is 4')
    parser.add_argument('--booksim-config', default='', required=True,
                        help='required config file for booksim')
    parser.add_argument('-l', '--enable-logger', default=[], action='append',
                        help='Enable logging for a specific module, append module name')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Set the log level to debug, printing out detailed messages during execution.')
    parser.add_argument('--only-allreduce', default=False, action='store_true',
                        help='Set the flag to only run allreduce communication')
    parser.add_argument('--message-buffer-size', default=0, type=int,
                        help='message buffer size, default is 0 (infinite)')
    parser.add_argument('--message-size', default=64, type=int,
                        help='size of message, default is 64 bytes')
    parser.add_argument('--strict-schedule', default=False, action='store_true',
                        help='Strict schedule, default=False')

    args = parser.parse_args()

    if args.outdir:
        args.logdir = args.outdir
    else:
        logpath = '{}/results/logs'.format(os.environ['SIMHOME'])
        args.logdir = logpath
    os.system('mkdir -p {}'.format(args.outdir))
    logfile = '{}/{}_{}.log'.format(args.logdir, args.run_name, args.allreduce)
    jsonfile = '{}/{}_{}.json'.format(args.logdir, args.run_name, args.allreduce)
    if os.path.exists(logfile) or os.path.exists(jsonfile):
        raise RuntimeError('Warn: {} or {} already existed, may overwritten'.format(logfile, jsonfile))

    if args.verbose:
        logging.basicConfig(filename=logfile, format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=logfile, format='%(message)s', level=logging.INFO)

    for scope in args.enable_logger:
        debug_logger = logging.getLogger(scope)
        debug_logger.setLevel(logging.DEBUG)

    config = cp.ConfigParser()
    config.read(args.arch_config)

    if not args.run_name:
        args.run_name = config.get('general', 'run_name')

    if not args.run_name:
        net_name = args.network.split('/')[-1].split('.')[0]
        args.run_name = net_name + args.data_flow

    path = './outputs/' + args.run_name
    args.outdir = '{}/outputs/{}'.format(args.outdir, args.run_name)

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
    if args.dump:
        if not os.path.exists(args.outdir):
            os.system('mkdir -p {}'.format(args.outdir))
        elif os.path.exists(args.outdir):
            t = time.time()
            old_path = args.outdir + '_' + str(t)
            os.system('mv ' + args.outdir + ' ' + old_path)

    logger.info("====================================================")
    logger.info("******************* SCALE SIM **********************")
    logger.info("====================================================")
    logger.info("Array Size:    {} x {}".format(args.pe_array_height, args.pe_array_width))
    logger.info("SRAM IFMAP:    {}".format(args.ifmap_sram_size))
    logger.info("SRAM Filter:   {}".format(args.filter_sram_size))
    logger.info("SRAM OFMAP:    {}".format(args.ofmap_sram_size))
    logger.info("CSV file path: {}".format(args.network))
    logger.info("Dataflow:      {}".format(args.data_flow))
    logger.info("====================================================\n")

    global_eventq = EventQueue()

    model = Model(args)
    logger.info('NN model size: {} parameters\n'.format(model.size))

    network = BookSim(args, global_eventq)
    if args.message_buffer_size == 0:
        inject_buf_size = network.booksim.GetInjectBufferSize()
        msg_buf_size = network.booksim.GetMessageBufferSize()
        if inject_buf_size != 0 or msg_buf_size != 0:
            raise RuntimeError('Message buffer is set to 0 (infinite) here,'
                    ' but message buffer size and inject buffer size are set'
                    'to {} and {} in booksim config (should set to 0 for'
                    ' infinite)'.format(msg_buf_size, inject_buf_size))

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
        from_network_message_buffers.append([])
        to_network_message_buffers.append([])
        for j in range(args.radix):
            from_network_message_buffers[i].append(MessageBuffer('from_network_node{}_ni{}'.format(i, j), args.message_buffer_size))
            to_network_message_buffers[i].append(MessageBuffer('to_network_node{}_ni{}'.format(i, j), args.message_buffer_size))
            from_network_message_buffers[i][j].set_consumer(hmcs[i])
            to_network_message_buffers[i][j].set_consumer(network)
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

    assert network.booksim.Idle()
    logger.debug('booksim network idle? {}'.format(network.booksim.Idle()))
    for i, hmc in enumerate(hmcs):
        logger.debug('HMC {}:'.format(i))
        logger.debug('   reduce-scatter-schedule:')
        assert len(hmc.reduce_scatter_schedule) == 0
        for schedule in hmc.reduce_scatter_schedule:
            logger.debug('       {}'.format(schedule))
        logger.debug('   all-gather-schedule:')
        assert len(hmc.all_gather_schedule) == 0
        for schedule in hmc.all_gather_schedule:
            logger.debug('       {}'.format(schedule))
        logger.debug('   from network message buffers:')
        for i, message_buffer in enumerate(hmc.from_network_message_buffers):
            assert message_buffer.size == 0
            logger.debug('       {}-{}: has {} messages'.format(i, message_buffer.name, message_buffer.size))
        logger.debug('   to network message buffers:')
        for i, message_buffer in enumerate(hmc.to_network_message_buffers):
            assert message_buffer.size == 0
            logger.debug('       {}-{}: has {} messages'.format(i, message_buffer.name, message_buffer.size))

    compute_cycles = hmcs[0].compute_cycles
    allreduce_compute_cycles = 0
    for hmc in hmcs:
        if allreduce_compute_cycles < hmc.allreduce_compute_cycles:
            allreduce_compute_cycles = hmc.allreduce_compute_cycles
    cycles = global_eventq.cycles
    allreduce_cycles = cycles - compute_cycles
    pure_communication_cycles = allreduce_cycles - allreduce_compute_cycles

    compute_percentile = compute_cycles / cycles * 100
    allreduce_percentile = allreduce_cycles / cycles * 100
    allreduce_compute_percentile = allreduce_compute_cycles / cycles * 100
    pure_communication_percentile = allreduce_percentile - allreduce_compute_percentile

    logger.info('\n======== Simulation Summary ========')
    logger.info('Training epoch runtime: {} cycles'.format(cycles))
    logger.info(' - computation: {} cycles ({:.2f}%)'.format(compute_cycles, compute_percentile))
    logger.info(' - allreduce: {} cycles ({:.2f}%)'.format(allreduce_cycles, allreduce_percentile))
    logger.info('     - overlapped computation: {} cycles ({:.2f}%)'.format(allreduce_compute_cycles, allreduce_compute_percentile))
    logger.info('     - pure communication: {} cycles ({:.2f}%)\n'.format(pure_communication_cycles, pure_communication_percentile))

    if args.dump:
        cleanup(args)

    # dump configuration and results
    sim = {}
    sim['configuration'] = vars(args)
    sim['results'] = {}

    sim['results']['performance'] = {}
    sim['results']['performance']['training'] = compute_cycles
    sim['results']['performance']['allreduce'] = {}
    sim['results']['performance']['allreduce']['computation'] = allreduce_compute_cycles
    sim['results']['performance']['allreduce']['communication'] = pure_communication_cycles
    sim['results']['performance']['allreduce']['total'] = allreduce_cycles
    sim['results']['performance']['total'] = cycles

    network.booksim.CalculatePower()
    net_dyn_power = network.booksim.GetNetDynPower()
    net_leak_power = network.booksim.GetNetLeakPower()
    router_dyn_power = network.booksim.GetRouterDynPower()
    router_leak_power = network.booksim.GetRouterLeakPower()
    link_dyn_power = network.booksim.GetLinkDynPower()
    link_leak_power = network.booksim.GetLinkLeakPower()
    sim['results']['power'] = {}
    sim['results']['power']['network'] = {}
    sim['results']['power']['network']['dynamic'] = net_dyn_power
    sim['results']['power']['network']['static'] = net_leak_power
    sim['results']['power']['network']['total'] = net_dyn_power + net_leak_power
    sim['results']['power']['network']['router'] = {}
    sim['results']['power']['network']['router']['dynamic'] = router_dyn_power
    sim['results']['power']['network']['router']['static'] = router_leak_power
    sim['results']['power']['network']['router']['total'] = router_dyn_power + router_leak_power
    sim['results']['power']['network']['link'] = {}
    sim['results']['power']['network']['link']['dynamic'] = link_dyn_power
    sim['results']['power']['network']['link']['static'] = link_leak_power
    sim['results']['power']['network']['link']['total'] = link_dyn_power + link_leak_power

    jsonpath = '{}/{}_{}.json'.format(args.logdir, args.run_name, args.allreduce)
    with open(jsonpath, 'w') as simfile:
        json.dump(sim, simfile, indent=4)
        simfile.close()


if __name__ == '__main__':
    main()

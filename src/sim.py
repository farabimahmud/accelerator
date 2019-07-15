import argparse
import configparser as cp
import os
import time
import sys

sys.path.append('SCALE-Sim')
sys.path.append('booksim2/src')

from model import Model
from hmc import HMC
import pybooksim


def cleanup(args):
    if not os.path.exists("./outputs/"):
        os.system("mkdir ./outputs")

    net_name = args.network.split('/')[-1].split('.')[0]

    if args.run_name == '':
        path = './outputs/' + net_name + '_' + self.data_flow
    else:
        path = './outputs/' + args.run_name

    if os.path.exists(path):
        t = time.time()
        old_path = path + '_' + str(t)
        os.system('mv ' + path + ' ' + old_path)
    os.system('mkdir ' + path)

    cmd = 'mv *.csv ' + path
    os.system(cmd)

    cmd = 'mkdir ' + path + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + path + '/*sram* ' + path + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + path + '/*dram* ' + path + '/layer_wise'
    os.system(cmd)

    if args.dump == False:
        cmd = 'rm -rf ' + path + '/layer_wise'
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
    parser.add_argument('--dump', default=False, action='store_true',
                        help='dump memory traces, default=False')
    parser.add_argument('--booksim-config', default='', required=True,
                        help='required config file for booksim')

    args = parser.parse_args()

    config = cp.ConfigParser()
    config.read(args.arch_config)

    if not args.run_name:
        args.run_name = config.get('general', 'run_name')

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

    compute_cycles = hmc.train(model)
    cycles += compute_cycles
    print('training compute cycles: ', compute_cycles)

    compute_cycles = hmc.aggregate(model)
    cycles += compute_cycles
    print('in-hmc weight aggregate cycles: ', compute_cycles)

    booksim.SetSimTime(int(cycles))
    booksim.WakeUp()
    comm_cycles = booksim.GetSimTime() - cycles
    print('communication cycles: ', comm_cycles)
    cycles += comm_cycles

    cleanup(args)

    print('Training cycles: ', cycles)

if __name__ == '__main__':
    main()

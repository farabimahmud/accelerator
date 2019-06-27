import argparse
import configparser as cp
from hmc import HMC

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--arch-config', default='./configs/express.cfg',
                        help='accelerator architecture file, '
                             'default=./configs/scale.cfg')
    parser.add_argument('--num-hmcs', default=16, type=int,
                        help='number of hybrid memory cubes, default=16')
    parser.add_argument('--network', default='./topologies/conv_nets/alexnet.csv',
                        help='neural network architecture topology file, ' 
                             'default=./topologies/conv_nets/alexnet.csv')
    parser.add_argument('--run-name', default='',
                        help='naming for this experiment run, default is empty')
    parser.add_argument('--dump', default=False, action='store_true',
                        help='dump memory traces, default=False')

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

    print(args)

    hmc = HMC(args)
    cycles = hmc.train()

    print('Training cycles: ', cycles)

if __name__ == '__main__':
    main()

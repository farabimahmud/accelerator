import os
import sys
import argparse

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from kncube import KNCube
from allreduce import Allreduce
from ring_allreduce import RingAllreduce
from dtree_allreduce import DTreeAllreduce
from multitree_allreduce import MultiTreeAllreduce
from mxnettree_allreduce import MXNetTreeAllreduce


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=None, type=int,
                        help='network dimension, default is 4')
    parser.add_argument('--mesh', default=False, action='store_true',
                        help='Create mesh network, default False (torus)')
    parser.add_argument('--sort', default=False, action='store_true',
                        help='Sort the trees for link allocation')

    args = parser.parse_args()
    if args.mesh:
        args.booksim_network = 'mesh'
    else:
        args.booksim_network = 'torus'

    if args.dimension == None:
        begin_dimension = 4
        end_dimension = 17
    else:
        begin_dimension = args.dimension
        end_dimension = args.dimension + 1
    for dimension in range(begin_dimension, end_dimension + 1, 2):
        args.dimension = dimension
        args.nodes = dimension * dimension
        args.kary = 2
        network = KNCube(args, args.mesh)
        network.build_graph()
        ring_allreduce = RingAllreduce(args, network)
        multitree_allreduce = MultiTreeAllreduce(args, network)
        dtree_allreduce = DTreeAllreduce(args, network)
        mxnettree_allreduce = MXNetTreeAllreduce(args, network)
        mxnettree_allreduce.silent = True
        print('Network size: {}'.format(args.nodes))
        ring_allreduce.compute_trees()
        multitree_allreduce.compute_trees(2, sort=args.sort)
        mxnettree_allreduce.compute_best_trees(10, 2, sort=args.sort)
        dtree_allreduce.compute_trees()
        print(' Timesteps for Ring: {}, 2Tree: {}, and binary trees - MultiTree: {}, MXNetTree: {}'.format(
            ring_allreduce.timesteps, dtree_allreduce.timesteps, multitree_allreduce.timesteps, mxnettree_allreduce.timesteps))
        for kary in range(3, 6):
            args.kary = kary
            network = KNCube(args, args.mesh)
            network.build_graph()
            multitree_allreduce = MultiTreeAllreduce(args, network)
            multitree_allreduce.compute_trees(kary, sort=args.sort)
            mxnettree_allreduce.mxnet_schedule(kary, sort=args.sort)
            print('                                         {}-ary trees - MultiTree: {}, MXNetTree: {}'.format(
                kary, multitree_allreduce.timesteps, mxnettree_allreduce.timesteps))


if __name__ == '__main__':
    main()

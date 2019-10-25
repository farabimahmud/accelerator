import networks
from ring_allreduce import RingAllreduce
from multitree_allreduce import MultiTreeAllreduce
from mxnettree_allreduce import MXNetTreeAllreduce


def main():
    begin_dimension = 4
    end_dimeinsion = 17
    for dimension in range(begin_dimension, end_dimeinsion + 1, 2):
        nodes = dimension * dimension
        network = networks.Torus(nodes, dimension)
        network.build_graph()
        ring_allreduce = RingAllreduce(network)
        multitree_allreduce = MultiTreeAllreduce(network)
        mxnettree_allreduce = MXNetTreeAllreduce(network)
        mxnettree_allreduce.silent = True
        print('Network size: {}'.format(nodes))
        ring_allreduce.compute_trees()
        multitree_allreduce.compute_trees(2)
        mxnettree_allreduce.compute_best_trees(10, 2)
        print(' Timesteps for Ring: {}, and binary trees - MultiTree: {}, MXNetTree: {}'.format(
            ring_allreduce.timesteps, multitree_allreduce.timesteps, mxnettree_allreduce.timesteps))
        for kary in range(3, 6):
            multitree_allreduce.compute_trees(kary)
            mxnettree_allreduce.mxnet_schedule(kary)
            print(' Timesteps for Ring: {}, and {}-ary trees - MultiTree: {}, MXNetTree: {}'.format(
                ring_allreduce.timesteps, kary, multitree_allreduce.timesteps, mxnettree_allreduce.timesteps))


if __name__ == '__main__':
    main()

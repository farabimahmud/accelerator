import networks
from multitree_allreduce import MultiTreeAllreduce
from mxnettree_allreduce import MXNetTreeAllreduce


def main():
    begin_dimension = 4
    end_dimeinsion = 11
    for dimension in range(begin_dimension, end_dimeinsion + 1, 2):
        nodes = dimension * dimension
        network = networks.Torus(nodes, dimension)
        network.build_graph()
        multitree_allreduce = MultiTreeAllreduce(network)
        mxnettree_allreduce = MXNetTreeAllreduce(network)
        mxnettree_allreduce.silent = True
        print('Network size: {}'.format(nodes))
        multitree_allreduce.compute_trees(2)
        mxnettree_allreduce.compute_trees(2)
        print(' Iterations for binary trees - MultiTree: {}, MXNetTree: {}'.format(
            multitree_allreduce.iterations, mxnettree_allreduce.iterations))
        for kary in range(3, 6):
            multitree_allreduce.compute_trees(kary)
            mxnettree_allreduce.mxnet_schedule(kary)
            print(' Iterations for {}-ary trees - MultiTree: {}, MXNetTree: {}'.format(
                kary, multitree_allreduce.iterations, mxnettree_allreduce.iterations))


if __name__ == '__main__':
    main()

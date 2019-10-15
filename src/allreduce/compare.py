import networks
from multitree_allreduce import MultiTreeAllreduce
from mxnettree_allreduce import MXNetTreeAllreduce


def main():
    dimension = 4
    nodes = dimension * dimension
    network = networks.Torus(nodes, dimension)
    network.build_graph()
    multitree_allreduce = MultiTreeAllreduce(network)
    mxnettree_allreduce = MXNetTreeAllreduce(network)
    multitree_allreduce.compute_trees(2)
    mxnettree_allreduce.compute_trees(2)
    print('Iterations - MultiTree: {}, MXNetTree: {}'.format(multitree_allreduce.iterations, mxnettree_allreduce.iterations))


if __name__ == '__main__':
    main()

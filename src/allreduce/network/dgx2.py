import argparse
import numpy as np

from network import Network

class DGX2(Network):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'DGX2'
        self.m = args.bigraph_m # No. sub-nodes per physical group
        self.n = args.bigraph_n # No. switches
        assert self.nodes == self.m * self.n


    '''
    build_graph() - build the topology graph
    @filename: filename to generate topology dotfile, optional
    '''
    def build_graph(self, filename=None, verbose=False):
        n = 0
        for sw in range(self.n):
            self.switch_to_node[sw] = []
            self.switch_to_switch[sw] = []
            for node in range(self.m):
                self.switch_to_node[sw].append(n)
                self.node_to_switch[n] = (sw, 1)
                n += 1

        # upper and switch
        for sw in range(self.n // 2):
            for neighbor_sw in range(self.n // 2, self.n):
              for i in range(self.m):
                self.switch_to_switch[sw].append(neighbor_sw)
                self.switch_to_switch[neighbor_sw].append(sw)

        # adjacency list
        for node in range(self.nodes):
            self.from_nodes[node] = []
            self.to_nodes[node] = []

            neighbor_base = 0
            if node < self.m:
                neighbor_base = self.m
            for n in range(self.m):
                neighbor = n + neighbor_base
                self.from_nodes[node].append(neighbor)
                self.to_nodes[node].append(neighbor)

        if verbose:
            print('DGX2 Topology:')
            print('  - Node connections:')
            for node in range(self.nodes):
                print('    node {} is connected to switch {}'.format(node, self.node_to_switch[node][0]))
            print('  - Switch connections:')
            for sw in range(self.n):
                print('    switch {}:'.format(sw))
                print('      connects to switches {}'.format(self.switch_to_switch[sw]))
                print('      connects to nodes {}'.format(self.switch_to_node[sw]))
            print('  - From and To nodes:')
            for node in range(self.nodes):
                print('    node {}\'s from nodes {}'.format(node, self.from_nodes[node]))
    # def build_graph(self, filename=None)


    def distance(self, src, dest):
        pass


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=1, type=int,
                        help='network dimension, default is 1')
    parser.add_argument('--dgx2_c', default=8, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--dgx2_n', default=2, type=int,
                        help='# switches')
    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches')
    parser.add_argument('--dgx2_k', default=1, type=int,
                        help='#network radix')
    parser.add_argument('--nodes', default=16, type=int,
                        help='#nodes')

    args = parser.parse_args()

    network = DGX2(args)
    network.build_graph(verbose=True)


if __name__ == '__main__':
    test()

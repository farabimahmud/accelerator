import argparse
import numpy as np
from copy import deepcopy

from network import Network

class BiGraph(Network):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'BiGraph'
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
                self.switch_to_switch[sw].append(neighbor_sw)
                self.switch_to_switch[neighbor_sw].append(sw)

        if verbose:
            print('BiGraph Topology:')
            print('  - Node connections:')
            for node in range(self.nodes):
                print('    node {} is connected to switch {}'.format(node, self.node_to_switch[node][0]))
            print('  - Switch connections:')
            for sw in range(self.n):
                print('    switch {}:'.format(sw))
                print('      connects to switches {}'.format(self.switch_to_switch[sw]))
                print('      connects to nodes {}'.format(self.switch_to_node[sw]))

        self.ring = [0]
        for i in range(self.m):
            for j in range(self.n // 2):
                up_sw = self.ring[-1] // self.m
                down_sw = (up_sw + i) % (self.n // 2) + (self.n // 2)
                self.ring.append(down_sw * self.m + i)
                next_up_sw = (down_sw - self.n // 2 + i + 1) % (self.n // 2)
                if i == (self.m - 1) and j == (self.n // 2 - 1):
                    break
                if j == (self.n // 2 - 1):
                    self.ring.append(next_up_sw * self.m + i + 1)
                else:
                    self.ring.append(next_up_sw * self.m + i)

        switches = []
        for i in self.ring:
            switches.append(i // self.m)
        print('ring: {}'.format(self.ring))
        print('switches in ring: {}'.format(switches))

        node_to_switch = deepcopy(self.node_to_switch)
        switch_to_switch = deepcopy(self.switch_to_switch)
        switch_to_node = deepcopy(self.switch_to_node)

        # verify the ring with topology
        current = 0
        while current < self.nodes:
            current_node = self.ring[current]
            next_node = self.ring[(current + 1) % self.nodes]
            cur_sw = self.node_to_switch[current_node][0]
            node_to_switch[current_node] = None
            next_sw = self.node_to_switch[next_node][0]
            switch_to_switch[cur_sw].remove(next_sw)
            switch_to_node[next_sw].remove(next_node)
            current += 1

        #print('node-to-switch: {}'.format(node_to_switch))
        #print('switch-to-switch: {}'.format(switch_to_switch))
        #print('switch-to-node: {}'.format(switch_to_node))

        '''
        # ring for 32 nodes with m = 4 and n = 8
        # [0, 16, 4, 20, 8, 24, 12, 28, 1, 21, 13, 17, 9, 29, 5, 25, 10, 22
        #  6, 18, 2, 30, 14, 26, 7, 31, 11, 19, 15, 23, 3, 27]
        if self.m == 4 and self.n == 8:
            self.ring = [0, 16, 4, 20, 8, 24, 12, 28, 1, 21, 13, 17, 9, 29,
                    5, 25, 10, 22, 6, 18, 2, 30, 14, 26, 7, 31, 11, 19, 15,
                    23, 3, 27]

            node_to_switch = deepcopy(self.node_to_switch)
            switch_to_switch = deepcopy(self.switch_to_switch)
            switch_to_node = deepcopy(self.switch_to_node)

            # verify the ring with topology
            current = 0
            while current < self.nodes:
                current_node = self.ring[current]
                next_node = self.ring[(current + 1) % self.nodes]
                cur_sw = self.node_to_switch[current_node][0]
                node_to_switch[current_node] = None
                next_sw = self.node_to_switch[next_node][0]
                switch_to_switch[cur_sw].remove(next_sw)
                switch_to_node[next_sw].remove(next_node)
                current += 1

            #print('node-to-switch: {}'.format(node_to_switch))
            #print('switch-to-switch: {}'.format(switch_to_switch))
            #print('switch-to-node: {}'.format(switch_to_node))
        '''
    # def build_graph(self, filename=None)


    '''
    distance() - distance between two nodes in switch hops
    @src: source node ID
    @dest: destination node ID
    '''
    def distance(self, src, dest):
        src_sw = self.node_to_switch[src][0]
        dest_sw = self.node_to_switch[dest][0]
        if src_sw == dest_sw:
            return 1
        elif src_sw in self.switch_to_switch[dest_sw]:
            return 2
        else:
            return 3
    # end of distance()


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nodes', default=32, type=int,
                        help='number of end nodes, default is 4')
    parser.add_argument('--bigraph-m', default=4, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--bigraph-n', default=8, type=int,
                        help='# switches')

    args = parser.parse_args()

    network = BiGraph(args)
    network.build_graph(verbose=True)


if __name__ == '__main__':
    test()

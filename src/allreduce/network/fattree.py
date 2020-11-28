import argparse
import numpy as np

from network import Network

class FatTree(Network):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'FatTree'
        self.m = args.bigraph_m # No. sub-nodes per switch
        self.n = args.bigraph_n # No. level
        assert self.nodes == self.m ** self.n
        self.num_switches = self.n * (self.m ** (self.n - 1)) # levels * sw_per_level


    '''
    build_graph() - build the topology graph
    @filename: filename to generate topology dotfile, optional
    '''
    def build_graph(self, filename=None, verbose=False):
        # construct ring
        self.ring = []
        for i in range(self.m):
            for j in range(self.m):
                self.ring.append(i + j * self.m)

        for sw in range(self.num_switches):
            self.switch_to_switch[sw] = []

        num_pos = self.m ** (self.n - 1)

        # connecting nodes to leaf switches
        for pos in range(num_pos):
            sw = (self.n - 1) * (self.m ** (self.n - 1)) + pos
            self.switch_to_node[sw] = []
            for i in range(self.m):
                node = pos * self.m + i
                self.switch_to_node[sw].append(node)
                self.node_to_switch[node] = (sw, 1)

        # connect switches in higher levels
        num_links = (2 * self.m * (self.m * (self.n - 1))) * (self.n - 1)
        out_link_from_switch = {}
        in_link_to_switch = {}

        links_per_direction = self.m * (self.m ** (self.n - 1))
        links_per_level = 2 * links_per_direction

        # down output links
        for level in range(self.n - 1):
            for pos in range(num_pos):
                for port in range(self.m):
                    link = level * links_per_level + pos * self.m + port
                    sw = level * (self.m ** (self.n - 1)) + pos
                    out_link_from_switch[link] = sw

        # up output links
        for level in range(1, self.n):
            for pos in range(num_pos):
                for port in range(self.m):
                    link = level * links_per_level - links_per_direction  + pos * self.m + port
                    sw = level * (self.m ** (self.n - 1)) + pos
                    out_link_from_switch[link] = sw

        # down input links
        for level in range(self.n):
            switches_per_neightborhood = self.m ** (self.n - 1 - level)
            switches_per_branch = self.m ** (self.n - 1 - (level + 1))
            level_offset = switches_per_neightborhood * self.m
            for pos in range(num_pos):
                neighborhood = pos // switches_per_neightborhood
                neighborhood_pos = pos % switches_per_neightborhood
                for port in range(self.m):
                    # level, region in level, subregion in region, switch in subregion, switch port
                    link = ((level + 1) * links_per_level - links_per_direction) \
                            + neighborhood * level_offset \
                            + port * switches_per_branch * self.m \
                            + neighborhood_pos % switches_per_branch * self.m \
                            + neighborhood_pos // switches_per_branch
                    sw = level * (self.m ** (self.n - 1)) + pos
                    in_link_to_switch[link] = sw

        # up input links
        for level in range(1, self.n):
            switches_per_neightborhood = self.m ** (self.n - 1 - (level - 1))
            switches_per_branch = self.m ** (self.n - 1 - level)
            level_offset = switches_per_neightborhood * self.m
            for pos in range(num_pos):
                neighborhood = pos // switches_per_neightborhood
                neighborhood_pos = pos % switches_per_neightborhood
                for port in range(self.m):
                    # level, region in level, subregion in region, switch in subregion, switch port
                    link = (level - 1) * links_per_level \
                            + neighborhood * level_offset \
                            + port * switches_per_branch * self.m \
                            + neighborhood_pos % switches_per_branch * self.m \
                            + neighborhood_pos // switches_per_branch
                    sw = level * (self.m ** (self.n - 1)) + pos
                    in_link_to_switch[link] = sw

        # connect switches
        for link in range(num_links):
            from_sw = out_link_from_switch[link]
            to_sw = in_link_to_switch[link]
            self.switch_to_switch[from_sw].append(to_sw)

        if verbose:
            print('FatTree Topology:')
            print('  - Node connections:')
            for node in range(self.nodes):
                print('    node {} is connected to switch {}'.format(node, self.node_to_switch[node][0]))
            print('  - Switch connections:')
            for sw in range(self.num_switches):
                print('    switch {}:'.format(sw))
                print('      connects to switches {}'.format(self.switch_to_switch[sw]))
                if sw in self.switch_to_node.keys():
                    print('      connects to nodes {}'.format(self.switch_to_node[sw]))
            print('  - Ring: {}'.format(self.ring))
    # def build_graph(self, filename=None)


    '''
    distance() - distance between two nodes
    @src: source node ID
    @dest: destination node ID
    '''
    def distance(self, src, dest):
        if self.node_to_switch[src][0] == self.node_to_switch[dest][0]:
            return 1
        else:
            return self.n + 1
    # end of distance()


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches')
    parser.add_argument('--nodes', default=64, type=int,
                        help='#nodes')

    args = parser.parse_args()

    network = FatTree(args)
    network.build_graph(verbose=True)


if __name__ == '__main__':
    test()

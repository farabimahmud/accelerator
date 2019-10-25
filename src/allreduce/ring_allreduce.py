import argparse

import networks
from allreduce import Allreduce


class RingAllreduce(Allreduce):
    def __init__(self, network):
        super().__init__(network)
        self.ring = []


    '''
    compute_trees() - computes allreduce rings (special tree) for the given network
    @kary: not used, skip
    @alternate: not used, skip
    @sort: not used, skip
    @verbose: print detailed info of ring construction process
    '''
    def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False):
        # initialize empty a ring
        self.ring.append(0)

        # construct a ring
        explored = {}
        while True:
            # current node
            current = self.ring[-1]

            if current not in explored.keys():
                explored[current] = []

            next_node = None
            for neightbor in self.network.to_nodes[current]:
                if neightbor not in self.ring and neightbor not in explored[current]:
                    next_node = neightbor
                    break

            found = True
            if next_node:
                self.ring.append(next_node)
                explored[current].append(next_node)
                if len(self.ring) == self.network.nodes:
                    if self.ring[0] in self.network.to_nodes[next_node]:
                        break
                    else:
                        # doesn't lead to a valid solution, not circle
                        self.ring.pop()
            else:
                if verbose:
                    print('Cannot reach a solution from current state: {}, backtrack'.format(self.ring))
                # remove the current node since it cannot reach to a solution
                self.ring.pop()
                explored.pop(current)
                if not explored:
                    break

        if len(self.ring) == self.network.nodes:
            self.timesteps = self.network.nodes - 1
            if verbose:
                print('Ring found: {}'.format(self.ring))
            # form the 'trees'
            self.trees = {}
            for root in range(self.network.nodes):
                self.trees[root] = []
                root_idx = self.ring.index(root)
                for timestep in range(self.network.nodes - 1):
                    parent_idx = (timestep + root_idx) % self.network.nodes
                    child_idx = (timestep + root_idx + 1) % self.network.nodes

                    parent = self.ring[parent_idx]
                    child = self.ring[child_idx]

                    self.trees[root].append((child, parent, timestep))
                if verbose:
                    print('ring {}: {}'.format(root, self.trees[root]))
        else:
            print('No solution found! Check the topology graph')
    # def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False)


    '''
    generate_ring_dotfile() - generate dotfile for computed rings
    @filename: name of dotfile
    '''
    def generate_ring_dotfile(self, filename):
        # color palette for ploting nodes of different tree levels
        colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
                '#e7298a', '#ce1256', '#980043', '#67001f']

        ring = 'digraph ring {\n'
        ring += '  rankdir = BT;\n'
        ring += '  /* ring */\n'

        for i in range(1, self.network.nodes):
            ring += '  {} -> {};\n'.format(self.ring[i-1], self.ring[i])
        ring += '  {} -> {};\n'.format(self.ring[-1], self.ring[0])

        ring += '  // note that rank is used in the subgraph\n'
        ring += '  subgraph {\n'
        ring += '    {rank = same; ' + str(self.ring[0]) + ';}\n'
        for i in range(1, self.network.nodes // 2):
            ring += '    {rank = same; '
            ring += '{}; {};'.format(self.ring[i], self.ring[self.network.nodes - i])
            ring += '}\n'
        ring += '    {rank = same; ' + str(self.ring[self.network.nodes // 2]) + ';}\n'

        ring += '  } /* closing subgraph */\n'
        ring += '}\n'

        f = open(filename, 'w')
        f.write(ring)
        f.close()
    # def generate_ring_dotfile(self, filename)


def test(args):
    dimension = args.dimension
    nodes = dimension * dimension
    network = networks.Torus(nodes, dimension)
    network.build_graph()
    # network.to_nodes[1].clear() # test no solution case

    allreduce = RingAllreduce(network)
    allreduce.compute_trees(verbose=False)
    allreduce.generate_schedule()
    if args.gendotfile:
        allreduce.generate_ring_dotfile('ring.dot')
        allreduce.generate_trees_dotfile('ring_trees.dot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=4, type=int,
                        help='network dimension, default is 4')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')

    args = parser.parse_args()

    test(args)

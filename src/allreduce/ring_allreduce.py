import argparse
import sys
import os
from copy import deepcopy

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class RingAllreduce(Allreduce):
    def __init__(self, args, network):
        super().__init__(args, network)
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
            if self.network.type == 'BiGraph':
                assert self.network.m == 4 and self.network.n == 8
                self.ring = [0, 16, 4, 20, 8, 24, 12, 28, 1, 21, 13, 17, 9, 29,
                        5, 25, 10, 22, 6, 18, 2, 30, 14, 26, 7, 31, 11, 19, 15,
                        23, 3, 27]

                node_to_switch = deepcopy(self.network.node_to_switch)
                switch_to_switch = deepcopy(self.network.switch_to_switch)
                switch_to_node = deepcopy(self.network.switch_to_node)

                # verify the ring with topology
                current = 0
                while current < self.network.nodes:
                    current_node = self.ring[current]
                    next_node = self.ring[(current + 1) % self.network.nodes]
                    cur_sw = self.network.node_to_switch[current_node][0]
                    node_to_switch[current_node] = None
                    next_sw = self.network.node_to_switch[next_node][0]
                    switch_to_switch[cur_sw].remove(next_sw)
                    switch_to_node[next_sw].remove(next_node)
                    current += 1

                #print('node-to-switch: {}'.format(node_to_switch))
                #print('switch-to-switch: {}'.format(switch_to_switch))
                #print('switch-to-node: {}'.format(switch_to_node))

                break

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
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from ring,
           verified with generate_schedule in MultiTree
    '''
    def generate_schedule(self, verbose=False):
        # compute parent-children dependency
        self.trees_parent = {}
        self.trees_children = {}
        for root in range(self.network.nodes):
            self.trees_parent[root] = {}
            self.trees_parent[root][root] = None
            self.trees_children[root] = {}
            for node in range(self.network.nodes):
                self.trees_children[root][node] = []
            for edge in self.trees[root]:
                child = edge[0]
                parent = edge[1]
                self.trees_parent[root][child] = parent
                self.trees_children[root][parent].append(child)

        # initialize the schedules
        self.reduce_scatter_schedule = {}
        self.all_gather_schedule = {}

        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = []
            self.all_gather_schedule[node] = []

            index = self.ring.index(node)
            parent = self.ring[index - 1]
            child = self.ring[(index + 1) % self.network.nodes]

            # reduce-scatter scheduled from 'leaf'
            rs_subflow = child
            self.reduce_scatter_schedule[node].append({rs_subflow: ((parent, 0), [], 1, 0)})
            # all-gather scheduled from 'root'
            ag_subflow = node
            self.all_gather_schedule[node].append({ag_subflow: ([(child, 0)], None, 1, self.network.nodes)})
            # add remianing schedules
            for i in range(self.network.nodes - 2):
                # reduce-scatter
                rs_subflow = self.ring[(index + i + 2) % self.network.nodes]
                self.reduce_scatter_schedule[node].append({rs_subflow: ((parent, 0), [(rs_subflow, child)], 1, i + 1)})

                # all-gather
                ag_subflow = self.ring[index - i - 1]
                self.all_gather_schedule[node].append({ag_subflow: ([(child, 0)], (ag_subflow, parent), 1, i + 1 + self.network.nodes)})

            self.reduce_scatter_schedule[node].append({node: ((None, None), [(node, child)], 0, self.network.nodes - 1)})

            if verbose:
                print('Accelerator {}:'.format(node))
                print('  reduce-scatter schedule:')
                for timestep, schedule in enumerate(self.reduce_scatter_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
                print('  all-gather schedule:')
                for timestep, schedule in enumerate(self.all_gather_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
    # def generate_schedule(self, verbose=False)


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
    network = construct_network(args)
    # network.to_nodes[1].clear() # test no solution case

    allreduce = RingAllreduce(args, network)
    allreduce.compute_trees(verbose=args.verbose)
    allreduce.generate_schedule(verbose=args.verbose)
    allreduce.max_num_concurrent_flows()
    if args.gendotfile:
        allreduce.generate_ring_dotfile('ring.dot')
        allreduce.generate_trees_dotfile('ring_trees.dot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=16, type=int,
                        help='number of nodes, default is 16')
    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch), default 8')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches, default 2')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='detailed print')
    parser.add_argument('--booksim-network', default='torus',
                        help='network topology (torus | mesh | dgx2), default is torus')

    args = parser.parse_args()

    test(args)

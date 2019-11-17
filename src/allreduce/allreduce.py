import os
import numpy as np
from abc import ABC, abstractmethod


class Allreduce(ABC):
    def __init__(self, network):
        self.network = network
        self.trees = None
        self.trees_parent = None
        self.trees_children = None
        self.timesteps = None
        '''
        schedules are organized as list of list, the list with lower index
        in the schedule should be scheduled earlier.
        - reduce_scatter_schedule:
            subflow: (parent, [dependent children]) // subflow is 'tree' root
        - all_gather_schedule:
            subflow: ([children], dependent parent)
        Ring:
            0->1->2->3->0
            reduce_scatter_schedule[0] = [
                {3: (1, [])},
                {2: (1, [3])},
                {1: (1, [3])},
                {0: (None, [3])}
            ]
            all_gather_schedule[0] = [
                {0: ([1], None)},
                {3: ([1], 3)},
                {2: ([1], 3)}
            ]
        MXNet: (only dependencies among children and parent)
              Tree 0      Tree 1        Tree 2        Tree 3
                0           1             2             3
              0   1       1   3         2   3         3   1
            0  2 1  3   1  0 3  2     2  0 3  1     3  2 1  0
            reduce_scatter_schedule[3] = [
                {0: (1, []), 1: (1, [2]), 2: (2, [1]), 3: (None, [2, 1])}
            ]
            all_gather_schedule[3] = [
                {1: ([2], 1), 2: ([1], 2), 3: ([1, 2], None)}
            ]
        MultiTree:
            Timestep    Tree 0      Tree 1        Tree 2        Tree 3
                2         0           1             2             3
                1          2           3             0             1
                0       1   3       0   2         3   1         2   0
            reduce_scatter_schedule[0] = [
                {1: (1, []), 3: (1, [])},
                {2: (2, [1])},
                {0: (None, [1, 2])}
            ]
            all_gather_schedule[0] = [
                {0: ([2], None)},
                {0: ([1], None), 2: ([1], 2)}
            ]
        '''
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None


    '''
    compute_schedule() - computes spanning trees and schedule for the given network
    '''
    def compute_schedule(self, kary, alternate=True, sort=True, verbose=False):
        self.compute_trees(kary, alternate, sort, verbose)
        self.generate_schedule(verbose)


    '''
    compute_trees() - computes allreduce spanning trees for the given network
    '''
    @abstractmethod
    def compute_trees(self, kary, alternate=False, sort=True, verbose=False):
        pass


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from trees
    '''
    @abstractmethod
    def generate_schedule(self, verbose=False):
        pass


    '''
    generate_trees_dotfile() - generate dotfile for computed trees
    @filename: name of dotfile
    '''
    def generate_trees_dotfile(self, filename):
        # color palette for ploting nodes of different tree levels
        colors = ['#ffffff', '#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7',
                '#df65b0', '#e7298a', '#ce1256', '#980043', '#67001f']

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'

        # group nodes with same rank (same tree level/iteration)
        # and set up the map for node name and its rank in node_rank
        ranks = {}
        node_rank = {}
        for rank in range(self.timesteps + 1):
            ranks[rank] = []

        for root in range(self.network.nodes):
            minrank = self.timesteps
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                rank = edge[2] + 1
                ranks[rank].append(child)
                node_rank[child] = rank
                if edge[1] == root and rank - 1 < minrank:
                    minrank = rank - 1
            ranks[minrank].append('"{}-{}"'.format(root, root))
            node_rank['"{}-{}"'.format(root, root)] = minrank

        for root in range(self.network.nodes):
            tree += '    /* tree {} */\n'.format(root)
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = self.timesteps - edge[2]
                minlen = node_rank[child] - node_rank[parent] # for strict separation of ranks
                tree += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open(filename, 'w')
        f.write(tree)
        f.close()
    # def generate_trees_dotfile(self, filename)


    '''
    generate_per_tree_dotfile() - generate dotfile for each computed tree
    @filename: name of dotfile
    '''
    def generate_per_tree_dotfile(self, filename):
        cmd = 'mkdir ' + filename
        os.system(cmd)

        # color palette for ploting nodes of different tree levels
        colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
                '#e7298a', '#ce1256', '#980043', '#67001f']

        header = 'digraph tree {\n'
        header += '  rankdir = BT;\n'
        header += '  subgraph {\n'

        # group nodes with same rank (same tree level/iteration)
        # and set up the map for node name and its rank in node_rank
        ranks = {}
        node_rank = {}
        trees = {}
        for root in range(self.network.nodes):
            ranks[root] = {}
            node_rank[root] = {}
            for rank in range(self.timesteps + 1):
                ranks[root][rank] = []

        for root in range(self.network.nodes):
            minrank = self.timesteps
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                rank = edge[2] + 1
                ranks[root][rank].append(child)
                node_rank[root][child] = rank
                if edge[1] == root and rank - 1 < minrank:
                    minrank = rank - 1
            ranks[root][minrank].append('"{}-{}"'.format(root, root))
            node_rank[root]['"{}-{}"'.format(root, root)] = minrank

        for root in range(self.network.nodes):
            trees[root] = header + '    /* tree {} */\n'.format(root)
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = self.timesteps - edge[2]
                minlen = node_rank[root][child] - node_rank[root][parent] # for strict separation of ranks
                trees[root] += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        for root in range(self.network.nodes):
            trees[root] += '    // note that rank is used in the subgraph\n'
            for rank in range(self.timesteps + 1):
                if ranks[root][rank]:
                    level = '    {rank = same;'
                    for node in ranks[root][rank]:
                        level += ' {};'.format(node)
                    level += '}\n'
                    trees[root] += level

            trees[root] += '    // node colors\n'
            style = '    {} [style="filled", fillcolor="{}"];\n'
            for rank in range(self.timesteps + 1):
                if ranks[root][rank]:
                    trees[root] += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[root][rank])

            trees[root] += '  } /* closing subgraph */\n'
            trees[root] += '}\n'

            f = open('{}/tree-{}.dot'.format(filename, root), 'w')
            f.write(trees[root])
            f.close()
    # def generate_trees_dotfile(self, filename)


    '''
    max_num_concurrent_flows() - compute the concurrent flows for an accelerator
    '''
    def max_num_concurrent_flows(self):
        max_concurrent_reduce_scatter = np.zeros(self.network.nodes, dtype=int)
        max_concurrent_reduce_scatter_timestep = np.zeros(self.network.nodes, dtype=int)
        for root in range(self.network.nodes):
            timesteps = len(self.reduce_scatter_schedule[root])
            for timestep in range(timesteps):
                num_concurrent_reduce_scatter = len(self.reduce_scatter_schedule[root][timestep])
                if max_concurrent_reduce_scatter[root] < num_concurrent_reduce_scatter:
                    max_concurrent_reduce_scatter[root] = num_concurrent_reduce_scatter
                    max_concurrent_reduce_scatter_timestep[root] = timestep + 1

        max_concurrent_all_gather = np.zeros(self.network.nodes, dtype=int)
        max_concurrent_all_gather_timestep = np.zeros(self.network.nodes, dtype=int)
        for root in range(self.network.nodes):
            timesteps = len(self.all_gather_schedule[root])
            for timestep in range(timesteps):
                num_concurrent_all_gather = 0
                for flow, children_parent_dependency in self.all_gather_schedule[root][timestep].items():
                    num_concurrent_all_gather += len(children_parent_dependency[0])
                if max_concurrent_all_gather[root] < num_concurrent_all_gather:
                    max_concurrent_all_gather[root] = num_concurrent_all_gather
                    max_concurrent_all_gather_timestep[root] = timestep + 1

        for root in range(self.network.nodes):
            print('Tree {}:'.format(root))
            print('  - reduce-scatter schedules:')
            for timestep in range(len(self.reduce_scatter_schedule[root])):
                print('    step {}: {}'.format(timestep + 1, self.reduce_scatter_schedule[root][timestep]))
            print('  - all-gather schedules:')
            for timestep in range(len(self.all_gather_schedule[root])):
                print('    step {}: {}'.format(timestep + 1, self.all_gather_schedule[root][timestep]))
            print('  - max number of concurrent reduce-scatter is {} (at timestep {})'
                    ', and and all-gather communications is {} (at timestep {})'.format(
                        max_concurrent_reduce_scatter[root],
                        max_concurrent_reduce_scatter_timestep[root],
                        max_concurrent_all_gather[root],
                        max_concurrent_all_gather_timestep[root]))
    # end of max_num_concurrent_flows()


import networks
from ring_allreduce import RingAllreduce
from multitree_allreduce import MultiTreeAllreduce
from mxnettree_allreduce import MXNetTreeAllreduce

import math


'''
construct_allreduce() - construct an allreduce schedule
@args: arguments of the top simulation

return: an allreduce object
'''
def construct_allreduce(args):
    dimension = int(math.sqrt(args.num_hmcs))
    assert args.num_hmcs == dimension * dimension
    network = networks.Torus(args.num_hmcs, dimension)
    network.build_graph()

    if args.allreduce == 'multitree':
        allreduce = MultiTreeAllreduce(network)
    elif args.allreduce == 'mxnettree':
        allreduce = MXNetTreeAllreduce(network)
    elif args.allreduce == 'ring':
        allreduce = RingAllreduce(network)
    else:
        raise RuntimeError('Unknow allreduce schedule: ' + args.allreduce)

    return allreduce

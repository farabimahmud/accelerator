import argparse
import os
import sys
import math

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class HDRMAllreduce(Allreduce):
    def __init__(self, args, network):
        assert network.type == 'BiGraph'
        super().__init__(args, network)
        self.M = self.network.m
        self.N = self.network.n
        assert self.network.nodes == self.M * self.N
        self.rank_map = {}


    def compute_trees(self, kary=2, alternate=False, sort=True, verbose=False):
        if verbose:
            print('{}x{} network, {} sub-nodes'.format(self.M, self.N, self.network.nodes))

        base_rank_pair = (0, 1)
        base_rank_pairs = ([], [])

        if verbose:
            print('HDRM: start rank mapping ...')

        i = 0
        while i < self.N // 2 and i < self.network.nodes // 2:
            rank_pair = (base_rank_pair[0] + i * 2, base_rank_pair[1] + i * 2)

            num = i
            num_one_bits = 0
            while num:
                num_one_bits += num & 1
                num = num >> 1

            if num_one_bits % 2:
                base_rank_pairs[0].append(rank_pair[1])
                base_rank_pairs[1].append(rank_pair[0])
            else:
                base_rank_pairs[0].append(rank_pair[0])
                base_rank_pairs[1].append(rank_pair[1])

            i += 1

        if verbose:
            print('HDRM: base rank pairs: {}'.format(base_rank_pairs))

        if self.network.nodes <= self.N:
            return

        j = 0
        rank_pairs = []
        while j < self.M and j < self.network.nodes // self.N:
            rank_pair = ([ele + j * self.N for ele in base_rank_pairs[0]],
                         [ele + j * self.N for ele in base_rank_pairs[1]])

            num = j
            num_one_bits = 0
            while num:
                num_one_bits += num & 1
                num = num >> 1

            shift = (self.N // 2 - j) % (self.N // 2)
            if num_one_bits % 2:
                rank_pairs.append((rank_pair[1][shift:] + rank_pair[1][0:shift], rank_pair[0]))
            else:
                rank_pairs.append((rank_pair[0][shift:] + rank_pair[0][0:shift], rank_pair[1]))

            j += 1

        for v, pairs in enumerate(rank_pairs):
            for p, rank in enumerate(pairs[0] + pairs[1]):
                self.rank_map[rank] = p * self.M + v
            if verbose:
                print('virtual group {}: between {} and {}'.format(v, pairs[0], pairs[1]))

        # initialize empty trees
        self.trees = {}
        for node in range(self.network.nodes):
            self.trees[node] = []

        self.timesteps = int(math.log(self.network.nodes, 2))
        for step in range(self.timesteps):
            rank_distance = int(math.pow(2, self.timesteps - 1 - step))
            ranks = list(range(self.network.nodes))
            while len(ranks):
                rank1 = ranks.pop(0)
                rank2 = rank1 + rank_distance
                node1 = self.rank_map[rank1]
                node2 = self.rank_map[rank2]
                child1 = node2
                child2 = node1
                # NOTE: tree in ranks not in rank mapped nodes
                self.trees[rank1].append((rank2, rank1, step))
                self.trees[rank2].append((rank1, rank2, step))
                #self.trees[node1].append((child1, node1, step))
                #self.trees[node2].append((child2, node2, step))
                ranks.remove(rank2)


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate schedules for halving-doubling with rank mapping
    '''
    def generate_schedule(self, verbose=False):
        # compute parent-children dependency
        self.trees_parent = {}
        self.trees_children = {}
        for root in range(self.network.nodes):
            self.trees_parent[self.rank_map[root]] = {}
            self.trees_parent[self.rank_map[root]][self.rank_map[root]] = None
            self.trees_children[self.rank_map[root]] = {}
            for node in range(self.network.nodes):
                self.trees_children[self.rank_map[root]][self.rank_map[node]] = []
            for edge in self.trees[root]:
                child = self.rank_map[edge[0]]
                parent = self.rank_map[edge[1]]
                self.trees_parent[self.rank_map[root]][child] = parent
                self.trees_children[self.rank_map[root]][parent].append(child)
        for root in range(self.network.nodes):
            print('Tree for flow {} (mapped rank: {})'.format(self.rank_map[root], root))
            for node in range(self.network.nodes):
                if len(self.trees_children[self.rank_map[root]][node]) > 0:
                    print(' - node {}\'s children: {}'.format(node, self.trees_children[self.rank_map[root]][node]))

        self.reduce_scatter_schedule = {}
        self.all_gather_schedule = {}
        child = {}
        parent = {}
        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = []
            self.all_gather_schedule[node] = []
            child[node] = []
            parent[node] = None

        if verbose:
            print('Recursive halving reduce-scatter:')
        for step in range(self.timesteps):
            rank_distance = int(math.pow(2, step))
            num_data_copy = self.network.nodes // int(math.pow(2, step + 1))
            ranks = list(range(self.network.nodes))
            if verbose:
                print(' - step {}:'.format(step), end = ' ')
            while len(ranks):
                rank1 = ranks.pop(0)
                rank2 = rank1 + rank_distance
                node1 = self.rank_map[rank1]
                node2 = self.rank_map[rank2]
                parent1 = node2
                parent2 = node1
                self.reduce_scatter_schedule[node1].append({node2: ((parent1, 0), child[rank1], num_data_copy)})
                self.reduce_scatter_schedule[node2].append({node1: ((parent2, 0), child[rank2], num_data_copy)})
                # form the dependent flow-child for the next step
                child[rank1] = [(node1, node2)]
                child[rank2] = [(node2, node1)]
                if verbose:
                    print('({}, {})'.format(rank1, rank2), end = ' ')
                ranks.remove(rank2)
            if verbose:
                print('')

        for rank in range(self.network.nodes):
            self.reduce_scatter_schedule[self.rank_map[rank]].append({self.rank_map[rank]: ((None, None), child[rank], 0)})

        if verbose:
            print('Recursive doubling all-gather:')
        for step in range(self.timesteps):
            rank_distance = int(math.pow(2, self.timesteps - 1 - step))
            num_data_copy = self.network.nodes // int(math.pow(2, self.timesteps - step))
            ranks = list(range(self.network.nodes))
            if verbose:
                print(' - step {}:'.format(step), end = ' ')
            while len(ranks):
                rank1 = ranks.pop(0)
                rank2 = rank1 + rank_distance
                node1 = self.rank_map[rank1]
                node2 = self.rank_map[rank2]
                child1 = node2
                child2 = node1
                self.all_gather_schedule[node1].append({node1: ([(child1, 0)], parent[rank1], num_data_copy)})
                self.all_gather_schedule[node2].append({node2: ([(child2, 0)], parent[rank2], num_data_copy)})
                # form the dependent flow-parent for the next step
                parent[rank1] = (node2, node2)
                parent[rank2] = (node1, node1)
                if verbose:
                    print('({}, {})'.format(rank1, rank2), end = ' ')
                ranks.remove(rank2)
            if verbose:
                print('')

        if verbose:
            for node in range(self.network.nodes):
                print('Accelerator {}:'.format(node))
                print('  reduce-scatter schedule:')
                for timestep, schedule in enumerate(self.reduce_scatter_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
                print('  all-gather schedule:')
                for timestep, schedule in enumerate(self.all_gather_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
    # def generate_schedule(self, verbose=False)



def test(args):
    args.booksim_network = 'bigraph'
    network = construct_network(args)

    allreduce = HDRMAllreduce(args, network)
    allreduce.compute_trees(verbose=True)
    if args.gendotfile:
        allreduce.generate_trees_dotfile('hdrm.dot')
    allreduce.generate_schedule(verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=32, type=int,
                        help='number of nodes, default is 32')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--bigraph-m', default=4, type=int,
                        help='logical groups size (# sub-node per switch)')
    parser.add_argument('--bigraph-n', default=8, type=int,
                        help='# switches')

    args = parser.parse_args()

    test(args)

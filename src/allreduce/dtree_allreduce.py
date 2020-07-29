import argparse
import sys
import os
import numpy as np
from copy import deepcopy

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class DTreeAllreduce(Allreduce):
    def __init__(self, args, network):
        super().__init__(args, network)
        self.dtree = {}
        self.tree0_root = None
        self.tree1_root = None
        self.tree0 = []
        self.tree1 = []
        # node: ((tree0-parent, tree0-child0, tree0-child1),
        #        (tree1-parent, tree1-child0, tree1-child1))


    '''
    compute_trees() - computes double-binary trees
    @kary: not used, skip
    @alternate: not used, skip
    @sort: not used, skip
    @verbose: print detailed info of double-binary tree construction process
    '''
    def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False):
        for node in range(self.network.nodes):
            s0, d0_0, d0_1, s1, d1_0, d1_1 = self.get_dtree(node)
            if s0 == -1:
                self.tree0_root = node
            if s1 == -1:
                self.tree1_root = node
            self.dtree[node] = ((s0, d0_0, d0_1), (s1, d1_0, d1_1))
            if verbose:
                print('node {} - tree 0: [parent: {}, child0: {}, child1: {}]'.format(node, s0, d0_0, d0_1))
                print('        - tree 1: [parent: {}, child0: {}, child1: {}]'.format(s1, d1_0, d1_1))

        # tree nodes at levels for scheduling
        self.tree0.append([self.tree0_root])
        self.tree1.append([self.tree1_root])
        changed = True
        while changed:
            changed = False
            current_level0 = self.tree0[-1]
            current_level1 = self.tree1[-1]

            new_level0 = []
            new_level1 = []
            for node in current_level0:
                connection0 = self.dtree[node][0]
                child0 = connection0[1]
                child1 = connection0[2]
                if child0 != -1:
                    new_level0.append(child0)
                if child1 != -1:
                    new_level0.append(child1)
            for node in current_level1:
                connection1 = self.dtree[node][1]
                child0 = connection1[1]
                child1 = connection1[2]
                if child0 != -1:
                    new_level1.append(child0)
                if child1 != -1:
                    new_level1.append(child1)

            if new_level0:
                assert new_level1
                changed = True
                self.tree0.append(new_level0)
                self.tree1.append(new_level1)
            else:
                assert not new_level1

        assert len(self.tree0) == len(self.tree1)

        # compute timesteps - the longest accumulative path from root to leaves
        self.timesteps = 0
        latency = np.zeros(self.network.nodes, dtype=int)
        queue = [self.tree0_root]
        while len(queue) > 0:
            node = queue.pop(0)
            parent, child0, child1 = self.dtree[node][0]
            if parent != -1:
                latency[node] = latency[parent] + self.network.distance(parent, node)
            if child0 != -1:
                queue.append(child0)
            if child1 != -1:
                queue.append(child1)

        self.timesteps = max(latency)

        latency = np.zeros(self.network.nodes, dtype=int)
        queue = [self.tree1_root]
        while len(queue) > 0:
            node = queue.pop(0)
            parent, child0, child1 = self.dtree[node][1]
            if parent != -1:
                latency[node] = latency[parent] + self.network.distance(parent, node)
            if child0 != -1:
                queue.append(child0)
            if child1 != -1:
                queue.append(child1)

        if self.timesteps < max(latency):
            self.timesteps = max(latency)

        # doube-binary tree only has two trees, pipelined
        # TODO: consider link conflicts
        self.timesteps += (self.network.nodes // 2) - 1

        if verbose:
            print('Total timestep: {}'.format(self.timesteps))
    # end of compute_trees()


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from double
           binary trees
    '''
    def generate_schedule(self, verbose=False):
        #compute parent-children dependency
        self.trees_parent = {}
        self.trees_children = {}
        for root in range(self.network.nodes):
            self.trees_parent[root] = {}
            self.trees_children[root] = {}
            for node in range(self.network.nodes):
                self.trees_children[root][node] = []
                if root < self.network.nodes // 2:
                    parent, child0, child1 = self.dtree[node][0]
                else:
                    parent, child0, child1 = self.dtree[node][1]
                if parent == -1:
                    assert node == self.tree0_root or node == self.tree1_root
                    self.trees_parent[root][node] = None
                else:
                    self.trees_parent[root][node] = parent
                if child0 != -1:
                    self.trees_children[root][node].append(child0)
                if child1 != -1:
                    self.trees_children[root][node].append(child1)

        # initialize the schedules
        self.reduce_scatter_schedule = {}
        self.all_gather_schedule = {}

        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = []
            self.all_gather_schedule[node] = []

        height = len(self.tree0)
        for l in range(height):
            current_level0 = self.tree0[l]
            current_level1 = self.tree1[l]
            for node in current_level0:
                parent, child0, child1 = self.dtree[node][0]
                if parent == -1:
                    parent = None
                children = []
                if child0 != -1:
                    children.append((child0, 0))
                if child1 != -1:
                    children.append((child1, 0))
                if children:
                    #self.all_gather_schedule[node].append({self.tree0_root: (children, parent)})
                    for sub_flow in range(self.network.nodes // 2):
                        if parent == None:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, None, 1, 1))}) # TODO: data copy may be more than 1
                        else:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, (sub_flow, parent), 1, 1))})
            for node in current_level1:
                parent, child0, child1 = self.dtree[node][1]
                if parent == -1:
                    parent = None
                children = []
                if child0 != -1:
                    children.append((child0, 0))
                if child1 != -1:
                    children.append((child1, 0))
                if children:
                    #self.all_gather_schedule[node].append({self.tree1_root: (children, parent)})
                    for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                        if parent == None:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, None, 1, 2))})
                        else:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, (sub_flow, parent), 1, 2))})

            current_reversed_level0 = self.tree0[height - l - 1]
            current_reversed_level1 = self.tree1[height - l - 1]
            for node in current_reversed_level0:
                parent, child0, child1 = self.dtree[node][0]
                if parent == -1:
                    parent = (None, None)
                    timestep = 1
                else:
                    parent = (parent, 0)
                    timestep = 0
                children = []
                if child0 != -1:
                    children.append(child0)
                if child1 != -1:
                    children.append(child1)
                #self.reduce_scatter_schedule[node].append({self.tree0_root: (parent, children)})
                for sub_flow in range(self.network.nodes // 2):
                    if children:
                        flow_children = [(sub_flow, child) for child in children]
                    else:
                        flow_children = []
                    self.reduce_scatter_schedule[node].append({sub_flow: deepcopy((parent, flow_children, 1, timestep))})
            for node in current_reversed_level1:
                parent, child0, child1 = self.dtree[node][1]
                if parent == -1:
                    parent = (None, None)
                    timestep = 1
                else:
                    parent = (parent, 0)
                    timestep = 0
                children = []
                if child0 != -1:
                    children.append(child0)
                if child1 != -1:
                    children.append(child1)
                #self.reduce_scatter_schedule[node].append({self.tree1_root: (parent, children)})
                for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                    if children:
                        flow_children = [(sub_flow, child) for child in children]
                    else:
                        flow_children = []
                    self.reduce_scatter_schedule[node].append({sub_flow: deepcopy((parent, flow_children, 1, timestep))})

        if verbose:
            for node in range(self.network.nodes):
                print('Accelerator {}:'.format(node))
                print('  reduce schedule:')
                for timestep, schedule in enumerate(self.reduce_scatter_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
                print('  gather schedule:')
                for timestep, schedule in enumerate(self.all_gather_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
    # end of generate_schedule()


    '''
    get_btree() - get the parent and both children for first tree
    @rank: the node ID

    desc - Ported from NCCL.
           Get parent and both children from a binary tree that alternates
           leaves and internal ndoes. Assuming the root is 0, which
           conveniently builds a tree on powers of two (because we have pow2-1
           ranks), which is easy to manipulate bits.
           Find the first non-zero bit, then:
           Find the parent:
             xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
             xx11[0] -> xx10[0] (3,7,11 below)
           Find the children :
             xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
             xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)

           Illustration :
           0---------------8
                    ______/ \______
                   4               12
                 /   \            /  \
               2       6       10     \
              / \     / \     /  \     \
             1   3   5   7   9   11    13

    return: parent, child0, child1 of the node
    '''
    def get_btree(self, rank):
        nranks = self.network.nodes
        bit = 1
        while bit < nranks:
            if bit & rank:
                break
            bit <<= 1

        if rank == 0:
            u = -1
            if nranks > 1:
                d0 = bit >> 1
            else:
                d0 = -1
            d1 = -1
            return (u, d0, d1)

        up = (rank ^ bit) | (bit << 1)
        if up >= nranks:
            up = rank ^ bit
        u = up

        lowbit = bit >> 1
        # down0 is always within bounds
        if lowbit == 0:
            down0 = -1
            down1 = -1
        else:
            down0 = rank - lowbit
            down1 = rank + lowbit
        # make sure down1 is within bounds
        while down1 >= nranks:
            if lowbit == 0:
                down1 = -1
            else:
                down1 = rank + lowbit
                lowbit >>= 1

        d0 = down0
        d1 = down1

        return (u, d0, d1)


    '''
    get_dtree() - get the parents and childrens of double-binary tree
    @rank: the node ID

    desc - Ported from NCCL.
           Build a double binary tree by firstly build one tree, then build a
           mirror tree if nranks is odd:
                         8---------0---------5
                  ______/ \______      _____/ \______
                 4               12   1              9
               /   \            /      \           /   \
             2       6       10          3       7      10
            / \     / \     /  \        / \     / \    /  \
           1   3   5   7   9   11      2   4   6   8  11  12
           or shift the first tree by one rank if nranks is even:
                         8---------0--------------9
                  ______/ \                ______/ \
                 4         \              5         \
               /   \        \           /   \        \
             2       6       10       3       7       11
            / \     / \     /  \     / \     / \     /  \
           1   3   5   7   9   11   2   4   6   8   10   1

    return:
    @(parent0, child0_0, child0_1, parent1, child1_0, child1_1)
    tree0: parent0, child0_0, child0_1
    tree1: parent1, child1_0, child1_1
    '''
    def get_dtree(self, rank):
        nranks = self.network.nodes
        # first tree .. use a binary tree
        s0, d0_0, d0_1 = self.get_btree(rank)
        # second tree ... mirror or shift
        s1 = -1
        d1_0 = -1
        d1_1 = -1
        if nranks % 2 == 0:
            # shift
            shiftrank = (rank - 1 + nranks) % nranks
            u, d0, d1 = self.get_btree(shiftrank)
            if u != -1:
                s1 = (u + 1) % nranks
            if d0 != -1:
                d1_0 = (d0 + 1) % nranks
            if d1 != -1:
                d1_1 = (d1 + 1) % nranks
        else:
            # mirror
            u, d0, d1 = self.get_btree(nranks - 1 - rank)
            if u != -1:
                s1 = nranks - 1 - u
            if d0 != -1:
                d1_0 = nranks - 1 - d0
            if d1 != -1:
                d1_1 = nranks - 1 - d1

        return (s0, d0_0, d0_1, s1, d1_0, d1_1)


def test(args):
    args.num_hmcs = int(args.dimension * args.dimension)
    network = construct_network(args)

    allreduce = DTreeAllreduce(args, network)
    allreduce.compute_trees(verbose=False)
    allreduce.generate_schedule(verbose=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=4, type=int,
                        help='network dimension, default is 4')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--booksim-network', default='torus',
                        help='network topology (torus | mesh), default is torus')

    args = parser.parse_args()

    test(args)

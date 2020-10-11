import argparse
import sys
import os
import numpy as np
import math
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
        self.tree0 = [] # node: ((tree0-parent, tree0-child0, tree0-child1),
        self.tree1 = [] #        (tree1-parent, tree1-child0, tree1-child1))
        self.tree_edge_colors = {}
        self.tree_edge_colors[0] = [-1] * self.network.nodes
        self.tree_edge_colors[1] = [-1] * self.network.nodes
        self.black = 1
        self.red = 0
        self.colors = ['Red', 'Black']


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
                print('node{:3d} - tree 0: [parent: {}, child0: {}, child1: {}]'.format(node, s0, d0_0, d0_1))
                print('        - tree 1: [parent: {}, child0: {}, child1: {}]'.format(s1, d1_0, d1_1))

        # tree nodes at levels for scheduling
        self.tree0.append([self.tree0_root])
        self.tree1.append([self.tree1_root])

        tree0 = set(range(0, self.network.nodes))
        tree0.remove(self.tree0_root)
        tree1 = set(range(0, self.network.nodes))
        tree1.remove(self.tree1_root)

        depth = math.ceil(math.log2(self.network.nodes))

        for height in range(depth-1, 0, -1):
            new_level0 = []
            new_level1 = []

            nodes = list(tree0)
            for node in nodes:
                h = 0
                node_copy = node
                while node_copy & 1 == 0:
                    h += 1
                    node_copy >>= 1
                if h == height:
                    new_level0.append(node)
                    tree0.remove(node)

            nodes = list(tree1)
            if self.network.nodes % 2 == 0:
                for node in nodes:
                    h = 0
                    mirrored_node = self.network.nodes - 1 - node
                    while mirrored_node & 1 == 0:
                        h += 1
                        mirrored_node >>= 1
                    if h == height:
                        new_level1.append(node)
                        tree1.remove(node)
            else:
                for node in nodes:
                    h = 0
                    shifted_node = (self.network.nodes + 1 + node) % self.network.nodes
                    while shifted_node & 1 == 0:
                        h += 1
                        shifted_node >>= 1
                    if h == height:
                        new_level1.append(node)
                        tree1.remove(node)

            if new_level0:
                assert new_level1
                self.tree0.append(new_level0)
                self.tree1.append(new_level1)
            else:
                assert not new_level1

        # add leaf nodes
        self.tree0.append([])
        self.tree1.append([])
        for node in range(self.network.nodes):
            if node != self.tree0_root and node % 2 == 1:
                self.tree0[-1].append(node)
            if node != self.tree1_root and node % 2 == 0:
                self.tree1[-1].append(node)

        assert len(self.tree0) == len(self.tree1)

        # color edges
        self.color_edges(verbose=self.args.verbose)

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


    def compute_parent_children_dependency(self):
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
    # end of compute_parent_children_denpendency()



    '''
    generate_schedule()
    @verbose: print the generated pipelined schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from pipelined
           double binary trees
    '''
    def generate_schedule(self, verbose=False):
        self.compute_parent_children_dependency()

        tree0_reduce_schedule = []
        tree0_broadcast_schedule = []
        tree1_reduce_schedule = []
        tree1_broadcast_schedule = []

        height = len(self.tree0)

        # compute reduce schedule
        color = self.black
        for l in range(height):
            tree0_level = set(self.tree0[height - l - 1])

            while tree0_level:
                tree0_reduce_schedule.append([])

                level = list(tree0_level)
                for node in level:
                    if node == self.tree0_root:
                        tree0_level.remove(node)
                        break
                    if self.tree_edge_colors[0][node] == color:
                        tree0_reduce_schedule[-1].append({'from': node, 'to': self.dtree[node][0][0]})
                        tree0_level.remove(node)

                color = (color + 1) % 2

        color = self.black
        for l in range(height):
            tree1_level = set(self.tree1[height - l - 1])

            while tree1_level:
                tree1_reduce_schedule.append([])

                level = list(tree1_level)
                for node in level:
                    if node == self.tree1_root:
                        tree1_level.remove(node)
                        tree1_reduce_schedule.pop()
                        break
                    if self.tree_edge_colors[1][node] == color:
                        tree1_reduce_schedule[-1].append({'from': node, 'to': self.dtree[node][1][0]})
                        tree1_level.remove(node)

                color = (color + 1) % 2

        assert len(tree0_reduce_schedule) == len(tree1_reduce_schedule)

        # compute broadcast schedule
        ready_nodes = deepcopy(self.tree0[1]) # can receive from root
        color = self.black
        while ready_nodes:
            tree0_broadcast_schedule.append([])

            nodes = deepcopy(ready_nodes)
            for node in nodes:
                if self.tree_edge_colors[0][node] == color:
                    parent, child0, child1 = self.dtree[node][0]
                    tree0_broadcast_schedule[-1].append({'from': parent, 'to': node})
                    ready_nodes.remove(node)
                    if child0 != -1:
                        ready_nodes.append(child0)
                    if child1 != -1:
                        ready_nodes.append(child1)

            color = (color + 1) % 2
        tree0_broadcast_schedule.append([])

        ready_nodes = deepcopy(self.tree1[1]) # can receive from root
        color = self.black
        while ready_nodes:
            tree1_broadcast_schedule.append([])

            nodes = deepcopy(ready_nodes)
            for node in nodes:
                if self.tree_edge_colors[1][node] == color:
                    parent, child0, child1 = self.dtree[node][1]
                    tree1_broadcast_schedule[-1].append({'from': parent, 'to': node})
                    ready_nodes.remove(node)
                    if child0 != -1:
                        ready_nodes.append(child0)
                    if child1 != -1:
                        ready_nodes.append(child1)

            color = (color + 1) % 2

        assert len(tree0_broadcast_schedule) == len(tree1_broadcast_schedule)

        # generate per-node reduce schedules
        pipe_stages = len(tree0_reduce_schedule)
        assert pipe_stages % 2 == 0
        timesteps = 2 * (self.network.nodes // 2 - 1) + pipe_stages

        # initialize the schedules
        self.reduce_scatter_schedule = {}
        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = [{} for i in range(timesteps + 1)]

        # generate reduce schedule for tree 0
        for step in range(pipe_stages):
            for comm in tree0_reduce_schedule[step]:
                from_node = comm['from']
                to_node = comm['to']
                parent, child0, child1 = self.dtree[from_node][0]
                assert parent == to_node
                parent = (parent, 0)
                children = []
                if child0 != -1:
                    children.append(child0)
                if child1 != -1:
                    children.append(child1)

                for sub_flow in range(self.network.nodes // 2):
                    pipe_step = 2 * sub_flow + step
                    if children:
                        flow_children = [(sub_flow, child) for child in children]
                    else:
                        flow_children = []
                    self.reduce_scatter_schedule[from_node][pipe_step][sub_flow] = (parent, flow_children, 1, pipe_step)

        # generate reduce schedule for tree 1
        for step in range(pipe_stages):
            for comm in tree1_reduce_schedule[step]:
                from_node = comm['from']
                to_node = comm['to']
                parent, child0, child1 = self.dtree[from_node][1]
                assert parent == to_node
                parent = (parent, 0)
                children = []
                if child0 != -1:
                    children.append(child0)
                if child1 != -1:
                    children.append(child1)

                for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                    pipe_step = 2 * (sub_flow - self.network.nodes // 2) + step
                    if children:
                        flow_children = [(sub_flow, child) for child in children]
                    else:
                        flow_children = []
                    self.reduce_scatter_schedule[from_node][pipe_step][sub_flow] = (parent, flow_children, 1, pipe_step)

        for node in range(self.network.nodes):
            children = []
            parent, child0, child1 = self.dtree[node][0]
            if child0 != -1:
                for sub_flow in range(self.network.nodes // 2):
                    children.append((sub_flow, child0))
            if child1 != -1:
                for sub_flow in range(self.network.nodes // 2):
                    children.append((sub_flow, child1))
            parent, child0, child1 = self.dtree[node][1]
            if child0 != -1:
                for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                    children.append((sub_flow, child0))
            if child1 != -1:
                for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                    children.append((sub_flow, child1))
            self.reduce_scatter_schedule[node][timesteps][node] = ((None, None), children, 0, timesteps)

        # generate per-node broadcast schedules
        reduce_timesteps = timesteps
        pipe_stages = len(tree1_broadcast_schedule)
        assert pipe_stages % 2 == 0
        timesteps = 2 * (self.network.nodes // 2 - 1) + pipe_stages

        # initialize the schedules
        self.all_gather_schedule = {}
        for node in range(self.network.nodes):
            self.all_gather_schedule[node] = [{} for i in range(timesteps)]

        # generate broadcast schedule for tree 0
        for step in range(pipe_stages):
            for comm in tree0_broadcast_schedule[step]:
                from_node = comm['from']
                to_node = comm['to']
                parent, child0, child1 = self.dtree[from_node][0]
                if parent == -1:
                    parent = None

                for sub_flow in range(self.network.nodes // 2):
                    pipe_step = 2 * sub_flow + step
                    if parent == None:
                        self.all_gather_schedule[from_node][pipe_step][sub_flow] = ([(to_node, 0)], None, 1, pipe_step + timesteps + 1)
                    else:
                        self.all_gather_schedule[from_node][pipe_step][sub_flow] = ([(to_node, 0)], (sub_flow, parent), 1, pipe_step + timesteps + 1)

        # generate broadcast schedule for tree 1
        for step in range(pipe_stages):
            for comm in tree1_broadcast_schedule[step]:
                from_node = comm['from']
                to_node = comm['to']
                parent, child0, child1 = self.dtree[from_node][1]
                if parent == -1:
                    parent = None
                for sub_flow in range(self.network.nodes // 2, self.network.nodes):
                    pipe_step = 2 * (sub_flow - self.network.nodes // 2) + step
                    if parent == None:
                        self.all_gather_schedule[from_node][pipe_step][sub_flow] = ([(to_node, 0)], None, 1, pipe_step + timesteps + 1)
                    else:
                        self.all_gather_schedule[from_node][pipe_step][sub_flow] = ([(to_node, 0)], (sub_flow, parent), 1, pipe_step + timesteps + 1)

        # prune schedules, TODO: may check estimate-lockstep (replace {} with None)
        for node in range(self.network.nodes):
            for s, schedule in enumerate(self.reduce_scatter_schedule[node]):
                if not schedule:
                    self.reduce_scatter_schedule[node][s] = None
            for s, schedule in enumerate(self.all_gather_schedule[node]):
                if not schedule:
                    self.all_gather_schedule[node][s] = None
            #self.reduce_scatter_schedule[node] = list(filter(None, self.reduce_scatter_schedule[node]))
            #self.all_gather_schedule[node] = list(filter(None, self.all_gather_schedule[node]))

        # print tree and schedules
        if verbose:
            print('Tree 0 reduce schedule:')
            for step in range(len(tree0_reduce_schedule)):
                communicatioins = []
                for comm in tree0_reduce_schedule[step]:
                    communicatioins.append('{}->{}'.format(comm['from'], comm['to']))
                print(' Step {}: {}'.format(step, communicatioins))
            print('Tree 1 reduce schedule:')
            for step in range(len(tree1_reduce_schedule)):
                communicatioins = []
                for comm in tree1_reduce_schedule[step]:
                    communicatioins.append('{}->{}'.format(comm['from'], comm['to']))
                print(' Step {}: {}'.format(step, communicatioins))

            print('Tree 0 broadcast schedule:')
            for step in range(len(tree0_broadcast_schedule)):
                communicatioins = []
                for comm in tree0_broadcast_schedule[step]:
                    communicatioins.append('{}->{}'.format(comm['from'], comm['to']))
                print(' Step {}: {}'.format(step, communicatioins))
            print('Tree 1 broadcast schedule:')
            for step in range(len(tree1_broadcast_schedule)):
                communicatioins = []
                for comm in tree1_broadcast_schedule[step]:
                    communicatioins.append('{}->{}'.format(comm['from'], comm['to']))
                print(' Step {}: {}'.format(step, communicatioins))

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
    generate_schedule_no_pipelined()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from double
           binary trees
    '''
    def generate_schedule_no_pipelined(self, verbose=False):
        self.compute_parent_children_dependency()

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
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, None, self.network.nodes // 2, 1))}) # TODO: data copy may be more than 1
                        else:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, (sub_flow, parent), self.network.nodes // 2, 1))})
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
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, None, self.network.nodes // 2, 2))})
                        else:
                            self.all_gather_schedule[node].append({sub_flow: deepcopy((children, (sub_flow, parent), self.network.nodes // 2, 2))})

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
                    self.reduce_scatter_schedule[node].append({sub_flow: deepcopy((parent, flow_children, self.network.nodes // 2, timestep))})
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
                    self.reduce_scatter_schedule[node].append({sub_flow: deepcopy((parent, flow_children, self.network.nodes // 2, timestep))})

        if verbose:
            for node in range(self.network.nodes):
                print('Accelerator {}:'.format(node))
                print('  reduce schedule:')
                for timestep, schedule in enumerate(self.reduce_scatter_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
                print('  gather schedule:')
                for timestep, schedule in enumerate(self.all_gather_schedule[node]):
                    print('    timestep {}: {}'.format(timestep, schedule))
    # end of generate_schedule_no_pipelined()


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
    # end of get_btree()


    '''
    get_dtree() - get the parents and childrens of double-binary tree
    @rank: the node ID

    desc - Ported from NCCL with some modification.
           Build a double binary tree by firstly build one tree, then build a
           shifted tree if nranks is odd:
                         8---------0       12-------7
                  ______/ \______             _____/ \______
                 4               12          3             11
               /   \            /          /   \          /
             2       6       10          1       5       9
            / \     / \     /  \        / \     / \     / \
           1   3   5   7   9   11      0   2   4   6   8  10
           or mirror the first tree by one rank if nranks is even:
                         8---------0  11---------3
                  ______/ \                     / \______
                 4         \                   /         7
               /   \        \                 /        /   \
             2       6       10              1       5       9
            / \     / \     /  \            / \     / \     /  \
           1   3   5   7   9   11          0   2   4   6   8   10

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
        if nranks % 2 == 1:
            # shift
            shiftrank = (rank + 1 + nranks) % nranks
            u, d0, d1 = self.get_btree(shiftrank)
            if u != -1:
                s1 = (u - 1) % nranks
            if d0 != -1:
                d1_0 = (d0 - 1) % nranks
            if d1 != -1:
                d1_1 = (d1 - 1) % nranks
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
    # end of get_dtree()


    '''
    in_edge_color() - get the color for an incoming edge
    @rank

    return: color of the incoming edge of the node
    '''
    def in_edge_color(self, rank, verbose=False):
        #TODO: currently support even number of ranks,
        #      odd number of ranks may not be correct

        # fake incoming edge color for roots
        if rank == self.tree0_root:
            return 0
        elif rank == self.tree1_root:
            return 1

        tree = rank % 2
        parent = self.dtree[rank][tree][0]
        parent_in_edge_color = self.tree_edge_colors[tree][parent]
        #print('     node {}\'s parent is node {} (tree {})'.format(rank, parent, tree))
        if parent_in_edge_color == -1:
            parent_in_edge_color = self.in_edge_color(parent)
        color = parent_in_edge_color ^ ((self.network.nodes // 2) % 2 == 1)
        if tree == 0:
            return color ^ (parent < rank)
        else:
            return color ^ (parent > rank)
    # end of in_edge_color()


    '''
    color_edges() - color the edges of the two trees

    '''
    def color_edges(self, verbose=False):
        # tree 0 internal nodes
        if verbose:
            print('coloring tree 0 internal nodes\' incoming edges')
        for rank in range(0, self.network.nodes, 2):
            #if verbose:
            #    print(' - color incoming edge for node {}'.format(rank))
            self.tree_edge_colors[0][rank] = self.in_edge_color(rank, verbose)
            if verbose:
                print(' - node {}\'s incoming edge color: {}'.format(rank, self.colors[self.tree_edge_colors[0][rank]]))

        # tree 1 internal nodes
        if verbose:
            print('coloring tree 1 internal nodes\' incoming edges')
        for rank in range(1, self.network.nodes, 2):
            #if verbose:
            #    print(' - color incoming edge for node {}'.format(rank))
            self.tree_edge_colors[1][rank] = self.in_edge_color(rank, verbose)
            if verbose:
                print(' - node {}\'s incoming edge color: {}'.format(rank, self.colors[self.tree_edge_colors[1][rank]]))

        # tree 0 leaf nodes
        for rank in range(1, self.network.nodes, 2):
            self.tree_edge_colors[0][rank] = (self.tree_edge_colors[1][rank] + 1) % 2

        # tree 1 leaf nodes
        for rank in range(0, self.network.nodes, 2):
            self.tree_edge_colors[1][rank] = (self.tree_edge_colors[0][rank] + 1) % 2

        # verification
        for rank in range(self.network.nodes):
            assert self.tree_edge_colors[0][rank] != self.tree_edge_colors[1][rank]

        print('Tree 0 edge colors:')
        for l, level in enumerate(self.tree0):
            for node in level:
                print(' - node {} incoming edge color: {}'.format(node, self.colors[self.tree_edge_colors[0][node]]))
        print('Tree 1 edge colors:')
        for l, level in enumerate(self.tree1):
            for node in level:
                print(' - node {} incoming edge color: {}'.format(node, self.colors[self.tree_edge_colors[1][node]]))
    # end of color_edges()


    '''
    generate_dtree_dotfile() - generate dot file for double binary tree
    '''
    def generate_dtree_dotfile(self):
        # color palette for ploting nodes of different tree levels
        colors = ['#ffffff', '#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7',
                '#df65b0', '#e7298a', '#ce1256', '#980043', '#67001f']
        black = '#000000'
        red = '#ff0000'

        tree = 'digraph tree {\n'
        tree += '  rankdir = TB;\n'
        tree += '  subgraph {\n'

        for root in [0, 1]:
            tree += '    /* tree {} */\n'.format(root)
            for node in range(self.network.nodes):
                parent = '"T{}-{}"'.format(root, node)
                child0 = self.dtree[node][root][1]
                child1 = self.dtree[node][root][2]
                if child0 != -1:
                    if self.tree_edge_colors[root][child0] == 0:
                        color = red
                    else:
                        color = black
                    child0 = '"T{}-{}"'.format(root, child0)
                    tree += ''.join('   {} -> {} [ dir=none, color="{}" ]\n'.format(parent, child0, color))
                if child1 != -1:
                    if self.tree_edge_colors[root][child1] == 0:
                        color = red
                    else:
                        color = black
                    child1 = '"T{}-{}"'.format(root, child1)
                    tree += ''.join('   {} -> {} [ dir=none, color="{}" ]\n'.format(parent, child1, color))

        tree += '    // note that rank is used in the subgraph\n'
        for l in range(len(self.tree0)):
            level = '    {rank = same;'
            for node in self.tree0[l]:
                level += ' "T0-{}";'.format(node)
            for node in self.tree1[l]:
                level += ' "T1-{}";'.format(node)
            level += '}\n'
            tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for l in range(len(self.tree0)):
            for node in self.tree0[l]:
                nodename = '"T0-{}"'.format(node)
                tree += ''.join(style.format(nodename, colors[l % len(colors)]))
            for node in self.tree1[l]:
                nodename = '"T1-{}"'.format(node)
                tree += ''.join(style.format(nodename, colors[l % len(colors)]))


        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open('dtree{}.dot'.format(self.network.nodes), 'w')
        f.write(tree)
        f.close()
    # end of generate_dtree_dotfile()

def test(args):
    network = construct_network(args)

    allreduce = DTreeAllreduce(args, network)
    allreduce.compute_trees(verbose=args.verbose)
    if args.gendotfile:
        allreduce.generate_dtree_dotfile()
    allreduce.generate_schedule_no_pipelined(verbose=args.verbose)
    allreduce.generate_schedule(verbose=args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=16, type=int,
                        help='number of nodes, default is 16')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary tree, default is 2 (binary)')
    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='detailed print')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--booksim-network', default='torus',
                        help='network topology (torus | mesh | dgx2), default is torus')

    args = parser.parse_args()

    test(args)

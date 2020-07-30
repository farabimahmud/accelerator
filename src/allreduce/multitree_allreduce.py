import argparse
import numpy as np
from copy import deepcopy
import sys
import os

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class MultiTreeAllreduce(Allreduce):
    def __init__(self, args, network):
        super().__init__(args, network)


    '''
    compute_trees() - computes allreduce spanning trees for the given network
    @kary: build kary-trees
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @sort: Whether sort the trees for link allocation based on conflicts from
           last allocation iteration
    @verbose: print detailed info of tree construction process
    '''
    def compute_trees(self, kary, alternate=True, sort=False, verbose=False):
        assert kary > 1

        # initialize empty trees
        self.trees = {}
        tree_nodes = {}
        for node in range(self.network.nodes):
            self.trees[node] = []
            tree_nodes[node] = [node]
            if verbose:
                print('initialized tree {}: {}'.format(node, tree_nodes[node]))

        # tree construction
        num_trees = 0
        self.timesteps = 0

        # sort the roots based on link conflicts during allocation
        sorted_roots = list(range(self.network.nodes))
        conflicts = [0] * self.network.nodes

        constructed_trees = []

        while num_trees < self.network.nodes:
            if verbose:
                print('timestep {}'.format(self.timesteps))

            node_to_switch = deepcopy(self.network.node_to_switch)
            switch_to_switch = deepcopy(self.network.switch_to_switch)
            switch_to_node = deepcopy(self.network.switch_to_node)
            last_tree_nodes = deepcopy(tree_nodes)

            # alternating the link allocation every time for each tree
            if alternate:

                changed = True

                turns = 0
                while changed:
                    changed = False

                    root = sorted_roots[turns % self.network.nodes]

                    if len(tree_nodes[root]) < self.network.nodes and verbose:
                        p = (turns // self.network.nodes) % len(tree_nodes[root])
                        parent = tree_nodes[root][p]
                        print('turns: {}, root: {}, p: {}, parent: {}'.format(turns, root, p, parent))

                    if len(tree_nodes[root]) < self.network.nodes:
                        for parent in last_tree_nodes[root]:
                            if parent not in node_to_switch.keys():
                                continue
                            switch = node_to_switch[parent][0]
                            # first check nodes on same switch
                            if switch in switch_to_node.keys():
                                children = deepcopy(switch_to_node[switch])
                                for child in children:
                                    if child not in tree_nodes[root]:
                                        if verbose:
                                            print(' -- add node {} to tree {} (connected to parent {} on same switch {})'.format(child, root, parent, switch))
                                            print('    before: {}'.format(self.trees[root]))
                                        node_to_switch[parent] = (switch, node_to_switch[parent][1] - 1)
                                        if node_to_switch[parent][1] == 0:
                                            node_to_switch.pop(parent, None)
                                        switch_to_node[switch].remove(child)
                                        if not switch_to_node[switch]:
                                            switch_to_node.pop(switch, None)
                                        tree_nodes[root].append(child)
                                        self.trees[root].append((child, parent, self.timesteps))
                                        if verbose:
                                            print('    after : {}'.format(self.trees[root]))
                                            print('    tree nodes: {}'.format(tree_nodes[root]))
                                        changed = True
                                        break
                                    else:
                                        conflicts[root] += 1
                            # check one hop distant nodes
                            if changed == False:
                                neighbor_switches = deepcopy(switch_to_switch[switch])
                                for neighbor_sw in neighbor_switches:
                                    if neighbor_sw in switch_to_node.keys():
                                        children = deepcopy(switch_to_node[neighbor_sw])
                                        for child in children:
                                            if child not in tree_nodes[root]:
                                                if verbose:
                                                    print(' -- add node {} ( with switch {}) to tree {} (connected to parent {} on neighbor switch {})'.format(child, neighbor_sw, root, parent, switch))
                                                    print('    before: {}'.format(self.trees[root]))
                                                node_to_switch[parent] = (switch, node_to_switch[parent][1] - 1)
                                                if node_to_switch[parent][1] == 0:
                                                    node_to_switch.pop(parent, None)
                                                switch_to_node[neighbor_sw].remove(child)
                                                if not switch_to_node[neighbor_sw]:
                                                    switch_to_node.pop(neighbor_sw, None)
                                                switch_to_switch[switch].remove(neighbor_sw)
                                                tree_nodes[root].append(child)
                                                self.trees[root].append((child, parent, self.timesteps))
                                                if verbose:
                                                    print('    after : {}'.format(self.trees[root]))
                                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                                changed = True
                                                break
                                            else:
                                                conflicts[root] += 1
                                        if changed:
                                            break
                            if changed:
                                break

                    turns += 1

                    if len(tree_nodes[root]) == self.network.nodes and root not in constructed_trees:
                        constructed_trees.append(root)
                        num_trees += 1
                        if verbose:
                            print('timestep {} - tree {} constructed: {}'.format(self.timesteps, root, self.trees[root]))
                        if num_trees == self.network.nodes:
                            break

                    if turns % self.network.nodes != 0:
                        changed = True
                    else:
                        if sort:
                            #print('before sorting: {}'.format(sorted_roots))
                            #print('conflicts: {}'.format(conflicts))
                            sorted_roots = list(range(self.network.nodes))
                            sorted_roots = [root for _ , root in sorted(zip(self.network.priority, sorted_roots), reverse=True)]
                            #sorted_roots = [root for _ , root in sorted(zip(conflicts, sorted_roots), reverse=True)]
                            conflicts = [0] * self.network.nodes
                            #print('after sorting: {}'.format(sorted_roots))

            else:   # else case: allocating links for one tree as much as possble
                for root in range(self.network.nodes):
                    if len(tree_nodes[root]) == self.network.nodes:
                        continue
                    current_tree_nodes = deepcopy(tree_nodes[root])
                    for p, parent in enumerate(current_tree_nodes):
                        children = deepcopy(from_nodes[parent])
                        num_new_children = 0
                        for child in children:
                            if num_new_children == kary - 1:
                                break
                            if verbose:
                                print(' child {}'.format(child))
                            if child not in tree_nodes[root]:
                                if verbose:
                                    print(' -- add node {} to tree {}'.format(child, root))
                                    print('    before: {}'.format(self.trees[root]))
                                from_nodes[parent].remove(child)
                                tree_nodes[root].append(child)
                                self.trees[root].append((child, parent, self.timesteps))
                                if verbose:
                                    print('    after : {}'.format(self.trees[root]))
                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                num_new_children += 1
                    if len(tree_nodes[root]) == self.network.nodes:
                        num_trees += 1
                        if verbose:
                            print('timestep {} - tree {} constructed: {}'.format(self.timesteps, root, self.trees[root]))
                    if verbose:
                        print('  tree {}: {}'.format(root, self.trees[root]))

            self.timesteps += 1

        # verify that there is no link conflicts
        for root in range(self.network.nodes):
            for i in range(root + 1, self.network.nodes):
                intersection = set(self.trees[root]) & set(self.trees[i])
                if len(intersection) != 0:
                    print('tree {} and tree {} have link conflicts {}'.format(root, i, intersection))
                    print('tree {}: {}'.format(root, self.trees[root]))
                    print('tree {}: {}'.format(i, self.trees[i]))
                    exit()

        if verbose:
            print('Total timesteps for network size of {}: {}'.format(self.network.nodes, self.timesteps))
    # def compute_trees(self, kary, alternate=False, sort=True, verbose=False)


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from trees
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
        reduce_scatter_schedule = {}
        all_gather_schedule = {}

        # construct schedules for each node from trees
        for node in range(self.network.nodes):
            reduce_scatter_schedule[node] = {}
            all_gather_schedule[node] = {}

        reduce_scatter_ni = np.zeros((self.network.nodes, self.timesteps), dtype=int)
        all_gather_ni = np.zeros((self.network.nodes, self.timesteps), dtype=int)
        for root in range(self.network.nodes):
            for edge in self.trees[root]:
                # reduce-scatter
                rs_child = edge[0]
                rs_parent = edge[1]
                rs_timestep = self.timesteps - edge[2] - 1

                # send from rs_child to rs_parent for tree root at rs_timestep
                if rs_timestep not in reduce_scatter_schedule[rs_child].keys():
                    reduce_scatter_schedule[rs_child][rs_timestep] = {}
                flow_children = [(root, child) for child in self.trees_children[root][rs_child]]
                reduce_scatter_schedule[rs_child][rs_timestep][root] = ((rs_parent, reduce_scatter_ni[rs_parent][rs_timestep]), flow_children, 1, rs_timestep)
                reduce_scatter_ni[rs_parent][rs_timestep] = (reduce_scatter_ni[rs_parent][rs_timestep] + 1) % self.args.radix

                # all-gather
                ag_child = edge[0]
                ag_parent = edge[1]
                ag_timestep = edge[2]

                # send from ag_parent to ag_child for tree root at ag_timestep
                if ag_timestep not in all_gather_schedule[ag_parent].keys():
                    all_gather_schedule[ag_parent][ag_timestep] = {}
                if root not in all_gather_schedule[ag_parent][ag_timestep].keys():
                    if ag_parent == root:
                        assert self.trees_parent[root][ag_parent] == None
                        all_gather_schedule[ag_parent][ag_timestep][root] = ([], None, 1, self.timesteps + 1)
                    else:
                        all_gather_schedule[ag_parent][ag_timestep][root] = ([], (root, self.trees_parent[root][ag_parent]), 1, ag_timestep + self.timesteps + 1)
                all_gather_schedule[ag_parent][ag_timestep][root][0].append((ag_child, all_gather_ni[ag_child][ag_timestep]))
                all_gather_ni[ag_child][ag_timestep] = (all_gather_ni[ag_child][ag_timestep] + 1) % self.args.radix

        # initialize the schedules
        self.reduce_scatter_schedule = {}
        self.all_gather_schedule = {}

        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = []
            self.all_gather_schedule[node] = []
            if verbose:
                print('Accelerator {}:'.format(node))
                print('  reduce-scatter schedule:')
            for timestep in range(self.timesteps):
                if timestep in reduce_scatter_schedule[node].keys():
                    self.reduce_scatter_schedule[node].append(reduce_scatter_schedule[node][timestep])
                    if verbose:
                        print('    timestep {}: {}'.format(timestep, reduce_scatter_schedule[node][timestep]))
                else:
                    self.reduce_scatter_schedule[node].append(None)
                    if verbose:
                        print('    timestep {}: no scheduled communication in this timestep'.format(timestep))
            flow_children = [(node, child) for child in self.trees_children[node][node]]
            self.reduce_scatter_schedule[node].append({node: ((None, None), flow_children, 0, self.timesteps)})
            if verbose:
                print('    root children: {}'.format(self.reduce_scatter_schedule[node][-1]))

            if verbose:
                print('  all-gather schedule:')
            for timestep in range(self.timesteps):
                if timestep in all_gather_schedule[node].keys():
                    self.all_gather_schedule[node].append(all_gather_schedule[node][timestep])
                    if verbose:
                        print('    timestep {}: {}'.format(timestep, all_gather_schedule[node][timestep]))
                else:
                    self.all_gather_schedule[node].append(None)
                    if verbose:
                        print('    timestep {}: no scheduled communication in this timestep'.format(timestep))
    # def generate_schedule(self, verbose=False)


def test(args):
    network = construct_network(args)

    kary = args.kary
    allreduce = MultiTreeAllreduce(args, network)
    # NOTE: sorted doesn't help for multitree since it only considers available links
    allreduce.compute_trees(kary, alternate=True, sort=False, verbose=args.verbose)
    if args.gendotfile:
        allreduce.generate_trees_dotfile('multitree.dot')
    timesteps = allreduce.timesteps
    allreduce.generate_schedule(verbose=args.verbose)
    allreduce.compute_trees(kary, alternate=True, sort=True, verbose=args.verbose)
    if args.gendotfile:
        allreduce.generate_trees_dotfile('multitree_sort.dot')
        #allreduce.generate_per_tree_dotfile('multitreedot')
    sort_timesteps = allreduce.timesteps
    allreduce.generate_schedule()
    allreduce.max_num_concurrent_flows()
    if timesteps > sort_timesteps:
        compare = 'Better'
    elif timesteps == sort_timesteps:
        compare = 'Same'
    else:
        compare = 'Worse'
    print('MultiTreeAllreduce takes {} timesteps (no sort), and {} timesteps (sort), {}'.format(
        timesteps, sort_timesteps, compare))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=32, type=int,
                        help='number of nodes, default is 32')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary tree, default is 2 (binary)')
    parser.add_argument('--radix', default=4, type=int,
                        help='node radix, default is 4')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='detailed print')
    parser.add_argument('--bigraph-m', default=4, type=int,
                        help='logical groups size (# sub-node per switch')
    parser.add_argument('--bigraph-n', default=8, type=int,
                        help='# switches')
    parser.add_argument('--booksim-network', default='bigraph',
                        help='network topology (torus | mesh | bigraph), default is torus')

    args = parser.parse_args()

    test(args)

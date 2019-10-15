import numpy as np
from copy import deepcopy

import networks
from allreduce import Allreduce


class MultiTreeAllreduce(Allreduce):
    def __init__(self, network):
        super().__init__(network)


    '''
    compute_trees() - computes allreduce spanning trees for the given network
    @kary: build kary-trees
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @verbose: print detailed info of tree construction process
    '''
    def compute_trees(self, kary, alternate=True, verbose=False):
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
        self.iterations = 0

        while num_trees < self.network.nodes:
            if verbose:
                print('iteration {}'.format(self.iterations))
            from_nodes = deepcopy(self.network.from_nodes)
            last_tree_nodes = deepcopy(tree_nodes)

            # alternating the link allocation every time for each tree
            if alternate:

                changed = True

                new_edges = {}
                for root in range(self.network.nodes):
                    new_edges[root] = {}
                    for parent in last_tree_nodes[root]:
                        new_edges[root][parent] = 0

                turns = 0
                while changed:
                    changed = False
                    old_from_nodes = deepcopy(from_nodes)

                    root = turns % self.network.nodes

                    while len(tree_nodes[root]) == self.network.nodes:
                        turns += 1
                        root = turns % self.network.nodes
                        continue
                    if verbose:
                        p = (turns // self.network.nodes) % len(tree_nodes[root])
                        parent = tree_nodes[root][p]
                        print('turns: {}, root: {}, p: {}, parent: {}'.format(turns, root, p, parent))

                    for parent in last_tree_nodes[root]:
                        children = deepcopy(from_nodes[parent])
                        if new_edges[root][parent] == kary - 1:
                            continue
                        for child in children:
                            if child not in tree_nodes[root]:
                                new_edges[root][parent] += 1
                                if verbose:
                                    print(' -- add node {} to tree {}'.format(child, root))
                                    print('    before: {}'.format(self.trees[root]))
                                from_nodes[parent].remove(child)
                                tree_nodes[root].append(child)
                                self.trees[root].append((child, parent, self.iterations))
                                if verbose:
                                    print('    after : {}'.format(self.trees[root]))
                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                changed = True
                                break
                        if changed:
                            break

                    turns += 1

                    if len(tree_nodes[root]) == self.network.nodes:
                        num_trees += 1
                        if verbose:
                            print('iteration {} - tree {} constructed: {}'.format(self.iterations, root, self.trees[root]))
                        if num_trees == self.network.nodes:
                            break

                    if turns % self.network.nodes != 0:
                        changed = True

            else:   # else case: allocating links for one tree as much as possble
                for root in range(self.network.nodes):
                    if len(tree_nodes[root]) == self.network.nodes:
                        continue
                    current_tree_nodes = deepcopy(tree_nodes[root])
                    for p, parent in enumerate(current_tree_nodes):
                        children = deepcopy(from_nodes[parent])
                        new_edges = 0
                        for child in children:
                            if new_edges == kary - 1:
                                break
                            if verbose:
                                print(' child {}'.format(child))
                            if child not in tree_nodes[root]:
                                if verbose:
                                    print(' -- add node {} to tree {}'.format(child, root))
                                    print('    before: {}'.format(self.trees[root]))
                                from_nodes[parent].remove(child)
                                tree_nodes[root].append(child)
                                self.trees[root].append((child, parent, self.iterations))
                                if verbose:
                                    print('    after : {}'.format(self.trees[root]))
                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                new_edges += 1
                    if len(tree_nodes[root]) == self.network.nodes:
                        num_trees += 1
                        if verbose:
                            print('iteration {} - tree {} constructed: {}'.format(self.iterations, root, self.trees[root]))
                    if verbose:
                        print('  tree {}: {}'.format(root, self.trees[root]))

            self.iterations += 1

        if verbose:
            print('Total iterations for network size of {}: {}'.format(self.network.nodes, self.iterations))
    # def compute_trees(self, kary, alternate=False, verbose=False)


def test():
    dimension = 4
    nodes = dimension * dimension
    network = networks.Torus(nodes, dimension)
    network.build_graph()
    allreduce = MultiTreeAllreduce(network)
    allreduce.compute_trees(2, alternate=True);
    allreduce.generate_trees_dotfile('multitree.dot')
    print('MultiTreeAllreduce takes {} iterations'.format(allreduce.iterations))


if __name__ == '__main__':
    test()

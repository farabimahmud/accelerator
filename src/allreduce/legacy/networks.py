import os
import numpy as np
from copy import deepcopy

class Torus:
    def __init__(self, nodes, dimension):
        self.nodes = nodes
        self.dimension = dimension
        self.from_nodes = {}
        self.to_nodes = {}
        self.edges = []
        self.adjacency_matrix = np.zeros((nodes, nodes))


    def build_graph(self, graphfile=None):

        link_weight = 2

        for node in range(self.nodes):
            self.from_nodes[node] = []
            self.to_nodes[node] = []

            row = node // self.dimension
            col = node % self.dimension
            #print('node {}: row {} col {}'.format(node, row, col))

            if row == 0:
                north = node + self.dimension * (self.dimension - 1)
                self.from_nodes[node].append(north)
                self.to_nodes[node].append(north)
                self.adjacency_matrix[node][north] = link_weight
                self.adjacency_matrix[north][node] = link_weight
            else:
                north = node - self.dimension
                self.from_nodes[node].append(north)
                self.to_nodes[node].append(north)
                self.adjacency_matrix[node][north] = link_weight
                self.adjacency_matrix[north][node] = link_weight

            if row == self.dimension - 1:
                south = node - self.dimension * (self.dimension - 1)
                self.from_nodes[node].append(south)
                self.to_nodes[node].append(south)
            else:
                south = node + self.dimension
                self.from_nodes[node].append(south)
                self.to_nodes[node].append(south)

            if col == 0:
                west = node + self.dimension - 1
                self.from_nodes[node].append(west)
                self.to_nodes[node].append(west)
                self.adjacency_matrix[node][west] = link_weight
                self.adjacency_matrix[west][node] = link_weight
            else:
                west = node - 1
                self.from_nodes[node].append(west)
                self.to_nodes[node].append(west)
                self.adjacency_matrix[node][west] = link_weight
                self.adjacency_matrix[west][node] = link_weight

            if col == self.dimension - 1:
                east = node - self.dimension + 1
                self.from_nodes[node].append(east)
                self.to_nodes[node].append(east)
            else:
                east = node + 1
                self.from_nodes[node].append(east)
                self.to_nodes[node].append(east)

        #print('torus graph: (node: from node list)')
        #for node in range(self.nodes):
        #    print(' -- {}: {}'.format(node, self.from_nodes[node]))

        if graphfile:
            for node in range(self.nodes):
                for i, n in enumerate(self.from_nodes[node]):
                    assert(not (node, n) in self.edges)
                    self.edges.append((node, n))

            graph = 'digraph G {\n'
            graph += '  subgraph {\n'
            graph += ''.join('    {} -> {};\n'.format(*e) for e in self.edges)

            for node in range(self.nodes):
                if node % self.dimension == 0:
                    graph += '  {rank = same; '
                graph += ' {};'.format(node)
                if node % self.dimension == self.dimension - 1:
                    graph += '}\n'

            graph += '  } /* closing subgraph */\n'
            graph += '}\n'

            f = open('torus_graph.dot', 'w')
            f.write(graph)
            f.close()
    # def build_graph(self, graphfile=None)


    def allreduce_trees(self, kary, alternate=False, verbose=False):
        assert kary > 1

        # initialize empty trees
        trees = {}
        tree_nodes = {}
        for node in range(self.nodes):
            trees[node] = []
            tree_nodes[node] = [node]
            if verbose:
                print('initialized tree {}: {}'.format(node, tree_nodes[node]))

        # tree construction
        num_trees = 0
        iteration = 0

        while num_trees < self.nodes:
            if verbose:
                print('iteration {}'.format(iteration))
            from_nodes = deepcopy(self.from_nodes)
            last_tree_nodes = deepcopy(tree_nodes)

            # alternating the link allocation every time for each tree
            if alternate:

                changed = True

                turns = 0
                while changed:
                    changed = False
                    old_from_nodes = deepcopy(from_nodes)

                    root = turns % self.nodes

                    while len(tree_nodes[root]) == self.nodes:
                        turns += 1
                        root = turns % self.nodes
                        continue
                    if verbose:
                        p = (turns // self.nodes) % len(tree_nodes[root])
                        parent = tree_nodes[root][p]
                        print('turns: {}, root: {}, p: {}, parent: {}'.format(turns, root, p, parent))

                    for parent in last_tree_nodes[root]:
                        children = deepcopy(from_nodes[parent])
                        for child in children:
                            if child not in tree_nodes[root]:
                                if verbose:
                                    print(' -- add node {} to tree {}'.format(child, root))
                                    print('    before: {}'.format(trees[root]))
                                from_nodes[parent].remove(child)
                                tree_nodes[root].append(child)
                                trees[root].append((child, parent, iteration))
                                if verbose:
                                    print('    after : {}'.format(trees[root]))
                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                changed = True
                                break
                        if changed:
                            break

                    turns += 1

                    if len(tree_nodes[root]) == self.nodes:
                        num_trees += 1
                        if verbose:
                            print('iteration {} - tree {} constructed: {}'.format(iteration, root, trees[root]))
                        if num_trees == self.nodes:
                            break

                    if turns % self.nodes != 0:
                        changed = True

            else:   # else case: allocating links for one tree as much as possble
                for root in range(self.nodes):
                    if len(tree_nodes[root]) == self.nodes:
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
                                    print('    before: {}'.format(trees[root]))
                                from_nodes[parent].remove(child)
                                tree_nodes[root].append(child)
                                trees[root].append((child, parent, iteration))
                                if verbose:
                                    print('    after : {}'.format(trees[root]))
                                    print('    tree nodes: {}'.format(tree_nodes[root]))
                                new_edges += 1
                    if len(tree_nodes[root]) == self.nodes:
                        num_trees += 1
                        if verbose:
                            print('iteration {} - tree {} constructed: {}'.format(iteration, root, trees[root]))
                    if verbose:
                        print('  tree {}: {}'.format(root, trees[root]))

            iteration += 1

        if verbose:
            print('Total iterations for network size of {}: {}'.format(self.nodes, iteration))

        return trees, iteration
    # def allreduce_trees(self, alternate=False, kary, verbose=False)


    def generate_trees_dotfile(self, filename, trees, iteration):
        colors = ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'

        ranks = {}
        node_rank = {}
        for rank in range(iteration + 1):
            ranks[rank] = []

        for root in range(self.nodes):
            tree += '    /* tree {} */\n'.format(root)
            ranks[0].append('"{}-{}"'.format(root, root))
            node_rank['"{}-{}"'.format(root, root)] = 0
            for edge in trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = iteration - edge[2]
                rank = edge[2] + 1
                ranks[rank].append(child)
                node_rank[child] = edge[2] + 1
                minlen = rank - node_rank[parent]
                tree += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(iteration + 1):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(iteration + 1):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open(filename, 'w')
        f.write(tree)
        f.close()
    # def generate_trees_dotfile(self, filename, trees, iteration)


def test():
    dimension = 4
    nodes = dimension * dimension
    network = Torus(nodes=nodes, dimension=dimension)
    network.build_graph()
    trees, iteration = network.allreduce_trees(2, True)
    network.generate_trees_dotfile('trees.dot', trees, iteration)


if __name__ == '__main__':
    test()

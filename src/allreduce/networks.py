import os
from copy import deepcopy

class Torus:
    def __init__(self, nodes, dimension):
        self.nodes = nodes
        self.dimension = dimension
        self.from_nodes = {}
        self.to_nodes = {}
        self.edges = []


    def build_graph(self, generate_graph=False):
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
            else:
                north = node - self.dimension
                self.from_nodes[node].append(north)
                self.to_nodes[node].append(north)

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
            else:
                west = node - 1
                self.from_nodes[node].append(west)
                self.to_nodes[node].append(west)

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

        if generate_graph:
            for node in range(self.nodes):
                for i, n in enumerate(self.from_nodes[node]):
                    assert(not (node, n) in self.edges)
                    self.edges.append((node, n))

            graph = 'digraph G {\n'
            graph += ''.join('    {} -> {};\n'.format(*e) for e in self.edges)

            #style = '    "{}" [style=""]'

            graph += '}\n'

            f = open('torus_graph.dot', 'w')
            f.write(graph)
            f.close()


    def allreduce_trees(self):
        # initialize empty trees
        trees = {}
        tree_nodes = {}
        for node in range(self.nodes):
            trees[node] = []
            tree_nodes[node] = [node]
            #print('initialized tree {}: {}'.format(node, tree_nodes[node]))

        # tree construction
        num_trees = 0
        iteration = 0

        while num_trees < self.nodes:
            #print('iteration {}'.format(iteration))
            from_nodes = deepcopy(self.from_nodes)
            for root in range(self.nodes):
                current_tree_nodes = deepcopy(tree_nodes[root])
                for p, parent in enumerate(current_tree_nodes):
                    children = deepcopy(from_nodes[parent])
                    new_edges = 0
                    for c, child in enumerate(children):
                        #print(' child {}'.format(child))
                        if child not in tree_nodes[root]:
                            #print(' -- add node {} to tree {}'.format(child, root))
                            #print('    before: {}'.format(trees[root]))
                            from_nodes[parent].remove(child)
                            tree_nodes[root].append(child)
                            trees[root].append((child, parent, iteration))
                            #print('    after : {}'.format(trees[root]))
                            #print('    tree nodes: {}'.format(tree_nodes[root]))
                            new_edges += 1
                            if new_edges == 2:
                                break
                if len(tree_nodes[root]) == self.nodes:
                    num_trees += 1
                    #print('iteration {} - tree {} constructed: {}'.format(iteration, root, trees[root]))
                #print('  tree {}: {}'.format(root, trees[root]))
            iteration += 1

        #extension = 0
        #directory = os.path.join('./trees')
        #directory = os.path.abspath(directory)

        #while os.path.exists('{}-{}'.format(directory, extension)):
        #    extension += 1

        #directory = '{}-{}'.format(directory, extension)

        #os.makedirs(directory)

        #colors = ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
        #colors = ['#ffffcc','#c7e9b4','#7fcdbb','#41b6c4','#2c7fb8','#253494']
        colors = ['#f1eef6','#d4b9da','#c994c7','#df65b0','#dd1c77','#980043']

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'


        ranks = {}
        for rank in range(iteration + 1):
            ranks[rank] = []

        for root in range(self.nodes):
            tree += '    /* tree {} */\n'.format(root)
            ranks[0].append('"{}-{}"'.format(root, root))
            for edge in trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = iteration - edge[2]
                ranks[edge[2] + 1].append(child)
                tree += ''.join('    {} -> {} [ label="{}" ];\n'.format(child, parent, cycle))

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
                tree += ''.join(style.format(node, colors[rank]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        #f = open('{}/tree{}.dot'.format(directory, root), 'w')
        f = open('trees.dot', 'w')
        f.write(tree)
        f.close()



def main():
    network = Torus(nodes=16, dimension=4)
    network.build_graph()
    network.allreduce_trees()

if __name__ == '__main__':
    main()

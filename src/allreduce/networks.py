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


def test():
    dimension = 4
    nodes = dimension * dimension
    network = Torus(nodes=nodes, dimension=dimension)
    network.build_graph()


if __name__ == '__main__':
    test()

import argparse

from network import Network

class KNCube(Network):
    def __init__(self, args, mesh=False):
        self.mesh = mesh
        super().__init__(args)


    '''
    build_graph() - build the topology graph
    @filename: filename to generate topology dotfile, optional
    '''
    def build_graph(self, filename=None):

        link_weight = 2

        for node in range(self.nodes):
            self.from_nodes[node] = []
            self.to_nodes[node] = []

            row = node // self.dimension
            col = node % self.dimension
            #print('node {}: row {} col {}'.format(node, row, col))

            if row == 0 and not self.mesh:
                if self.dimension > 2:
                    north = node + self.dimension * (self.dimension - 1)
                    self.from_nodes[node].append(north)
                    self.to_nodes[node].append(north)
                    self.adjacency_matrix[node][north] = link_weight
                    self.adjacency_matrix[north][node] = link_weight
            elif row != 0:
                north = node - self.dimension
                self.from_nodes[node].append(north)
                self.to_nodes[node].append(north)
                self.adjacency_matrix[node][north] = link_weight
                self.adjacency_matrix[north][node] = link_weight

            if row == self.dimension - 1 and not self.mesh:
                if self.dimension > 2:
                    south = node - self.dimension * (self.dimension - 1)
                    self.from_nodes[node].append(south)
                    self.to_nodes[node].append(south)
            elif row != self.dimension - 1:
                south = node + self.dimension
                self.from_nodes[node].append(south)
                self.to_nodes[node].append(south)

            if col == 0 and not self.mesh:
                if self.dimension > 2:
                    west = node + self.dimension - 1
                    self.from_nodes[node].append(west)
                    self.to_nodes[node].append(west)
                    self.adjacency_matrix[node][west] = link_weight
                    self.adjacency_matrix[west][node] = link_weight
            elif col != 0:
                west = node - 1
                self.from_nodes[node].append(west)
                self.to_nodes[node].append(west)
                self.adjacency_matrix[node][west] = link_weight
                self.adjacency_matrix[west][node] = link_weight

            if col == self.dimension - 1 and not self.mesh:
                if self.dimension > 2:
                    east = node - self.dimension + 1
                    self.from_nodes[node].append(east)
                    self.to_nodes[node].append(east)
            elif col != self.dimension - 1:
                east = node + 1
                self.from_nodes[node].append(east)
                self.to_nodes[node].append(east)

        #if self.mesh:
        #    print('mesh graph: (node: from node list)')
        #else:
        #    print('torus graph: (node: from node list)')
        #for node in range(self.nodes):
        #    print(' -- {}: {}'.format(node, self.from_nodes[node]))

        if filename:
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

            f = open(filename, 'w')
            f.write(graph)
            f.close()
    # def build_graph(self, filename=None)


    '''
    distance() - distance between two nodes
    @src: source node ID
    @dest: destination node ID
    '''
    def distance(self, src, dest):
        src_x = src // self.dimension
        src_y = src % self.dimension
        dest_x = dest // self.dimension
        dest_y = dest % self.dimension
        if self.mesh:
            dist = abs(src_x - dest_x) + abs(dest_x - dest_y)
        else:
            dist_x = abs(src_x - dest_x)
            dist_y = abs(src_y - dest_y)
            if dist_x > self.dimension // 2:
                dist_x = self.dimension - dist_x
            if dist_y > self.dimension // 2:
                dist_y = self.dimension - dist_y
            dist = dist_x + dist_y

        return dist
    # end of distance()


def test():
    dimension = 4
    nodes = dimension * dimension
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=4, type=int,
                        help='network dimension, default is 4')
    parser.add_argument('--filename', default=None,
                        help='File name for topology dotfile, default None (no dotfile)')
    parser.add_argument('--mesh', default=False, action='store_true',
                        help='Create mesh network, default False (torus)')

    args = parser.parse_args()

    network = KNCube(args, args.mesh)
    network.build_graph(args.filename)


if __name__ == '__main__':
    test()

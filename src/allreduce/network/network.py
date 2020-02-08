import os
import numpy as np
import math
from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self, args):
        self.args = args
        self.nodes = args.dimension * args.dimension
        self.dimension = args.dimension
        self.from_nodes = {}
        self.to_nodes = {}
        self.edges = []
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))


    '''
    build_graph() - build the topology graph
    @filename: filename to generate topology dotfile, optional
    '''
    @abstractmethod
    def build_graph(self, filename=None):
        pass


    '''
    distance() - distance between two nodes
    @src: source node ID
    @dest: destination node ID
    '''
    @abstractmethod
    def distance(self, src, dest):
        pass


from kncube import KNCube


'''
construct_network() - construct a network
@args: argumetns of the top simulation

return: a network object
'''
def construct_network(args):
    dimension = int(math.sqrt(args.num_hmcs))
    assert args.num_hmcs == dimension * dimension
    args.dimension = dimension
    network = None

    if args.booksim_network == 'mesh':
        network = KNCube(args, mesh=True)
    elif args.booksim_network == 'torus':
        network = KNCube(args)
    else:
        raise RuntimeError('Unknown network topology: ' + args.booksim_network)

    network.build_graph()

    return network

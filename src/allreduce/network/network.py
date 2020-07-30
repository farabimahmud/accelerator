import os
import numpy as np
import math
from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self, args):
        self.args = args
        self.nodes = args.nodes
        self.from_nodes = {}
        self.to_nodes = {}
        self.edges = []
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
        self.node_to_switch = {}
        self.switch_to_switch = {}
        self.switch_to_node = {}
        self.priority = [0] * self.nodes # used for allocation sequence


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
from bigraph import BiGraph
from dgx2 import DGX2


'''
construct_network() - construct a network
@args: argumetns of the top simulation

return: a network object
'''
def construct_network(args):
    args.nodes = args.num_hmcs
    network = None

    if args.booksim_network == 'mesh':
        network = KNCube(args, mesh=True)
    elif args.booksim_network == 'torus':
      network = KNCube(args)
    elif args.booksim_network == 'bigraph':
        network = BiGraph(args)
    elif args.booksim_network == 'dgx2':
        network = DGX2(args)
    else:
        raise RuntimeError('Unknown network topology: ' + args.booksim_network)

    network.build_graph()

    return network

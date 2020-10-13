import argparse
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class Ring2DAllreduce(Allreduce):
    def __init__(self, args, network):
        super().__init__(args, network)
        assert network.type == 'Mesh' or network.type == 'Torus'
        self.xrings = []
        self.yrings = []


    '''
    compute_trees() - computes 2D rings (special tree) for the given network
    @kary: not used, skip
    @alternate: not used, skip
    @sort: not used, skip
    @verbose: print detailed info of ring construction process
    '''
    def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False):
        dimension = self.network.dimension

        for d1 in range(dimension):
            xbase = d1 * dimension
            ybase = d1
            self.xrings.append([])
            self.yrings.append([])
            for d2 in range(dimension):
                self.xrings[-1].append(xbase + d2)
                self.yrings[-1].append(ybase + d2 * dimension)

        if verbose:
            print('2D Rings:')
            print('  X-Rings:')
            for i, ring in enumerate(self.xrings):
                print('   - {}: {}'.format(i, ring))
            print('  Y-Rings:')
            for i, ring in enumerate(self.yrings):
                print('   - {}: {}'.format(i, ring))

        self.timesteps = dimension - 1

        self.num_flows = 4 * self.network.nodes

        directions = {0: 'east', 1: 'west', 2: 'south', 3: 'north'}

        # from the trees - fake trees
        self.trees = {}
        for flow in range(self.num_flows):
            self.trees[flow] = []

            root = flow // 4
            flow_dir = directions[flow % 4]
            base = (root // dimension) * dimension

            if verbose:
                print('Flow tree {} (direction {})'.format(flow, flow_dir))

            if flow_dir == 'east':
                for timestep in range(dimension - 1):
                    parent_east = (root % dimension + timestep) % dimension + base
                    child_east = (root % dimension + timestep + 1) % dimension + base
                    self.trees[flow].append((child_east, parent_east, timestep))

                    if verbose:
                        print('  step {}: {}->{}'.format(timestep, parent_east, child_east))

            elif flow_dir == 'west':
                for timestep in range(dimension - 1):
                    parent_west = (root % dimension - timestep) % dimension + base
                    child_west = (root % dimension - timestep - 1) % dimension + base
                    self.trees[flow].append((child_west, parent_west, timestep))

                    if verbose:
                        print('  step {}: {}->{}'.format(timestep, parent_west, child_west))

            elif flow_dir == 'south':
                for timestep in range(dimension - 1):
                    parent_south = (root + timestep * dimension) % self.network.nodes
                    child_south = (root + (timestep + 1) * dimension) % self.network.nodes
                    self.trees[flow].append((child_south, parent_south, timestep))

                    if verbose:
                        print('  step {}: {}->{}'.format(timestep, parent_south, child_south))

            elif flow_dir == 'north':
                for timestep in range(dimension - 1):
                    parent_north = (root - timestep * dimension) % self.network.nodes
                    child_north = (root - (timestep + 1) * dimension) % self.network.nodes
                    self.trees[flow].append((child_north, parent_north, timestep))

                    if verbose:
                        print('  step {}: {}->{}'.format(timestep, parent_north, child_north))

            else:
                raise RuntimeError('Unknown flow direction {} for flow {}'.format(flow_dir, flow))

    # def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False)


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule for 2d-ring
    '''
    def generate_schedule(self, verbose=False):
        # compute parent-children dependency
        self.trees_parent = {}
        self.trees_children = {}

        for flow in range(self.num_flows):
            root = flow // 4

            self.trees_parent[flow] = {}
            self.trees_parent[flow][root] = None

            self.trees_children[flow] = {}
            for node in range(self.network.nodes):
                self.trees_children[flow][node] = []

            for edge in self.trees[flow]:
                child = edge[0]
                parent = edge[1]
                self.trees_parent[flow][child] = parent
                self.trees_children[flow][parent].append(child)

        # initialize the schedules
        reduce_scatter_schedule = {}
        all_gather_schedule = {}

        # construct schedules for each node from trees
        for node in range(self.network.nodes):
            reduce_scatter_schedule[node] = {}
            all_gather_schedule[node] = {}

        reduce_scatter_ni = np.zeros((self.network.nodes, self.timesteps), dtype=int)
        all_gather_ni = np.zeros((self.network.nodes, self.timesteps), dtype=int)

        datachunk = self.network.dimension / 4

        for flow in range(self.num_flows):
            for edge in self.trees[flow]:
                # reduce-scatter
                rs_child = edge[0]
                rs_parent = edge[1]
                rs_timestep = self.timesteps - edge[2] - 1

                # send from rs_child to rs_parent for tree root at rs_timestep
                if rs_timestep not in reduce_scatter_schedule[rs_child].keys():
                    reduce_scatter_schedule[rs_child][rs_timestep] = {}

                flow_children = [(flow, child) for child in self.trees_children[flow][rs_child]]
                reduce_scatter_schedule[rs_child][rs_timestep][flow] = ((rs_parent, reduce_scatter_ni[rs_parent][rs_timestep]), flow_children, datachunk, rs_timestep)
                reduce_scatter_ni[rs_parent][rs_timestep] = (reduce_scatter_ni[rs_parent][rs_timestep] + 1) % self.args.radix

                # all-gather
                ag_child = edge[0]
                ag_parent = edge[1]
                ag_timestep = edge[2]

                # send from ag_parent to ag_child for tree root at ag_timestep
                if ag_timestep not in all_gather_schedule[ag_parent].keys():
                    all_gather_schedule[ag_parent][ag_timestep] = {}

                if flow not in all_gather_schedule[ag_parent][ag_timestep].keys():
                    if self.trees_parent[flow][ag_parent] == None:
                        all_gather_schedule[ag_parent][ag_timestep][flow] = ([], None, datachunk, self.timesteps + ag_timestep + 1)
                    else:
                        all_gather_schedule[ag_parent][ag_timestep][flow] = ([], (flow, self.trees_parent[flow][ag_parent]), datachunk, ag_timestep + self.timesteps + 1)
                all_gather_schedule[ag_parent][ag_timestep][flow][0].append((ag_child, all_gather_ni[ag_child][ag_timestep]))
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

            flow_children = []
            for i in range(4):
                flow = node * 4 + i
                assert len(self.trees_children[flow][node]) == 1
                flow_children.append((flow, self.trees_children[flow][node][0]))
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
    # network.to_nodes[1].clear() # test no solution case

    allreduce = Ring2DAllreduce(args, network)
    allreduce.compute_trees(verbose=args.verbose)
    allreduce.generate_schedule(verbose=args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=16, type=int,
                        help='number of nodes, default is 16')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary tree, default is 2 (binary)')
    parser.add_argument('--radix', default=4, type=int,
                        help='node radix, default is 4')
    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch), default 8')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches, default 2')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='detailed print')
    parser.add_argument('--booksim-network', default='torus',
                        help='network topology (torus | mesh | dgx2), default is torus')

    args = parser.parse_args()

    test(args)

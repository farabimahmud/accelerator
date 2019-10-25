from abc import ABC, abstractmethod

class Allreduce(ABC):
    def __init__(self, network):
        self.network = network
        self.trees = None
        self.timesteps = None
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None


    '''
    compute_trees() - computes allreduce spanning trees for the given network
    '''
    @abstractmethod
    def compute_trees(self, kary, alternate=False, sort=True, verbose=False):
        pass


    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from trees
    '''
    def generate_schedule(self, verbose=False):
        # initialize the schedules
        self.reduce_scatter_schedule = {}
        self.all_gather_schedule = {}

        # construct schedules for each node from trees
        for node in range(self.network.nodes):
            self.reduce_scatter_schedule[node] = {}
            self.all_gather_schedule[node] = {}

        for root in range(self.network.nodes):
            for edge in self.trees[root]:
                # reduce-scatter
                rs_child = edge[0]
                rs_parent = edge[1]
                rs_timestep = self.timesteps - edge[2] - 1

                # send from rs_child to rs_parent for tree root at rs_timestep
                if rs_timestep not in self.reduce_scatter_schedule[rs_child].keys():
                    self.reduce_scatter_schedule[rs_child][rs_timestep] = []
                self.reduce_scatter_schedule[rs_child][rs_timestep].append((root, rs_parent))

                # all-gather
                ag_child = edge[0]
                ag_parent = edge[1]
                ag_timestep = edge[2]

                # send from ag_parent to ag_child for tree root at ag_timestep
                if ag_timestep not in self.all_gather_schedule[ag_parent].keys():
                    self.all_gather_schedule[ag_parent][ag_timestep] = {}
                if root not in self.all_gather_schedule[ag_parent][ag_timestep].keys():
                    self.all_gather_schedule[ag_parent][ag_timestep][root] = []
                self.all_gather_schedule[ag_parent][ag_timestep][root].append(ag_child)

        if verbose:
            for node in range(self.network.nodes):
                print('Accelerator {}:'.format(node))
                print('  reduce-scatter schedule: {}'.format(self.reduce_scatter_schedule[node]))
                print('  all-gather schedule: {}'.format(self.all_gather_schedule[node]))
    # def generate_schedule(self, verbose=False)


    '''
    generate_trees_dotfile() - generate dotfile for computed trees
    @filename: name of dotfile
    '''
    def generate_trees_dotfile(self, filename):
        # color palette for ploting nodes of different tree levels
        colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
                '#e7298a', '#ce1256', '#980043', '#67001f']

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'

        # group nodes with same rank (same tree level/iteration)
        # and set up the map for node name and its rank in node_rank
        ranks = {}
        node_rank = {}
        for rank in range(self.timesteps + 1):
            ranks[rank] = []

        for root in range(self.network.nodes):
            minrank = self.timesteps
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                rank = edge[2] + 1
                ranks[rank].append(child)
                node_rank[child] = rank
                if edge[1] == root and rank - 1 < minrank:
                    minrank = rank - 1
            ranks[minrank].append('"{}-{}"'.format(root, root))
            node_rank['"{}-{}"'.format(root, root)] = minrank

        for root in range(self.network.nodes):
            tree += '    /* tree {} */\n'.format(root)
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = self.timesteps - edge[2]
                minlen = node_rank[child] - node_rank[parent] # for strict separation of ranks
                tree += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open(filename, 'w')
        f.write(tree)
        f.close()
    # def generate_trees_dotfile(self, filename)

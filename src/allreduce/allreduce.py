from abc import ABC, abstractmethod

class Allreduce(ABC):
    def __init__(self, network):
        self.network = network
        self.trees = None
        self.timesteps = None
        '''
        schedules are organized as list of list, the list with lower index
        in the schedule should be scheduled earlier.
        - reduce_scatter_schedule:
            subflow: (parent, [dependent children]) // subflow is 'tree' root
        - all_gather_schedule:
            subflow: ([children], dependent parent)
        Ring:
            0->1->2->3->0
            reduce_scatter_schedule[0] = [
                {3: (1, [])},
                {2: (1, [3])},
                {1: (1, [3])}
            ]
            all_gather_schedule[0] = [
                {0: ([1], None)},
                {3: ([1], 3)},
                {2: ([1], 3)}
            ]
        MXNet: (only dependencies among children and parent)
              Tree 0      Tree 1        Tree 2        Tree 3
                0           1             2             3
              0   1       1   3         2   3         3   1
            0  2 1  3   1  0 3  2     2  0 3  1     3  2 1  0
            reduce_scatter_schedule[3] = [
                {0: (1, []), 1: (1, [2]), 2: (2, [1])}
            ]
            all_gather_schedule[3] = [
                {1: ([2], 1), 2: ([1], 2), 3: ([1, 2], None)}
            ]
        MultTree:
            Timestep    Tree 0      Tree 1        Tree 2        Tree 3
                2         0           1             2             3
                1          2           3             0             1
                0       1   3       0   2         3   1         2   0
            reduce_scatter_schedule[0] = [
                {1: (1, []), 3: (1, [])},
                {2: (2, [1])},
            ]
            all_gather_schedule[0] = [
                {0: ([2], None)},
                {0: ([1], None), 2: ([1], 2)}
            ]
        '''
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
    @abstractmethod
    def generate_schedule(self, verbose=False):
        pass


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

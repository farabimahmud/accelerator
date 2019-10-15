from abc import ABC, abstractmethod

class Allreduce(ABC):
    def __init__(self, network):
        self.network = network
        self.trees = None
        self.iterations = None


    '''
    compute_trees() - computes allreduce spanning trees for the given network
    '''
    @abstractmethod
    def compute_trees(self, kary, alternate=False, verbose=False):
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
        for rank in range(self.iterations + 1):
            ranks[rank] = []

        for root in range(self.network.nodes):
            tree += '    /* tree {} */\n'.format(root)
            ranks[0].append('"{}-{}"'.format(root, root))
            node_rank['"{}-{}"'.format(root, root)] = 0
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = self.iterations - edge[2]
                rank = edge[2] + 1
                ranks[rank].append(child)
                node_rank[child] = edge[2] + 1
                minlen = rank - node_rank[parent] # for strict separation of ranks
                tree += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(self.iterations + 1):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(self.iterations + 1):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open(filename, 'w')
        f.write(tree)
        f.close()
    # def generate_trees_dotfile(self, filename)

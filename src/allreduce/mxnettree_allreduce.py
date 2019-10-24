import argparse
import numpy as np
import random
from datetime import datetime
from copy import deepcopy

import networks
from allreduce import Allreduce


class MXNetTreeAllreduce(Allreduce):
    def __init__(self, network):
        super().__init__(network)
        self.alpha = 0.7    # link usage penalty
        self.backtrack = False   # whether or not to use backtracking to generate trees
        self.topology = None   # topology stores the generated (maybe) link-conflict trees
        self.scan = None       # scan stores the start of each level of each conflict tree
        self.conflict_trees = None
        self.adjacency_matrix = None #deepcopy(self.network.adjacency_matrix)
        self.mxnet_maxdepth = 16
        self.silent = False


    '''
    compute_best_trees() - computes binary spanning trees for the given network
    @kary: build kary-trees
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @sort: Whether sort the trees for link allocation based on conflicts from
           last allocation iteration
    @verbose: print detailed info of tree construction process

    desc - try multiple random seeds for KL algorithm and select the best trees
    '''
    def compute_best_trees(self, trials, kary, alternate=True, sort=True, verbose=False):
        # TODO: this is for torus network, for other networks, it should be changed
        if self.network.nodes > 16:
            self.compute_trees(kary, alternate=alternate, sort=sort, verbose=verbose)
            return

        # set random seed as the current time
        random.seed(datetime.now())

        if trials < 10:
            end = 100
        else:
            end = trials * 10

        seeds = random.sample(range(0, end), trials)
        best_seed = None
        best_iterations = None
        for seed in seeds:
            random.seed(seed)
            self.compute_trees(kary, alternate=alternate, sort=sort, verbose=verbose)
            if best_iterations == None or best_iterations > self.iterations:
                best_seed = seed
                best_iterations = self.iterations

        if best_seed != seeds[-1]:
            random.seed(best_seed)
            self.compute_trees(kary, alternate=alternate, sort=sort, verbose=verbose)
    # def compute_best_trees(self, trials, kary, alternate=True, sort=True, verbose=False)

    '''
    compute_trees() - computes binary spanning trees for the given network
    @kary: useless, skip
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @sort: Whether sort the trees for link allocation based on conflicts from
           last allocation iteration
    @verbose: print detailed info of tree construction process
    '''
    def compute_trees(self, kary, alternate=True, sort=True, verbose=False):
        self.topology = []
        self.scan = []
        self.adjacency_matrix = deepcopy(self.network.adjacency_matrix)

        for root in range(self.network.nodes):
            self.topology.append([])
            self.scan.append([])
            self.topology[root].append(root)
            self.scan[root].append(0)
            self.compute_tree_from_root(root)

        filename = None
        if verbose:
            for root in range(self.network.nodes):
                self.print_topo("\nTree {}".format(root), self.topology[root], self.scan[root])

            self.link_conflict_detection()

            if not self.silent:
                filename = 'conflict_trees.dot'

        filename = 'conflict_trees.dot'
        self.generate_conflict_trees(filename)
        self.mxnet_schedule(kary, alternate=alternate, cross_level=True, sort=sort, verbose=verbose)
        #self.topdown_schedule(kary, alternate=alternate, cross_level=True, sort=sort, verbose=verbose)
    # def compute_trees(self, kary, alternate=True, verbose=False)


    '''
    compute_tree_from_root() - compute a tree from a fixed root
    @adjacency_matrix: adjacency matrix of the link topology
    @root: the root id of the tree
    '''
    def compute_tree_from_root(self, root):
        num_partitions = 1

        # Initialize partition array to indicate which partition each element
        # belongs to, beginning with 0
        partitions = np.zeros(self.network.nodes, dtype=int)

        # Initialize vector of pairs that tells the edges between what 2
        # clusters should be looked to build the tree from
        cluster_pairs = []

        # Initialize vector of roots that will tell us edges between
        roots = set()
        roots.add(root)

        # Temporary variables for rewinding
        partitions_temp = []
        num_partitions_temp = None
        roots_temp = set()
        topo_temp = []
        scan_temp = []

        # Determine number of partition levels
        # If first partition, determine root of maximal spanning tree
        stop = False
        reset = True
        level = 0

        # TODO: the last condition is just for torus
        while (not self.backtrack) and (not stop or reset) and (self.network.nodes <= 16):
            if reset:
                cluster_pairs.clear()
                partitions_temp = deepcopy(partitions)
                num_partitions_temp = num_partitions
                roots_temp = deepcopy(roots)
                topo_temp = deepcopy(self.topology[root])
                scan_temp = deepcopy(self.scan[root])

            # Run Kernighan-Lin to generate partition
            stop, num_partitions_temp = self.kernighan_lin(partitions_temp, num_partitions_temp, cluster_pairs)

            # Use partitions found and a given root to find best inter-cluster edge
            # for each pair of clusters, and returns them as roots of next cluster.
            # If reset is true, then rewind back to previous clustering
            reset = self.kl_generate_binary_tree(partitions_temp, cluster_pairs, roots_temp, topo_temp, scan_temp)

            if reset:
                level += 1
            if level > 10:
                break

        success = True
        if reset:
            if self.network.nodes > 16:
                if not self.silent:
                    print('No valid tree found from root {} and network too big ({} nodes)'
                            ', compute torus tree'.format(root, self.network.nodes))
                self.compute_torus_tree(root)
            else:
                if not self.silent:
                    print("No valid tree found from root {}, try backtracking".format(root))
                success = self.backtrack_generate_binary_tree(root)
        else:
            self.topology[root].clear()
            self.topology[root] = deepcopy(topo_temp)
            self.scan[root].clear()
            self.scan[root] = deepcopy(scan_temp)
            self.scan[root].append(len(self.topology[root]))

        if success:
            self.update_weight(root)
        else:
            print("No valid binary tree found from root {} using backtracking".format(root))
    # def compute_tree_from_root(self, adjacency_matrix, root)


    '''
    kernighan_lin() - Kernighan-Lin partitioning implementation
    @partitions: current partitions
    @num_partitions: current number of partitions
    @cluster_pairs: stores the mapping that tells which 2 clusters are the output of
                    partitioning one larger cluster

    desc - performs partition on each existing partition in link topology graph if
           partition has more than 4 elements in it

    return:
    @stop: True - no partitions with >= 4 elements found
           False - otherwise
    @num_partitions: number of updated partitions
    '''
    def kernighan_lin(self, partitions, num_partitions, cluster_pairs):
        histogram = np.zeros(num_partitions)
        partitions_temp = np.zeros(len(partitions))
        partitions_temp2 = np.zeros(len(partitions))
        D = np.zeros(len(partitions))
        D_temp = np.zeros(len(partitions))

        # Step 0: For every partition, determine if it can be partitioned further
        #   First od a histogram of each partiotion
        for partition in partitions:
            histogram[partition] += 1

        stop = True
        for color in range(len(histogram)):
            partition_size = histogram[color]
            # Save cluster in preparation for push to topo in GenerateBianryTree()
            if partition_size <= 2:
                cluster_pairs.append((color, -int(partition_size)))
            else:
                # Do Kernighan-Lin if clustering is necessary
                stop = False

                # Step 1: If it has more than 4 elements, we can partition further.
                #   Assign random balaned partition of it
                #   -balanced is more important than random, so allocate first half to A
                #   and rest to B
                first_partition = 0
                target_partition = int(partition_size // 2)
                cluster_list = []

                for i, partition in enumerate(partitions):
                    # Required to shift from [0,1] to {-1,1}
                    #  1 means vertex i is in Cluster A
                    # -1 means vertex i is in Cluster B
                    if partition == color:
                        cluster_list.append(i)
                    else:
                        partitions_temp[i] = 0

                # Step 1b: Shuffle using random generator
                random.shuffle(cluster_list)
                for cluster in cluster_list:
                    if first_partition < target_partition:
                        partitions_temp[cluster] = 1
                        first_partition += 1
                    else:
                        partitions_temp[cluster] = -1

                # Step 2: Iterate Kernighan-Lin until convergence
                g_max = 0
                g_k = -1
                count = 0
                while True:
                    count += 1
                    partitions_temp2 = deepcopy(partitions_temp)

                    # a) Compute difference between external and internal costs of all
                    #    elements of vector D
                    self.gemv(partitions_temp, D)
                    self.ewisemult(partitions_temp, -1.0, D)

                    # av and bv are used to hold candidates for moving
                    # gv stores the score associated with move
                    av = []
                    bv = []
                    gv = []

                    # used stores the ones have been moved
                    used = set()

                    nrows = int(partition_size // 2)
                    for iteration in range(nrows):
                        # b) Find best move by looking through upper triangular of weight matrix
                        g, a, b = self.find_best_move(partitions_temp, D, used)
                        if g == 0:
                            g_max = 0
                            break

                        # c) Store best move from consideration in vector parition_temp
                        av.append(a)
                        bv.append(b)
                        gv.append(g)

                        # d) Eliminate best move from consideration in vector partitions_temp
                        partitions_temp[a] *= -1
                        partitions_temp[b] *= -1
                        used.add(a)
                        used.add(b)

                        # e) Update D usig partitiion_temp
                        self.gemv(partitions_temp, D)
                        self.ewisemult(partitions_temp, -1.0, D)
                        D[a] = 0
                        D[b] = 0

                    # Step 3: Find when to stop by doing linear scan through gv
                    #    Recompute score g_max
                    for k in range(len(gv)):
                        if k > 0:
                            gv[k] += gv[k - 1]
                        if gv[k] > g_max:
                            g_max = gv[k]
                            g_k = k + 1

                    # Step 4: If move is "good", commit moves by updating partitions_temp and partitions_temp2
                    #    Otherwise, rollback changes to partitions_temp2
                    if g_max > 0:
                        for i in range(g_k):
                            a = av[i]
                            b = bv[i]
                            temp = partitions_temp2[a]
                            partitions_temp2[a] = partitions_temp2[b]
                            partitions_temp2[b] = temp
                    parition_temp = deepcopy(partitions_temp2)

                    if g_max == 0 or count > len(partitions):
                        break

                # Step 5: Update partitions using partitions_temp
                moves = 0
                for i in range(len(partitions)):
                    if partitions_temp[i] == -1:
                        partitions[i] = num_partitions
                        moves += 1
                cluster_pairs.append((color, num_partitions))

                num_partitions += 1

        return stop, num_partitions
    # def kernighan_lin(self, partitions, num_partitions, cluster_pairs)


    '''
    kl_generate_binary_tree() - append new nodes to binary tree
    @partitions: partitioning result of KL algorithm
    @cluser_pairs: pairing between clusters, an edge is found between each pair
    @roots: source vertices

    return:
    @reset: True if cannot find new edges to add
    '''
    def kl_generate_binary_tree(self, partitions, cluster_pairs, roots, topo_row, scan_row):
        new_roots = set()
        new_topo = {}
        reset = False

        for i, (first, second) in enumerate(cluster_pairs):
            if i == 0:
                scan_row.append(len(topo_row))
            parent = -1
            child = -1
            if second == -2:
                # Root must be color of first
                color = first
                parent = self.get_root(partitions, color, roots)
                if parent == -1:
                    return True
                child = self.get_child(partitions, color, parent)
            elif second == -1:
                color = first
                parent = self.get_root(partitions, color, roots)
                if parent == -1:
                    return True
                child = parent
            else:
                # Root must exist in either first or second element of pair
                color = first
                parent = self.get_root(partitions, color, roots)
                if parent == -1:
                    color = second
                    parent = self.get_root(partitions, color, roots)

                from_cluster = color
                to_cluster = first
                if from_cluster == first:
                    to_cluster = second

                candidates, weight = self.find_best_edge(partitions, parent, to_cluster)

                # if no candidates
                if candidates[0] != -1:
                    random.shuffle(candidates)
                    child = candidates[0]

                if child == -1:
                    reset = True
                    return reset
                else:
                    new_roots.add(parent)
                    new_roots.add(child)

            new_topo[parent] = child

        depth = len(scan_row)
        start = scan_row[depth - 2]
        end = scan_row[depth - 1]

        for i in range(start, end):
            parent = topo_row[i]
            child = None

            # If not first, check previous level whether or not we are encountering
            # this root for the first time in this level of the tree
            if i != start and parent == topo_row[i - 1]:
                child = parent
            else:
                child = new_topo[parent]
            topo_row.append(parent)
            topo_row.append(child)

        cluster_pairs.clear()
        roots.clear()
        roots.update(new_roots)

        return reset
    # def kl_generate_binary_tree(self, partitions, cluster_pairs, roots, topo_row, scan_row)


    '''
    compute_torus_tree() - compute trees for torus from root
    @root: root id of the current tree
    '''
    def compute_torus_tree(self, root):
        # Clear before starting
        self.topology[root].clear()
        self.scan[root].clear()

        tree = [[root]]
        nodes = set()
        nodes.add(root)

        depth = 0
        while len(nodes) < self.network.nodes:

            tree.append([])
            num_nodes = 1 << depth

            for parent in reversed(tree[depth]):
                left_child = parent
                right_child = -1
                for neighbor in self.network.from_nodes[parent]:
                    if neighbor not in nodes:
                        right_child = neighbor
                        nodes.add(neighbor)
                        break
                if right_child == -1:
                    right_child = parent
                tree[depth+1].insert(0, right_child)
                tree[depth+1].insert(0, left_child)

            depth += 1

        result = np.array(tree[-1], dtype=int)

        self.post_process(result, depth)
        #print('torus tree {} result: {}'.format(root, result))

        self.form_topology(result, root, depth)
    # def compute_torus_tree(self, root)


    '''
    backtrack_generate_binary_tree() - brute-force backtracking to find a binary tree
    @root: root id of the current tree

    desc - If Kernighan-Lin fails to find a binary tree, use brute-force backtracking
           approach to find one with following constraints:
           1) minimize depth (balance)
           2) maximize edge weight
           3) tree is binary

    return:
    @success: Ture if successfully find one and False otherwise
    '''
    def backtrack_generate_binary_tree(self, root):
        # Clear before starting
        self.topology[root].clear()
        self.scan[root].clear()

        # Compute depth
        # num_elemennts: depth
        # 5: 3 8
        # 6: 3 8
        # 7: 3 8
        # 8: 3 8
        # 9: 4 16
        depth = self.compute_depth()
        depth_leaves = 1 << depth

        # State vector
        # -1 means unplaced
        state = -np.ones(depth_leaves, dtype=int)
        result = -np.ones(depth_leaves, dtype=int)
        result_weight = 0

        # Place root and try all combinations
        state[0] = root

        # Seek optimal solution until depth <= 3 i.e. 8 accelerators
        # For larger number of accelerators, settle for first tree found (non-optimal),
        # but this saves a lot of runtime since Backtrack is exponential time
        if depth <= 3:
            result_weight, result = self.iterative_backtrack(state, result, result_weight, depth, True)
        else:
            result_weight, result = self.iterative_backtrack(state, result, result_weight, depth, False)

        success = self.form_topology(result, root, depth)

        return success
    # def backtrack_generate_binary_tree(self, root)


    '''
    iterative_backtrack() - recursive/iterative backtracking to find the satisfied tree
    @state: state vector for the leaves to indicate placed or not
    @best_result: best tree found so far
    @best_result_weight: weight of the best tree found so far
    @depth: depth of the tree
    @optimal: whether to find optimal tree or not (True or False)

    return:
    @best_result_weight: weight of the best tree found in current iteration
    '''
    def iterative_backtrack(self, state, best_result, best_result_weight, depth, optimal):
        state_stack = []
        row = 1
        pos = 0
        state_stack.append(pos)

        while True:
            # If there is no valid position, 2 cases:
            # a) if stack is empty, break and stop search
            # b) if stack is not empty, pop stack and set current position to next
            #    position backtrack to previous row
            while state_stack and pos >= self.network.nodes:
                pos = state_stack.pop()
                pos += 1
                state[len(state_stack) + 1] = -1
                row -= 1

            if not state_stack:
                break

            state[row] = pos
            # If there is a valid position push the position to stack, set current
            # position to 0 and move to next row
            if self.is_valid(state, row + 1, depth):
                state_stack.append(pos)
                pos = 0
                row += 1
            else:
                pos += 1
                state[row] = -1

            # If stack has size N, a solution is found,
            # pop stack, set current position to next position
            # backtrack to find next solution
            if row == len(state):
                result = deepcopy(state)
                self.post_process(result, depth)
                weight = self.compute_tree_weight(result, depth, True)

                # Save this spanning tree if it is highest weight tree found so far
                if weight > best_result_weight:
                    best_result_weight = weight
                    best_result = deepcopy(result)
                if not optimal:
                    break

                pos = state_stack.pop()
                pos += 1
                state[len(state_stack)] = -1
                row -= 1

        return best_result_weight, best_result
    # def iterative_backtrack(state, best_result, best_result_weight, depth, optimal):


    '''
    is_valid() - checks whether the state can form a valid spanning tree
    @state: leaves states
    @row: corresponding level in the tree
    @depth: tree depth

    desc - checks whether a given state forms a spanning tree that satisfies:
            1) balanced
            2) binary
            3) each edge in tree corresponds to link in network topology
            4) each edge in tree does not form self-loop

    return: True if valid, otherwise False
    '''
    def is_valid(self, state, row, depth):
        # At each level of tree, check whether edge:
        #   - corresponds to link in network topology
        #   - corresponds to self-loop
        for i in range(depth):
            stride = 1 << i
            for j in range(0, row - stride, 2 * stride):
                from_node = state[j]
                to_node = state[j + stride]
                if self.adjacency_matrix[from_node][to_node] == 0 and from_node != to_node:
                    return False

        # If we encounter an accelerator for the first time, increment found_vec
        # Otherwise, do nothing
        found = set()
        found_vec = np.zeros(self.network.nodes, dtype=int)
        for val in state:
            if val == -1:
                continue
            if val < self.network.nodes:
                if val not in found:
                    found.add(val)
                    found_vec[val] = 1
            else:
                return False

        # modifier is maximum number of repeats a single accelerator can take
        #   e.g. 5 accelerators in 3-level binary tree => one accelerator can repeat 3x
        #        acc0 acc0 acc0 acc0 acc1 acc2 acc3 acc4
        modifier = (1 << depth) - self.network.nodes
        num_found = len(found)

        # So we know we have an invalid state if we find
        #   - only 4 unique accelerators
        #   = 9 unique accelerators
        if row < self.network.nodes:
            if num_found > row or num_found < row - modifier:
                return False

        # If we are at last recursive level, we can apply a more stringent check:
        #   - if some accelerator is not found, then we are in invalid state
        elif row == len(state):
            for i in range(self.network.nodes):
                if found_vec[i] == 0:
                    return False

        return True
    # def is_valid(self, state, row, depth)


    '''
    poset_process() - get rid of redunant sends
    @result: encoded tree
    @depth: tree depth

    desc - This function takes a spanning tree encoded as state (result), which may have
           repeated accelerators representing NO-SENDs and converts it into a unique format.
           This has the effect of recognizing redundant sends, grouping them together,

           Initial result: [3 0 0 4 1 2 5 6]
           Final result:   [3 3 0 4 1 2 5 6]

           Initial:
                   3
               3     1
             3   0   1   5
           3 0 0 4 1 2 5 6    // accelerator 3 will make redundant send to accelerator 0

           Final:
                   3
               3     1
             3   0   1   5
           3 3 0 4 1 2 5 6    // accelerator 3 knows not to make redundant send to itself
    '''
    def post_process(self, result, depth):
        num_elements = self.network.nodes

        for level in range(depth-1, -1, -1):
            stride = 1 << level
            histogram_above = np.zeros(num_elements)
            for i in range(0, len(result), 2*stride):
                val = result[i]
                histogram_above[val] += 1
            histogram = np.zeros(num_elements)
            for i in range(0, len(result), stride):
                val = result[i]
                histogram[val] += 1

            for i in range(len(result) - stride, stride - 1, -2*stride):
                from_node = result[i]
                to_node = result[i-stride]
                if (histogram[from_node] > 1 or histogram_above[from_node] >= 1) and from_node != to_node:
                    result[i] = to_node
                    histogram[from_node] -= 1
    # def post_process(self, result, depth)


    '''
     compute_tree_weight() - sum the edge weights of the given spanning tree
     @result: encoded spanning tree
     @depth: tree depth
     @penalty: controls whether or not to apply penalties to tree
                - usually turned on when backtracking to get better solutions
                - usually turned off when outside the penalty to get weight of tree

    return: computed weight
    '''
    def compute_tree_weight(self, result, depth, penalty):
        num_elements = self.network.nodes
        weight = 0.0
        links_used = set()

        for i in range(depth):
            stride = 1 << i
            nodes_used = [False] * num_elements
            for j in range(0, len(result) - stride, 2*stride):
                from_node = result[j]
                to_node = result[j + stride]
                if from_node != to_node:
                    weight += self.adjacency_matrix[from_node][to_node]

                    # Penalize: (1) use of redundant edges in a single tree
                    #           (2) repeated use of a GPU in a signle tree at the same
                    #               level above the leaf level
                    if (from_node * num_elements + to_node) in links_used and penalty:
                        weight -= 100
                    links_used.add(from_node * num_elements + to_node)
                    links_used.add(to_node * num_elements + from_node)

                nodes_used[from_node] = True
                if i > 0 and nodes_used[to_node] and penalty:
                    weight -= 10
                nodes_used[to_node] = True

        return weight
    # def compute_tree_weight(self, result, depth, penalty)


    '''
    form_topology() - process result to form tree in topology
    @result: encoded spanning tree
    @root: root id of the tree under construction
    @depth: depth of the tree

    desc - Given a spanning tree encoded as result, which was convenient for performing
           backtracking, convert it to topology and scan in the classic "binary tree
           stored in an array" format. For binary trees scan is redundant, but this
           additional data structure leaves future generalization to k-radix trees.

           Initial result: [3 3 0 4 1 2 5 6]
           topology[root]:      [3 3 1 3 0 1 5 3 3 0 4 1 2 5 6]
           scan[root]:          [0 1 3 7 15]

           topology is stored in the classic "binary tree stored in an array" format
           e.g.    3
               3     1
             3   0   1   5
           3 3 0 4 1 2 5 6

    returns:
    @success: False if invalid tree in result, otherwise True
    '''
    def form_topology(self, result, root, depth):
        success = True
        failure = False

        for result_value in result:
            if result_value == -1:
                return failure

        self.scan[root].append(len(self.topology[root]))
        for i in range(depth, -1, -1):
            stride = 1 << i
            for j in range(0, len(result), stride):
                from_node = result[j]
                self.topology[root].append(from_node)
            self.scan[root].append(len(self.topology[root]))

        return success
    # def form_topology(self, result, root, depth)


    '''
    update_weight() - Apply penalty factor alpha to each link that is used by the spanning tree
    @root: root id of the tree
    '''
    def update_weight(self, root):
        num_elements = self.network.nodes

        for i in range(1, len(self.topology[root]) - 1, 2):
            parent = self.topology[root][i]
            child = self.topology[root][i + 1]
            if not (parent >= num_elements * num_elements or
                    child >= num_elements * num_elements) and parent != child:
                self.adjacency_matrix[parent][child] *= self.alpha
                self.adjacency_matrix[child][parent] *= self.alpha
    # def update_weight(self, root)


    '''
    find_best_move() - find the best element to swap
    @partition_temp: current partitions
    @D: difference of external cost and internal cost of each node
    @used: the set of nodes have been moved

    desc - Computes best 2 nodes a,b to swap given objective function:
               g = max_{a \in A, b \in B} D(a) + D(b) - 2*W(a,b)

           Optimization: Only need to look at upper triangular since weight
           matrix is symmetric.

    return:
    @g: score of moving/swapping a and b
    @a: candidate for the move
    @b: candidate for the move
    '''
    def find_best_move(self, partitions_temp, D, used):
        nrows = len(partitions_temp)
        g = 0
        a = -1
        b = -1
        for row in range(nrows):
            if partitions_temp[row] == 0 or row in used:
                continue
            for col in range(row + 1, nrows):
                if partitions_temp[col] == 0 or partitions_temp[row] == partitions_temp[col]:
                    continue

                cost = D[row] + D[col] - 2 * self.adjacency_matrix[row][col]
                if cost > g:
                    g = cost
                    a = row
                    b = col

        return g, a, b
    # def find_best_move(self, partitions_temp, D, used)


    '''
    find_best_edge() - Computes highest weighted edge a-b
    @partitions: partitions of the nodes
    @parent: as it means, which is a
    @to_cluster: the cluster to be searched

    desc - Contraints:
            - vertex a must be parent -vertex b must be in to_cluster
           Optimization: Only need to look at row a in matrix

    return:
    @candidates: candidate node(s) for b
    @g: weight of edge
    '''
    def find_best_edge(self, partitions, parent, to_cluster):
        nrows = len(partitions)
        row = parent
        g = 0
        candidates = [-1]
        for col in range(nrows):
            if col == row or partitions[col] != to_cluster:
                continue

            cost = self.adjacency_matrix[row][col]
            if cost > g:
                candidates = []
            if cost >= g:
                candidates.append(col)
                g = cost

        return candidates, g
    #def find_best_edge(self, partitions, parent, to_cluster)


    '''
    gemv() - Dense matrix-vector multiplication, vy = self.adjacency_matrix*x (no accumulate)
    @x: vector
    @y: result vector
    '''
    def gemv(self, x, y):
        nrows = len(x)
        for row in range(nrows):
            y[row] = 0
            for col in range(nrows):
                y[row] += self.adjacency_matrix[row][col] * x[col]
    # def gemv(self, x, y)


    '''
    ewisemult() - Element-wise multiplication between 2 dense vectors
                  w = w * alpha*u
    '''
    def ewisemult(self, u, alpha, w):
        nelem = len(u)
        for i in range(nelem):
            w[i] *= alpha * u[i]
    # def ewisemult(self, u, alpha, w)


    '''
    compute_depth() - compute depth of binary tree

    return:
    @depth: number of levels of binary tree
    '''
    def compute_depth(self):
        n = self.network.nodes
        for depth in range(self.mxnet_maxdepth):
            num = 2 << depth
            if n <= num:
                return depth + 1

        return 0
    # def compute_depth(self)


    '''
    get_root() - find root of a given color if found in roots
    @partitions: partitions of nodes
    @color: partition color
    @roots: set of roots

    return: -1 if it is not found, otherwise the found root
    '''
    def get_root(self, partitions, color, roots):
        for root in roots:
            if partitions[root] == color:
                return root

        return -1
    # def get_root(self, parititions, color, roots)


    '''
    get_child() - find the child in a particular partition for a parent
    @partitions: partitions of the nodes
    @color: partition color
    @parent: as it means

    return: the found child if found, otherwise -1
    '''
    def get_child(self, partitions, color, parent):
        for i, c in enumerate(partitions):
            if c == color and i != parent:
                return i

        return -1
    # def get_child(self, partitions, color, parent)


    '''
    print_topo() - utitlity for tree printing
    '''
    def print_topo(self, head, topo_row, scan_row):
        print(head + ':')
        depth = len(scan_row) - 1
        for row in range(depth):
            start = scan_row[row]
            end = scan_row[row + 1]
            output = ''
            temp = depth - row - 2
            if temp >= 0:
                terminate = (2 << temp) + 1
            else:
                terminate = 1
            for j in range(start, end):
                for k in range(terminate):
                    output += ' '
                output += str(topo_row[j])
            print(output)
    # def print_topo(self, head, topo_row, scan_row)


    '''
    link_conflict_detection() - detect the link conflicts of generated trees
    @verbose: True for detailed prints of detection process
    '''
    def link_conflict_detection(self, verbose=False):
        links_tree_map = []
        conflicts = []

        for root in range(self.network.nodes):
            depth = len(self.scan[root]) - 1
            for row in range(1, depth):
                if row > len(links_tree_map):
                    links_tree_map.append({})
                    conflicts.append(0)
                start = self.scan[root][row]
                end = self.scan[root][row + 1]
                for i in range(start, end):
                    if i % 2 == 0:
                        parent = self.topology[root][i - 1]
                        child = self.topology[root][i]
                        link = "{}->{}".format(child, parent)
                        if link in links_tree_map[row - 1]:
                            if verbose:
                                print('Conflict found for link {} between tree {} and tree {}'.format(link, links_tree_map[row - 1][link], root))
                            conflicts[row - 1] += 1
                        else:
                            links_tree_map[row - 1][link] = root

        print('Conflicts summary:')
        for i, num in enumerate(conflicts):
            print('  row {}: {}'.format(i + 1, num))
    # def link_conflict_detection(self, verbose=False)


    '''
    generate_conflict_trees() - generate conflict trees
    @filename: filename if want to generate dotfile
    '''
    def generate_conflict_trees(self, filename=None):
        colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a',
                '#ce1256', '#980043', '#67001f']

        self.conflict_trees = {}

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'

        ranks = {}

        max_depth = -1
        ranks[0] = []
        for root in range(self.network.nodes):
            self.conflict_trees[root] = []

            tree += '    /* tree {} */\n'.format(root)
            ranks[0].append('"{}-{}"'.format(root, root))

            depth = len(self.scan[root]) - 1
            if depth > max_depth:
                max_depth = depth

            for row in range(1, depth):
                start = self.scan[root][row]
                end = self.scan[root][row + 1]
                iteration = depth - row
                if end > start and not row in ranks.keys():
                    ranks[row] = []
                self.conflict_trees[root].append([])
                for i in range(start, end):
                    if i % 2 == 0:
                        parent = '"{}-{}"'.format(root, self.topology[root][i - 1])
                        child = '"{}-{}"'.format(root, self.topology[root][i])
                        if parent == child: # no self loop, redundant
                            continue
                        ranks[row].append(child)
                        tree += ''.join('    {} -> {} [ label="{}" ];\n'.format(child, parent, iteration))
                        self.conflict_trees[root][row - 1].append((self.topology[root][i], self.topology[root][i - 1]))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(max_depth):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(max_depth):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        if filename:
            f = open(filename, 'w')
            f.write(tree)
            f.close()
    # def generate_conflict_trees(self, filename=None)


    '''
    mxnet_schedule() - resolve conflicts and generate schedule trees
    @kary: build kary-trees
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @cross_level: allocate links from parent levels even some nodes at children
                  level are not scheduled, but should not have dependency on those
    @sort: Whether sort the trees for link allocation based on conflicts from
           last allocation iteration
    @verbose: print detailed info of tree construction process

    desc - schedule the communications in time steps by allocating links using bottom-up
           approach from leaves to root. Cross-level will schedule nodes at parent level
           even some nodes at children level have yet been scheduled. However, for a
           Bottom-Up approach, it does not help since the link requirments at levels closer
           to root is less (Note that higher level of trees are more sparse in terms of link
           usage).
    '''
    def mxnet_schedule(self, kary, alternate=True, cross_level=True, sort=True, verbose=False):
        assert kary > 1

        if verbose:
            print('Conflict trees:')

        # expand and linearize the trees for cross-level scheduling
        expanded_conflict_trees = {}
        pending_dependent_children = {}
        for root in range(self.network.nodes):
            expanded_conflict_trees[root] = []
            pending_dependent_children[root] = np.zeros(self.network.nodes)
            if verbose:
                print('     tree {}:'.format(root))
            for row in reversed(self.conflict_trees[root]):
                expanded_conflict_trees[root].extend(row)
                if verbose:
                    print('         {}'.format(row))
                for (child, parent) in row:
                    pending_dependent_children[root][parent] += 1
            if verbose:
                print('         expanded: {}'.format(expanded_conflict_trees[root]))


        # initialize empty trees
        self.trees = {}
        tree_nodes = {}
        tree_level_nodes = {}
        for root in range(self.network.nodes):
            self.trees[root] = []
            tree_nodes[root] = set()
            ## add the leave nodes since they can be scheduled all the time
            #for (child, parent) in self.conflict_trees[root][-1]:
            #    tree_nodes[root].add(child)
            tree_level_nodes[root] = {0: []}

        # tree construction
        num_trees = 0
        self.iterations = 0

        from_nodes = deepcopy(self.network.from_nodes)

        # keep track of number of new children added for the iteration, constrained by k-ary
        num_new_children = {}
        for root in range(self.network.nodes):
            num_new_children[root] = {}

        if verbose:
            print('iteration {}'.format(self.iterations))

        # sort the roots based on link conflicts during allocation
        sorted_roots = list(range(self.network.nodes))
        conflicts = [0] * self.network.nodes

        while num_trees < self.network.nodes:

            change = False

            #for root in range(self.network.nodes):
            for root in sorted_roots:
                if len(tree_nodes[root]) == self.network.nodes:
                    continue

                if cross_level: # take the whole tree
                    child_parent_row = expanded_conflict_trees[root]
                else:           # take only the highest level
                    child_parent_row = self.conflict_trees[root][-1]

                for (child, parent) in child_parent_row:
                    if verbose:
                        print(' try to add edge {}->{} to tree {}'.format(child, parent, root))
                    if child in from_nodes[parent]: # link available
                        if parent not in num_new_children[root].keys():
                            num_new_children[root][parent] = 0
                        # all dependent children of 'child' have been scheduled in eariler iteration
                        # and parent's readix < k in current iteration
                        if child not in tree_level_nodes[root][self.iterations] and \
                                pending_dependent_children[root][child] == 0 and \
                                num_new_children[root][parent] < kary - 1:
                            num_new_children[root][parent] += 1
                            assert child not in tree_nodes[root]
                            change = True
                            if verbose:
                                print(' -- add node {} to tree {}'.format(child, root))
                                print('    before: {}'.format(self.trees[root]))
                            tree_nodes[root].add(child)
                            from_nodes[parent].remove(child)
                            self.trees[root].append((child, parent, self.iterations))
                            tree_level_nodes[root][self.iterations].append(parent)
                            pending_dependent_children[root][parent] -= 1
                            if verbose:
                                print('    after : {}'.format(self.trees[root]))
                                print('    tree nodes: {}'.format(tree_nodes[root]))
                            if cross_level:
                                expanded_conflict_trees[root].remove((child, parent))
                            else:
                                self.conflict_trees[root][-1].remove((child, parent))
                            if alternate:
                                break
                        else:
                            if verbose:
                                if num_new_children[root][parent] == kary - 1:
                                    print(' ** reach kary {} for parent {}'.format(num_new_children[root][parent]+1, parent))
                                elif pending_dependent_children[root][child] > 0:
                                    print(' ** {} dependent child(ren) of {} not added yet'.format(
                                        pending_dependent_children[root][child], child))
                                else:
                                    assert child in tree_level_nodes[root][self.iterations]
                                    print(' ** child {} already added as parent in this iteration'.format(child))
                    else:
                        conflicts[root] += 1
                        if verbose:
                            print(' ** link {}->{} not avaliable'.format(child, parent))

                if not cross_level and len(self.conflict_trees[root][-1]) == 0:
                    self.conflict_trees[root].pop(-1)

                if len(tree_nodes[root]) == self.network.nodes - 1:
                    assert root not in tree_nodes[root]
                    tree_nodes[root].add(root)
                    if cross_level:
                        assert len(expanded_conflict_trees[root]) == 0
                    num_trees += 1
                    if verbose:
                        print('iteration {} - tree {} constructed: {}'.format(
                            self.iterations, root, self.trees[root]))

            if sort:
                #print('before sorting: {}'.format(sorted_roots))
                #print('conflicts: {}'.format(conflicts))
                sorted_roots = [root for _ , root in sorted(zip(conflicts, sorted_roots), reverse=True)]
                conflicts = [0] * self.network.nodes
                #print('after sorting: {}'.format(sorted_roots))

            if not change:
                from_nodes = deepcopy(self.network.from_nodes)
                self.iterations += 1
                if verbose:
                    print('iteration {}'.format(self.iterations))

                # reset for new iteration
                for root in range(self.network.nodes):
                    tree_level_nodes[root][self.iterations] = []
                    num_new_children[root] = {}

        # verify that there is no link conflicts
        for root in range(self.network.nodes):
            for i in range(root + 1, self.network.nodes):
                intersection = set(self.trees[root]) & set(self.trees[i])
                if len(intersection) != 0:
                    print('tree {} and tree {} have link conflicts {}'.format(root, i, intersection))
                    print('tree {}: {}'.format(root, self.trees[root]))
                    print('tree {}: {}'.format(i, self.trees[i]))
                    exit()

        for root in range(self.network.nodes):
            tree = self.trees[root]
            self.trees[root] = []
            for child, parent, iteration in reversed(tree):
                self.trees[root].append((child, parent, self.iterations - iteration))

        self.iterations += 1
    #def mxnet_schedule(self, kary, alternative=True, cross_level=True, verbose=False)


    '''
    topdown_schedule() - resolve conflicts and generate trees
    @kary: build kary-trees
    @alternate: Ture - allocate the links by alternating trees every allocation
                False - allocating links for one tree as much as possble
    @cross_level: allocate links from children levels even some nodes at parent
                  level are not scheduled, but should not have dependency on those
    @sort: Whether sort the trees for link allocation based on conflicts from
           last allocation iteration
    @verbose: print detailed info of tree construction process

    desc - schedule the communications in time steps by allocating links using top-down
           approach from root to leaves. Cross-level tends to schedule higher potential
           link conflicts at lower (closer to leaves) levels earilier, reducing the chance
           of conflicts (Note that higher level of trees are more sparse in terms of link
           usage).
    '''
    def topdown_schedule(self, kary, alternate=True, cross_level=True, sort=True, verbose=False):
        assert kary > 1

        if verbose:
            print('Conflict trees:')

        # expand and linearize the trees for cross-level scheduling
        expanded_conflict_trees = {}
        for root in range(self.network.nodes):
            expanded_conflict_trees[root] = []
            if verbose:
                print('     tree {}:'.format(root))
            for row in self.conflict_trees[root]:
                expanded_conflict_trees[root].extend(row)
                if verbose:
                    print('         {}'.format(row))
            if verbose:
                print('         expanded: {}'.format(expanded_conflict_trees[root]))


        # initialize empty trees
        self.trees = {}
        tree_nodes = {}
        tree_level_nodes = {}
        for root in range(self.network.nodes):
            self.trees[root] = []
            tree_nodes[root] = [root]
            tree_level_nodes[root] = {0: []}

        # tree construction
        num_trees = 0
        self.iterations = 0

        from_nodes = deepcopy(self.network.from_nodes)

        # keep track of number of new children added for the iteration, constrained by k-ary
        num_new_children = {}
        for root in range(self.network.nodes):
            num_new_children[root] = {}

        if verbose:
            print('iteration {}'.format(self.iterations))

        # sort the roots based on link conflicts during allocation
        sorted_roots = list(range(self.network.nodes))
        conflicts = [0] * self.network.nodes

        while num_trees < self.network.nodes:

            change = False

            #for root in range(self.network.nodes):
            for root in sorted_roots:
                if len(tree_nodes[root]) == self.network.nodes:
                    continue

                if cross_level: # take the whole tree
                    child_parent_row = expanded_conflict_trees[root]
                else:           # take only the highest level
                    child_parent_row = self.conflict_trees[root][0]

                for (child, parent) in child_parent_row:
                    if verbose:
                        print(' try to add edge {}->{} to tree {}'.format(child, parent, root))
                    if child in from_nodes[parent]: # link available
                        if parent not in num_new_children[root].keys():
                            num_new_children[root][parent] = 0
                        # parent has been scheduled in earilier iteration (first two checks)
                        # and parent's readix < k in current iteration
                        if parent not in tree_level_nodes[root][self.iterations] and \
                                parent in tree_nodes[root] and \
                                num_new_children[root][parent] < kary - 1:
                            num_new_children[root][parent] += 1
                            assert child not in tree_nodes[root]
                            change = True
                            if verbose:
                                print(' -- add node {} to tree {}'.format(child, root))
                                print('    before: {}'.format(self.trees[root]))
                            tree_nodes[root].append(child)
                            from_nodes[parent].remove(child)
                            self.trees[root].append((child, parent, self.iterations))
                            tree_level_nodes[root][self.iterations].append(child)
                            if verbose:
                                print('    after : {}'.format(self.trees[root]))
                                print('    tree nodes: {}'.format(tree_nodes[root]))
                            if cross_level:
                                expanded_conflict_trees[root].remove((child, parent))
                            else:
                                self.conflict_trees[root][0].remove((child, parent))
                            if alternate:
                                break
                        else:
                            #conflicts[root] += 1
                            if verbose:
                                if num_new_children[root][parent] == kary - 1:
                                    print(' ** reach kary {} for parent {}'.format(num_new_children[root][parent]+1, parent))
                                elif parent not in tree_nodes[root]:
                                    print(' ** parent {} not added yet'.format(parent))
                                else:
                                    assert parent in tree_level_nodes[root][self.iterations]
                                    print(' ** child {} already added in this iteration'.format(child))
                    else:
                        conflicts[root] += 1
                        if verbose:
                            print(' ** link {}->{} not avaliable'.format(child, parent))

                if not cross_level and len(self.conflict_trees[root][0]) == 0:
                    self.conflict_trees[root].pop(0)

                if len(tree_nodes[root]) == self.network.nodes:
                    if cross_level:
                        assert len(expanded_conflict_trees[root]) == 0
                    num_trees += 1
                    if verbose:
                        print('iteration {} - tree {} constructed: {}'.format(
                            self.iterations, root, self.trees[root]))

            if sort:
                #print('before sorting: {}'.format(sorted_roots))
                #print('conflicts: {}'.format(conflicts))
                sorted_roots = [root for _ , root in sorted(zip(conflicts, sorted_roots), reverse=True)]
                conflicts = [0] * self.network.nodes
                #print('after sorting: {}'.format(sorted_roots))

            if not change:
                from_nodes = deepcopy(self.network.from_nodes)
                self.iterations += 1
                if verbose:
                    print('iteration {}'.format(self.iterations))

                # reset for new iteration
                for root in range(self.network.nodes):
                    tree_level_nodes[root][self.iterations] = []
                    num_new_children[root] = {}

        # verify that there is no link conflicts
        for root in range(self.network.nodes):
            for i in range(root + 1, self.network.nodes):
                intersection = set(self.trees[root]) & set(self.trees[i])
                if len(intersection) != 0:
                    print('tree {} and tree {} have link conflicts {}'.format(root, i, intersection))
                    print('tree {}: {}'.format(root, self.trees[root]))
                    print('tree {}: {}'.format(i, self.trees[i]))
                    exit()

        self.iterations += 1
    #def topdown_schedule(self, kary, alternative=True, cross_level=True, verbose=False)


def test(args):
    dimension = args.dimension
    nodes = dimension * dimension
    network = networks.Torus(nodes, dimension)
    network.build_graph()

    kary = args.kary
    allreduce = MXNetTreeAllreduce(network)
    allreduce.backtrack = args.backtrack
    begin_seed = 100
    end_seed = 101
    better = {}
    same = {}
    worse = {}
    comparison_distribution = {'Better': 0, 'Same': 0, 'Worse': 0}
    total_iterations = 0
    total_sort_iterations = 0
    num_seeds = 0
    # NOTE: It seems sorted won't help much due to random picks to break ties in the during KL algorithm.
    #       Sometimes better and sometimes worse, most of the time are same. Seed 47 run forever, buggy!
    #       For example, random seed 8 makes it worse and random seed 9 makes it better. The hypothesis
    #       is that the limitation is inherent in the decoupling of tree construction and scheduling.
    for seed in range(begin_seed, end_seed):
        #print('seed: {}'.format(seed))
        if seed == 47:
            continue
        random.seed(seed)
        allreduce.compute_trees(kary, alternate=True, sort=False, verbose=False)
        if args.gendotfile:
            allreduce.generate_trees_dotfile('mxnettree.dot')
        iterations = allreduce.iterations
        if allreduce.backtrack:
            print('MXNetTreeAllreduce takes {} iterations'.format(allreduce.iterations))
            continue
        allreduce.mxnet_schedule(kary, alternate=True, sort=True, verbose=False)
        if args.gendotfile:
            allreduce.generate_trees_dotfile('mxnettree_sort.dot')
        sort_iterations = allreduce.iterations
        #print('MXNetTreeAllreduce (sorted) takes {} iterations'.format(allreduce.iterations))
        if iterations > sort_iterations:
            compare = 'Better'
            diff = iterations - sort_iterations
            if diff in better.keys():
                better[diff] += 1
            else:
                better[diff] = 1
        elif iterations == sort_iterations:
            compare = 'Same'
            if iterations in same.keys():
                same[iterations] += 1
            else:
                same[iterations] = 1
        else:
            compare = 'Worse'
            diff = sort_iterations - iterations
            if diff in worse.keys():
                worse[diff] += 1
            else:
                worse[diff] = 1
        comparison_distribution[compare] += 1
        num_seeds += 1
        total_iterations += iterations
        total_sort_iterations += sort_iterations
        print('Seed {}: MXNetTreeAllreduce takes {} iterations (no sort), and {} iterations (sort), {}'.format(
            seed, iterations, sort_iterations, compare))
    if num_seeds > 1:
        print('Comparison distribution: {}'.format(comparison_distribution))
        print('Iteration distribution for Same: {}'.format(same))
        print('Iteration difference for Better: {}'.format(better))
        print('Iteration difference for Worse: {}'.format(worse))
        print('Average iterations: {} (no sort), and {} (sort)'.format(
            total_iterations / num_seeds, total_sort_iterations / num_seeds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', default=4, type=int,
                        help='network dimension, default is 4')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary tree, default is 2 (binary)')
    parser.add_argument('--backtrack', default=False, action='store_true',
                        help='use backtracking only, default is False')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')

    args = parser.parse_args()

    test(args)

import numpy as np
import random
from copy import deepcopy
import networks


def print_topo(head, topo_row, scan_row):
    print(head + ':')
    depth = len(scan_row) - 1
    print('depth: {}'.format(depth))
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
# def print_topo(head, topo_row, scan_row)


'''
 #brief Uses BFS to find whether undirected graph is connected or not given its
 adjacency matrix
 Note: only consider matrix values > 1, because we care about whether it is
 connected using only NVLink connections
'''
def is_connected(adjacency_matrix, num_gpus):
    source = 0
    visited = [False for i in range(num_gpus)]
    work_list = []

    work_list.append(source)
    visited[source] = True
    while work_list:
        curr = work_list.pop()

        for i in range(num_gpus):
            neighbor = adjacency_matrix[curr][i]
            if i != curr and neighbor > 1 and visited[i] == False:
                visited[i] = True
                work_list.append(i)

    for i in range(num_gpus):
        if visited[i] == False:
            return False

    return True
# def is_connected(adjacency_matrix, num_gpus)


'''
 #brief Dense matrix-vector multiplication
 Assume: matrix is square
   y = A*x (no accumulate)
'''
def gemv(A, x, y):
    nrows = len(x)
    for row in range(nrows):
        y[row] = 0
        for col in range(nrows):
            y[row] += A[row][col] * x[col]
# def gemv(A, x, y)


'''
 #brief Element-wise multiplication between 2 dense vectors
   w = w * alpha*u
'''
def ewisemult(u, alpha, w):
    nelem = len(u)
    for i in range(nelem):
        w[i] *= alpha * u[i]
# def ewisemult(u, alpha, w)


'''
 #brief Computes best 2 nodes a,b to swap given objective function:
   g = max_{a \in A, b \in B} D(a) + D(b) - 2*W(a,b)

 Optimization: Only need to look at upper triangular since weight matrix is
 symmetric
'''
def find_best_move(adjacency_matrix, partitions_temp, D, used):
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

            cost = D[row] + D[col] - 2 * adjacency_matrix[row][col]
            if cost > g:
                g = cost
                a = row
                b = col

    return g, a, b
# def find_best_move(adjacency_matrix, partitions_temp, D, used)


'''
 #brief Performs partition on each existing partition in graph W if partition has
 more than 4 elements in it
 #param stop returns true if no partitions with >=4 elements found
             returns false otherwise
 #param cluster_pairs stores the mapping that tells us which 2 clusters are
        the output of partitioning one large cluster
'''
def kernighan_lin(adjacency_matrix, partitions, num_partitions, cluster_pairs):
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
            #random.shuffle(cluster_list)
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
                gemv(adjacency_matrix, partitions_temp, D)
                ewisemult(partitions_temp, -1.0, D)

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
                    g, a, b = find_best_move(adjacency_matrix, partitions_temp, D, used)
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
                    gemv(adjacency_matrix, partitions_temp, D)
                    ewisemult(partitions_temp, -1.0, D)
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
# def kernighan_lin(adjacency_matrix, partitions, num_partitions, cluster_pairs)


'''
 #brief Returns root of a given color if found in roots
        Returns -1 if it is not found
'''
def get_root(partitions, color, roots):
    for root in roots:
        if partitions[root] == color:
            return root

    return -1
# def get_root(parititions, color, roots)


'''
 #brief Returns root of a given color if found in roots
        Returns -1 if it is not found
'''
def get_child(partitions, color, parent):
    for i, c in enumerate(partitions):
        if c == color and i != parent:
            return i

    return -1
# def get_child(partitions, color, parent)


'''
 Computes highest weighted edge a-b

 Contraints:
  -vertex a must be parent
  -vertex b must be in to_cluster

 @output: b is vector of candidates if a tie happens
          g is weight of edge
 Optimization: Only need to look at row a in matrix
'''
def find_best_edge(adjacency_matrix, partitions, parent, to_cluster):
    nrows = len(partitions)
    row = parent
    g = 0
    candidates = [-1]
    for col in range(nrows):
        if col == row or partitions[col] != to_cluster:
            continue

        cost = adjacency_matrix[row][col]
        if cost > g:
            candidates = []
        if cost >= g:
            candidates.append(col)
            g = cost

    return candidates, g
#def find_best_edge(adjacency_matrix, partitions, parent, to_cluster)


'''
 Given a vector of color pairs, appends to binary tree matrix topo
 @input:  adjacency_matrix gives the link topology
          partitions gives the result of KL partitioning
          cluster_pairs gives pairing between clusters, an edge is found
                        between each pairing
          roots gives source vertices
          #gen gives random number generation to break ties
 @output: cluster_pairs
          topo_row says where new edges are appended to
          scan_row says where we should start looking for topo_row
'''
def kl_generate_binary_tree(adjacency_matrix, partitions, cluster_pairs, roots, topo_row, scan_row):
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
            parent = get_root(partitions, color, roots)
            if parent == -1:
                return True
            child = get_child(partitions, color, parent)
        elif second == -1:
            color = first
            parent = get_root(partitions, color, roots)
            if parent == -1:
                return True
            child = parent
        else:
            # Root must exist in either first or second element of pair
            color = first
            parent = get_root(partitions, color, roots)
            if parent == -1:
                color = second
                parent = get_root(partitions, color, roots)

            from_cluster = color
            to_cluster = first
            if from_cluster == first:
                to_cluster = second

            candidates, weight = find_best_edge(adjacency_matrix, partitions, parent, to_cluster)

            # if no candidates
            if candidates[0] != -1:
                #random.shuffle(candidates)
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
# def kl_generate_binary_tree(adjacency_matrix, partitions, cluster_pairs, roots, topo_row, scan_row)

'''
 @input: n is the number of nodes in a balanced binary tree
 @output: returns how many levels of binary tree there are
'''
def compute_depth(n):
    for depth in range(n):
        num = 2 << depth
        if n <= num:
            return depth + 1

    return 0

'''
 Checks whether a given state forms a spanning tree that satisfies:
   -balanced
   -binary
   -each edge in tree corresponds to link in network topology
   -each edge in tree does not form self-loop
'''
def is_valid(adjacency_matrix, state, num_elements, row, depth):
    # At each level of tree, check whether edge:
    #   - corresponds to link in network topology
    #   - corresponds to self-loop
    for i in range(depth):
        stride = 1 << i
        for j in range(0, row - stride, 2 * stride):
            from_node = state[j]
            to_node = state[j + stride]
            if adjacency_matrix[from_node][to_node] == 0 and from_node != to_node:
                return False

    # If we encounter an accelerator for the first time, increment found_vec
    # Otherwise, do nothing
    found = set()
    found_vec = np.zeros(num_elements)
    for val in state:
        if val == -1:
            continue
        if val < num_elements:
            if val not in found:
                found.add(val)
                found_vec[val] = 1
        else:
            return False

    # modifier is maximum number of repeats a single accelerator can take
    #   e.g. 5 accelerators in 3-level binary tree => one accelerator can repeat 3x
    #        acc0 acc0 acc0 acc0 acc1 acc2 acc3 acc4
    modifier = (1 << depth) - num_elements
    num_found = len(found)

    # So we know we have an invalid state if we find
    #   - only 4 unique accelerators
    #   = 9 unique accelerators
    if row < num_elements:
        if num_found > row or num_found < row - modifier:
            return False

    # If we are at last recursive level, we can apply a more stringent check:
    #   - if some accelerator is not found, then we are in invalid state
    elif row == len(state):
        for i in range(num_elements):
            if found_vec[i] == 0:
                return False

    return True
# def is_valid(adjacency_matrix, state, num_elements, row, depth)


'''
 This function takes a spanning tree encoded as state (result), which may have
 repeated GPUs representing NO-SENDs and converts it into a unique format.
 This has the effect of recognizing redundant sends, grouping them together,
 so that the Reduce call knows not to perform a CopyFromTo.

 Initial result: [3 0 0 4 1 2 5 6]
 Final result:   [3 3 0 4 1 2 5 6]

 Initial:
         3
     3     1
   3   0   1   5
 3 0 0 4 1 2 5 6    // GPU3 will make redundant send to GPU0

 Final:
         3
     3     1
   3   0   1   5
 3 3 0 4 1 2 5 6    // GPU3 knows not to make redundant send to itself
'''
def post_process(result, num_elements, depth):
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
# def post_process(result, num_elements, depth)


'''
 Given a spanning tree encoded as a state (result) and weight of each edge
 in the link topology graph, compute its weight.
 @input: penalty controls whether or not penalties are applied to tree
         -usually turned on when backtracking to get better solutions
         -usually turned off when outside the penalty to get weight of tree
'''
def compute_tree_weight(adjacency_matrix, result, num_elements, depth, penalty):
    weight = 0.0
    links_used = set()

    for i in range(depth):
        stride = 1 << i
        nodes_used = [False] * num_elements
        for j in range(0, len(result) - stride, 2*stride):
            from_node = result[j]
            to_node = result[j + stride]
            if from_node != to_node:
                weight += adjacency_matrix[from_node][to_node]

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
# def compute_tree_weight(adjacency_matrix, result, num_elements, depth, penalty)


'''
 #brief Given a spanning tree encoded as result, which was convenient for performing
 backtracking, convert it topology_ and scan_ in the classic "binary tree
 stored in an array" format. For binary trees scan_ is redundant, but this
 additional data structure leaves future generalization to k-radix trees.

 Initial result: [3 3 0 4 1 2 5 6]
 topology_:      [3 3 1 3 0 1 5 3 3 0 4 1 2 5 6]
 scan_:          [0 1 3 7 15]

 topology_ is stored in the classic "binary tree stored in an array" format
 e.g.    3
     3     1
   3   0   1   5
 3 3 0 4 1 2 5 6

 Returns false if invalid tree in result
 Otherwise returns true
'''
def form_topology(result, topo_row, scan_row, depth):
    for result_value in result:
        if result_value == -1:
            return False

    scan_row.append(len(topo_row))
    for i in range(depth, -1, -1):
        stride = 1 << i
        for j in range(0, len(result), stride):
            from_node = resullt[j]
            topo_row.append(from_node)
        scan_row.append(len(topo_row))

    # Inseert at the end, result vector
    topo_row.append(result)
    scan_row.append(len(topo_row))

    return True
# def form_topology(result, topo_row, scan_row, depth)


'''
 #brief Recursive function that finds a spanning tree, which fulfills the following
 conditions:
   -balanced
   -binary
   -maximum weight
'''
def recursive_backtrack(adjacency_matrix, state, best_result, best_result_weight, row, num_elements, dpeth, optimal):
    if row == len(state):
        result = deepcopy(state)
        post_process(result, num_elements, depth)
        weight = compute_tree_weight(adjacency_matrix, result, num_elements, depth, True)

        # Save this spanning tree if it is heighest weight tree found sofar
        if weight > best_result_weight:
            best_result_weight = weight
            best_result = result

        return (not optimal), best_result_weight

    # If not last recursive level, try to find valid tree for next level
    stop = False
    for j in range(num_elements):
        state[row] = j
        if is_valid(adjacency_matrix, state, num_elements, row + 1, depth):
            stop, best_result_weight = recursive_backtrack(adjacency_matrix, state, best_result_weight,
                    row + 1, num_elements, depth, optimal)

        state[row] = -1
        if stop:
            return stop, best_result_weight

    return stop, best_result_weight
# def recursive_backtrack(adjacency_matrix, state, best_result, best_result_weight, row, num_elements, dpeth, optimal)


def iterative_backtrack(adjacency_matrix, state, best_result, best_result_weight, num_elements, depth, optimal):
    state_stack = []
    row = 1
    pos = 0
    state_stack.append(pos)

    while True:
        # If there is no valid position, 2 cases:
        # a) if stack is empty, break and stop search
        # b) if stack is not empty, pop stack and set current position to next
        #    position backtrack to previous row
        while state_stack and pos >= num_elements:
            pos = state_stack.pop()
            pos += 1
            state[len(state_stack) + 1] = -1
            row -= 1

        if not state_stack:
            break

        state[row] = pos
        # If there is a valid position push the position to stack, set current
        # position to 0 and move to next row
        if is_valid(adjacency_matrix, state, num_elements, row + 1, depth):
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
            post_process(result, num_elements, depth)
            weight = compute_tree_weight(adjacency_matrix, result, num_elements, depth, True)

            # Save this spanning tree if it is highest weight tree found so far
            if weight > best_result_weight:
                best_result_weight = weight
                #best_result.clear()
                best_result = deepcopy(result)
            if not optimal:
                break

            pos = state_stack.pop()
            pos += 1
            state[len(state_stack)] = -1
            row -= 1

    return best_result_weight
# def iterative_backtrack(adjacency_matrix, state, best_result, best_result_weight, num_elements, depth, optimal):


'''
 #brief Apply penalty factor alpha to each link in link topology graph that is used
 by the spanning tree
'''
def update_weight(adjacency_matrix, topo_row, num_elements, alpha):
    for i in range(1, len(topo_row) - 1, 2):
        parent = topo_row[i]
        child = topo_row[i + 1]
        if not (parent >= num_elements * num_elements or
                child >= num_elements * num_elements) and parent != child:
            adjacency_matrix[parent][child] *= alpha
            adjacency_matrix[child][parent] *= alpha
# def update_weight(adjacency_matrix, topo_row, num_elements, alpha)


'''
 #brief Do brute-force backtracking approach if Kernighan-Lin fails to find a binary
 tree of height Log P.

 Constraints:
 1) minimize depth (balance)
 2) maximize edge weight
 3) tree is binary
'''
def backtrack_generate_binary_tree(adjacency_matrix, num_elements, root, topo_row, scan_row):
    # Clear before starting
    topo_row.clear()
    scan_row.clear()

    # Compute depth
    # num_elemennts: depth
    # 5: 3 8
    # 6: 3 8
    # 7: 3 8
    # 8: 3 8
    # 9: 4 16
    depth = compute_depth(num_elements)
    depth_leaves = 1 << depth

    # State vector
    # -1 means unplaced
    state = -np.ones(depth_leaves, dtype=int)
    result = -np.ones(depth_leaves)
    result_weight = 0

    # Place root and try all combinations
    state[0] = root

    # Seek optimal solution until depth <= 3 i.e. 8 accelerators
    # For larger number of accelerators, settle for first tree found (non-optimal),
    # but this saves a lot of runtime since Backtrack is exponential time
    if depth <= 3:
        result_weight = iterative_backtrack(adjacency_matrix, state, result, result_weight, num_elements, depth, True)
    else:
        result_weight = iterative_backtrack(adjacency_matrix, state, result, result_weight, num_elements, depth, False)

    return form_topology(result, topo_row, scan_row, depth)
# def backtrack_generate_binary_tree(adjacency_matrix, num_elements, root, topo_row, scan_row)


'''
 #brief ComputeTreesFromRoot does the same thing as ComputeTrees, with the only
 exception being it will do it from a fixed GPU as root
'''
def compute_trees_from_root(adjacency_matrix, num_elements, root, alpha, backtrack, topo, scan):
    num_partitions = 1

    # Initialize partition array to indicate which partition each element
    # belongs to, beginning with 0
    partitions = np.zeros(num_elements, dtype=int)

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

    while (not backtrack) and (not stop or reset):
        if reset:
            cluster_pairs.clear()
            partitions_temp = deepcopy(partitions)
            num_partitions_temp = num_partitions
            roots_temp = deepcopy(roots)
            topo_temp = deepcopy(topo)
            scan_temp = deepcopy(scan)

        # Run Kernighan-Lin to generate partition
        stop, num_partitions_temp = kernighan_lin(adjacency_matrix, partitions_temp, num_partitions_temp, cluster_pairs)

        # Use partitions found and a given root to find best inter-cluster edge
        # for each pair of clusters, and returns them as roots of next cluster.
        # If reset is true, then rewind back to previous clustering
        reset = kl_generate_binary_tree(adjacency_matrix, partitions_temp, cluster_pairs, roots_temp, topo_temp, scan_temp)

        if reset:
            level += 1
        if level > 10:
            break

    success = True
    if reset:
        print("No valid tree found from root {}, try backtracking".format(root))
        success = backtrack_generate_binary_tree(adjacency_matrix, num_elements, root, topo, scan)
    else:
        topo.clear()
        topo = deepcopy(topo_temp)
        scan.clear()
        scan = deepcopy(scan_temp)
        scan.append(len(topo))

    if success:
        update_weight(adjacency_matrix, topo, num_elements, alpha)
        return topo, scan
    else:
        print("No valid binary tree found from root {} using backtracking".format(root))
# def compute_trees_from_root(adjacency_matrix, num_elements, root, alpha, backtrack, topo, scan)


'''
 #brief ComputeTrees computes balanced binary spanning trees of maximum edge weight
 given a link topology graph stored in adjacency matrix format
 #param W is the link topology matrix
 #param num_elements is the number of GPUs
 #param alpha is the link usage penalty
 #param backtrack is whether or not we use backtracking to generate trees
 #param topo stores the trees generated
 #param scan stores the start of each level of each tree
'''
def compute_trees(adjacency_matrix, num_elements, alpha, backtrack, log_tree = False):
    adjacency_matrix_cp = deepcopy(adjacency_matrix)

    topo = []
    scan = []
    for i in range(num_elements):
    #for i in range(1):
        topo.append([])
        scan.append([])
        topo[i].append(i)
        scan[i].append(0)
        print('before: {}'.format(scan[i]))
        topo[i], scan[i] = compute_trees_from_root(adjacency_matrix_cp, num_elements, i, alpha, backtrack, topo[i], scan[i])
        print('after: {}'.format(scan[i]))

    # NOTE: must sum up adjacent weight matrix to show link usage before readjusting repo
    # from 0, 1, ..., n_gpus format to dev_id format, which will cause segfault
    adj = np.zeros(adjacency_matrix.shape)
    for row in range(num_elements):
        for col in range(1, len(topo[0]), 2):
            from_node = min(topo[row][col], topo[row][col + 1])
            to_node = max(topo[row][col], topo[row][col + 1])
            if from_node != to_node:
                adj[from_node][to_node] += 1
                adj[to_node][from_node] += 1

    if log_tree:
        for i in range(num_elements):
            print_topo("Tree {}".format(i), topo[i], scan[i])
            #print("Tree {}:".format(i))
            #depth = len(scan[i]) - 1
            #for row in range(depth):
            #    start = scan[i][row]
            #    end = scan[i][row + 1]
            #    print(scan)
            #    output = ""
            #    for j in range(start, end):
            #        for k in range((2 << (depth - row - 2)) + 1):
            #            output += " "
            #        output += str(topo[j])
            #    print(output)

        print("W:")
        print(adjacency_matrix)
        print("Links:")
        print(adj)
# def compute_trees(adjacency_matrix, num_elements, alpha, backtrack, log_tree = False)


def test():
    network = networks.Torus(nodes=16, dimension=4)
    network.build_graph()
    compute_trees(network.adjacency_matrix, 16, 0.7, False, True)


if __name__ == '__main__':
    test()

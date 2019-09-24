import numpy as np
import random
from copy import deepcopy

def kernighan_lin(weight_matrix, partitions, num_partitions, cluster_pairs):
    histogram = np.zeros(num_partitions)
    partitions_temp = np.zeros(len(partitions))
    partitions_temp2 = np.zeros(len(partitions))
    D = np.zeros(len(partitions))
    D_temp = np.zeros(len(paritions))
    cluster_pairs = []

    # Step 0: For every partition, determine if it can be partitioned further
    #   First od a histogram of each partiotion
    for partition in partitions:
        histogram[partition] += 1

    stop = True
    for color in range(len(histogram)):
        partition_size = histogram[color]
        # Save cluster in preparation for push to topo in GenerateBianryTree()
        if partition_size <= 2:
            cluster_pairs.append((color, -partition_size))
        else:
            # Do Kernighan-Lin if clustering is necessary
            stop = False

            # Step 1: If it has more than 4 elements, we can partition further.
            #   Assign random balaned partition of it
            #   -balanced is more important than random, so allocate first half to A
            #   and rest to B
            first_partition = 0
            target_partition = partition_size // 2
            cluster_list = []

            for i, partition in enumerate(partitions):
                if partition == color:
                    cluster_list.append(i)
                else:
                    p_temp[i] = 0

            # Step 1b: Shuffle using random generator
            cluster_list = random.shuffle(cluster_list)
            for cluster in cluster_list:
                if first_partition < target_partition:
                    p_temp[cluster] = 1
                    first_partition += 1
                else:
                    p_temp[cluster] = -1

            # Step 2: Iterate Kernighan-Lin until convergence
            g_max = 0
            g_k = -1
            count = 0
            while True:
                count += 1
                p_temp2 = deepcopy(p_temp)

                # a) Compute difference between external and internal costs of all
                #    elements of vector D
                for row in range(len(paritions)):
                    D[row] = 0
                    for col in range(len(paritions)):
                        D[row] += weight_matrix[row][col] * partitions_temp[col]
                    D[row] *= -1.0 * partitions_temp[row]

                # av and bv are used to hold candidates for moving
                # gv stores the score associated with move
                av = []
                bv = []
                gv = []

                # used stores the ones have been moved
                used = []

                nrows = partition_size // 2
                for iteration in range(nrows):
                    # b) Find best move by looking through upper triangular of weight matrix
                    g = 0
                    a = -1
                    b = -1
                    for row in range(len(partitions_temp)):
                        if partitions_temp[row] == 0 or row in used:
                            continue
                        for col in range(row+1, nrows):
                            if partitions_temp[col] == 0 or partitions_temp[row] == partitions_temp[col]:
                                continue

                            cost = D[row] + D[col] - 2 * weight_matrix[row][col]
                            if cost > g:
                                g = cost
                                a = row
                                b = col
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
                used.append(a)
                used.append(b)

                # e) Update D usig partitiion_temp
                for row in range(len(paritions)):
                    D[row] = 0
                    for col in range(len(paritions)):
                        D[row] += weight_matrix[row][col] * partitions_temp[col]
                    D[row] *= -1.0 * partitions_temp[row]
                D[a] = 0
                D[b] = 0

                # 3) Find when to stop by doing linear scan through gv
                #    Recompute score g_max
                for k in range(len(gv)):
                    if k > 0:
                        gv[k] += gv[k - 1]
                    if gv[k] > g_max:
                        g_max = gv[k]
                        g_k = k + 1

                # 4) If move is "good", commit moves by updating partitions_temp and partitions_temp2
                #    Otherwise, rollback changes to partitions_temp2
                if g_max > 0:
                    for i in range(g_k):
                        a = av[i]
                        b = bv[i]
                        temp = partitions_temp2[a]
                        partitions_temp2[a] = parition_temp2[b]
                        parition_temp2[b] = temp
                parition_temp = deepcopy(partitions_temp2)

                if g_max == 0 || count > len(partitions):
                    break

            # 5) Update P using partitions_temp
            moves = 0
            for i in range(len(paritions)):
                if partitions_temp[i] == -1:
                    partitions[i] = num_partitions
                    moves += 1
            cluster_pairs.append((color, num_partitions))

            num_partitions += 1

    return stop, num_partitions


def get_root(partitions, color, roots):
    for root in roots:
        if partitions[root] == color:
            return root

    return -1


def get_child(partitins, color, parent):
    for i, c in enumerate(partitions):
        if c == color and i != parent:
            return i

    return -1


def kl_generate_binary_tree(weight_matrix, paritions, cluster_pairs, roots, topo, scan):
    new_roots = []
    new_topo = {}
    reset = False

    for i, (first, second) in enumerate(cluster_pairs):
        if i == 0:
            scan.append(len(topo))
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

            candidates, weight = find_best_edge(weight_matrix, partitions, parent, to_cluster)

            if candidates[0] != -1:
                candidates = random.shuffle(candidates)
                child = candidates[0]

            if child == -1:
                reset = True
                return reset
            else:
                new_roots.append(parent)
                new_roots.append(child)

        new_topo[parent] = child

    depth = len(scan)
    start = scan[depth - 2]
    end = scan[depth - 1]

    for i in range(start, end):
        parent = topo[i]
        child = None

        # If not first, check previous level whether or not we are encountering
        # this root for the first time in this level of the tree
        if i != start and parent == topo[i - 1]:
            child = parent
        else:
            child = new_topo[parent]
        topo.append(parent)
        topo.append(child)

    cluster_pairs.clear()
    roots.clear()
    roots.append(new_roots)

    return reset


def compute_depth(n):
    for depth in range(n):
        num = 2 << depth
        if n <= num:
            return depth + 1

    return 0


def is_valid(weight_matrix, state, num_elements, row, depth):
    # At each level of tree, check whether edge:
    #   - corresponds to link in network topology
    #   - corresponds to self-loop
    for i in range(depth):
        stride = 1 << i
        for j in range(0, row, 2 * stride):
            from_node = state[j]
            to_node = state[j + stride]
            if weight_matrix[from_node][to_node] == 0 and from_node != to_node:
                return False

    # If we encounter an accelerator for the first time, increment found_vec
    # Otherwise, do nothing
    found = {}
    found_vec = np.zeros(num_elements)
    for val in state:
        if val == -1:
            continue
        if val < num_elements:
            if val not in found.keys():
                found[val] = 1
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
    elif: row == len(state)
    for i in range(num_elements):
        if found_vec[i] == 0:
            return False

    return True


def iterative_backtrack(weight_matrix, state, best_result, best_result_weight, num_elements, depth, optimal):
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
        if is_valid(weight_matrix, state, num_elements, row + 1, depth):
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
            postprocess(result, num_elements, depth)
            weight = compute_tree_weight(weight_matrix, result, num_elements, depth, True)

            # Save this spanning tree if it is highest weight tree found so far
            if weight > best_result_weight:
                best_result_weight = weight
                best_result.clear()
                best_result.append(result)
            if not optimal:
                break

            pos = state_stack.pop()
            pos += 1
            state[len(state_stack)] = -1
            row -= 1


# This function takes a spanning tree encoded as state (result), which may have
# repeated GPUs representing NO-SENDs and converts it into a unique format.
# This has the effect of recognizing redundant sends, grouping them together,
# so that the Reduce call knows not to perform a CopyFromTo.
#
# Initial result: [3 0 0 4 1 2 5 6]
# Final result:   [3 3 0 4 1 2 5 6]
#
# Initial:
#         3
#     3     1
#   3   0   1   5
# 3 0 0 4 1 2 5 6    // GPU3 will make redundant send to GPU0
#
# Final:
#         3
#     3     1
#   3   0   1   5
# 3 3 0 4 1 2 5 6    // GPU3 knows not to make redundant send to itself
def postprocess(result, num_elements, depth):
    for level in range(depth-1, -1, -1):
        stride = 1 << level
        histogram_above = np.zeros(num_elements)
        for i in range(0, len(result), 2 * stride):
            val = result[i]
            histogram_above[val] += 1
        histogram = np.zeros(num_elements)
        for i in range(0, len(result), stride):
            val = result[i]
            histogram[val] += 1

        for i in range(len(result)-stride, stride-1, -2*stride):
            from_node = result[i]
            to_node = result[i-stride]
            if (histogram[from_node] > 1 or histogram_above[from_node] >= 1) and from_node != to_node:
                result[i] = to_node
                histogram[from_node] -= 1


def backtrack_generate_binary_tree(weight_matrix, num_elements, root, topo, scan):
    # Clear before starting
    topo.clear()
    scan.clear()

    # Compute depth
    # num_elemennts: depth
    # 5: 3 8
    # 6: 3 8
    # 7: 3 8
    # 8: 3 8
    # 9: 4 16
    depth = comupte_depth(num_elements)
    depth_leaves = 1 << depth

    # State vector
    # -1 means unplaced
    state = -np.ones(depth_leaves)
    result = -np.ones(depth_leaves)
    result_weight = None

    # Place root and try all combinations
    state[0] = root

    # Seek optimal solution until depth <= 3 i.e. 8 accelerators
    # For larger number of accelerators, settle for first tree found (non-optimal),
    # but this saves a lot of runtime since Backtrack is exponential time
    if depth <= 3:
        iterative_backtrack(weight_matrix, state, result, result_weight, num_elements, depth, True)
    else:
        iterative_backtrack(weight_matrix, state, result, result_weight, num_elements, depth, False)

    return form_topology(result, topo, scan, depth)


def compute_tree_from_root(weight_matrix, num_elements, root, alpha, topo, scan):
    num_partitions = 1

    # Initialize partition array to indicate which partition each element
    # belongs to, beginning with 0
    partitions = np.zeros(num_elements, dtype=int)

    # Initialize vector of pairs that tells the edges between what 2
    # clusters should be looked to build the tree from
    roots = []
    roots.append(root)

    # Temporary variables for rewinding
    partitions_temp
    num_partitions_temp
    roots_temp = []
    topo_temp = []
    scan_temp = []

    # Determine number of partition levels
    # If first partition, determine root of maximal spanning tree
    stop = False
    reset = True
    level = 0

    while not stop or reset:
        if reset:
            cluster_pairs = []
            partitions_temp = deepcopy(partitions)
            num_partitions_temp = num_partitions
            roots_temp = deepcopy(roots)
            topo_temp = deepcopy(topo)
            scan_temp = deepcopy(scan)

        # Run Kernighan-Lin to generate partition
        stop, num_partitions_temp = kernighan_lin(weight_matrix, num_partitions_temp, cluster_pairs)

        # Use partitions found and a given root to find best inter-cluster edge
        # for each pair of clusters, and returns them as roots of next cluster.
        # If reset is true, then rewind back to previous clustering
        reset = kl_generate_binary_tree(weight_matrix, cluster_pairs, roots_temp, topo_temp, scan_temp)

        if reset:
            level += 1
        if level > 10:
            break

    success = True
    if reset:
        print("No valid tree found from root {}, try backtracking".format(root))
        success = backtrack_generate_binary_tree(weight_matrix, num_elements, root, topo, scan)
    else:
        topo.clear()
        topo.append(topo_temp)
        scan.clear()
        scan.append(scan_temp)
        scan.append(len(topo))

    if success:
        update_weight(weight_matrix, topo, num_elements, alpha)
    else:
        print("No valid binary tree found from root {} using backtracking".format(root))


def compute_trees(weight_matrix, num_elements, alpha, log_tree = False):
    weight_matrix_cp = deepcopy(weight_matrix)

    topo = []
    scan = []
    topo_temp = []
    for i in range(num_elements):
        topo.append([])
        scan.append([])
        topo_temp.append([])
        topo[i].append(i)
        scan[i].append(i)
        compute_trees_from_root()

    # NOTE: must sum up adjacent weight matrix to show link usage before readjusting
    adj = np.zeros(weight_matrix.shape)
    for row in range(num_elements):
        for col in range(1, len(topo[0]), 2):
            from_node = min(topo[row][col], topo[row][col + 1])
            dest_node = max(topo[row][col], topo[row][col + 1])
            if from_node != dest_node:
                adj[from_node][dest_node] += 1
                adj[dest_node][from_node] += 1

    if log_tree:
        for i in range(num_elements):
            print("Tree {}:".format(i))
            depth = len(scan[i]) - 1
            for row in range(depth):
                start = scan[i][row]
                end = scan[i][row + 1]
                output = ""
                for j in range(start, end):
                    for k in range((2 << (depth - row - 2)) + 1):
                        output += " "
                    output += str(topo[j])
                print(output)

        print("W:")
        print(weight_matrix)
        print("Links:")
        print(adj)

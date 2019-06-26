import math 
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)

def sram_train(
        dimension_rows = 4,
        dimension_cols = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3,
        strides = 1, num_filters = 8,
        filter_base = 4000000, ifmap_base = 0,
        ifmap_gradient_base = 6000000,
        ofmap_gradient_base = 8000000,
        filter_gradient_base = 10000000
        ):

    # Dimensions of output feature map channel
    ofmap_h = (ifmap_h - filt_h + strides) / strides
    ofmap_w = (ifmap_w - filt_w + strides) / strides
    ofmap_channels = num_filters

    # Number of pixels in one convolution window
    input_gradient_px_per_conv_window = filt_h * filt_w
    filter_gradient_px_per_conv_window = ofmap_h * ofmap_w

    # Total number of filter weights across all channels
    filter_size = filt_h * filt_w * num_channels

    # Variables to calculate folds in runtime
    #   for input gradient
    num_h_fold_input_gradient = math.ceil(ifmap_h * ifmap_w / dimension_rows)
    num_v_fold_input_gradient = math.ceil(num_channels / dimension_cols)
    #   for filter gradient
    num_h_fold_filter_gradient = math.ceil(filter_size / dimension_rows)
    num_v_fold_filter_gradient = math.ceil(ofmap_channels / dimension_cols)

    cycles = 0

    print(' - compute input-gradient:')
    input_grad_read_cycles, input_grad_util = gen_input_gradient_read_trace(
            cycle = 0,
            dim_rows = dimension_rows,
            dim_cols = dimension_cols,
            num_v_fold = num_v_fold_input_gradient,
            num_h_fold = num_h_fold_input_gradient,
            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
            filt_h = filt_h, filt_w = filt_w,
            num_channels = num_channels, stride = strides,
            ofmap_h = int(ofmap_h), ofmap_w = int(ofmap_w), num_filters = num_filters,
            ofmap_gradient_base = ofmap_gradient_base, filter_base = filter_base
            )

    input_grad_write_cycles = gen_input_gradient_write_trace(
            cycle = cycles,
            dim_rows = dimension_rows,
            dim_cols = dimension_cols,
            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
            num_channels = num_channels,
            num_filters = num_filters,
            ifmap_gradient_base = ifmap_gradient_base,
            conv_window_size = input_gradient_px_per_conv_window
            )
    print(' -- input-gradient: read cycles: ', input_grad_read_cycles, ', write cycles: ', input_grad_write_cycles)

    print(' - compute filter-gradient:')
    filter_grad_read_cycles, filter_grad_util = gen_filter_gradient_read_trace(
            cycle = cycles,
            dim_rows = dimension_rows,
            dim_cols = dimension_cols,
            num_v_fold = int(num_v_fold_filter_gradient),
            num_h_fold = int(num_h_fold_filter_gradient),
            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
            filt_h = filt_h, filt_w = filt_w,
            num_channels = num_channels, stride = strides,
            ofmap_h = int(ofmap_h), ofmap_w = int(ofmap_w), num_filters = num_filters,
            ifmap_base = ifmap_base, ofmap_gradient_base = ofmap_gradient_base
            )

    filter_grad_write_cycles = gen_filter_gradient_write_trace(
            cycle = cycles,
            dim_rows = dimension_rows,
            dim_cols = dimension_cols,
            filt_h = filt_h, filt_w = filt_w,
            num_channels = num_channels,
            num_filters = num_filters,
            filter_gradient_base = filter_gradient_base,
            conv_window_size = filter_gradient_px_per_conv_window
            )
    print(' -- filter-gradient: read cycles: ', filter_grad_read_cycles, ', write cycles: ', filter_grad_write_cycles)

    # TODO: change it after adding deconvolution
    input_grad_cycles = max(input_grad_read_cycles, input_grad_write_cycles)
    filter_grad_cycles = max(filter_grad_read_cycles, filter_grad_write_cycles)
    cycles = input_grad_cycles + filter_grad_cycles
    print(' - input-gradient-cycles: ', input_grad_read_cycles, ', uitl: ', input_grad_util, '%')
    print(' - filter-gradent-cycles: ', filter_grad_cycles, ', uitl: ', filter_grad_util, '%')
    util = (input_grad_util * input_grad_read_cycles + filter_grad_util * filter_grad_read_cycles) / \
            (input_grad_read_cycles + filter_grad_read_cycles)

    return cycles, util

def gen_filter_gradient_read_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, stride = 1,
        ofmap_h = 5, ofmap_w = 5, num_filters = 8,
        ifmap_base = 0, ofmap_gradient_base = 8000000
        ):
    # Layer specific variables
    num_ofmap_channels = num_filters
    px_per_delta_weight = ofmap_h * ofmap_w # for one element of delta_filter
    ofmap_row_offset = ofmap_w * num_ofmap_channels
    ifmap_row_offset = ifmap_w * num_channels
    filter_size = filt_h * filt_w * num_channels

    # Tracking variables
    local_cycle = 0
    remaining_filters = num_filters
    ifmap_done        = False
    ofmap_done        = False
    row_base_addr     = []
    row_clk_offset    = []
    row_filter_idx    = []
    v_fold_row        = []
    col_base_addr     = []
    col_clk_offset    = []
    v_fold_col        = []
    h_fold_col        = []
    lane_done         = []
    v_fold_barrier    = []

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0

    # This initialization assumes num_rows << filter_size
    # This assignment logic needs to modified if that is not the case
    for r in range(dim_rows):
        # Calculate base_addr of ifmap for a particular filter element
        base_row_col_id = math.floor(r / num_channels)
        base_ch_id  = r % num_channels
        base_row_id = math.floor(base_row_col_id / filt_w) * stride
        base_col_id = base_row_col_id % filt_w * stride
        base_addr = base_row_id * ifmap_row_offset + base_col_id * num_channels + base_ch_id

        if r < filter_size:
            clk_offset = r * -1     # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf    # In case filter_size < dim_rows

        row_base_addr.append(base_addr)
        row_clk_offset.append(clk_offset)
        row_filter_idx.append(r)
        v_fold_row.append(0)
        v_fold_barrier.append(False)

    for c in range(dim_cols):
        base_addr = c

        if c < remaining_filters:
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)
        v_fold_col.append(0)
        h_fold_col.append(0)

    # Progress bar
    total = filter_size * num_v_fold
    pbar = tqdm(total = total)

    # Start computation
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the ofmap traces have been generated
    while ifmap_done == False or ofmap_done == False:
        rows_used = 0
        cols_used = 0

        # Address generation for ifmap
        for r in range(dim_rows):

            if row_clk_offset[r] >= 0:

                linear_idx = row_clk_offset[r]

                addr_row_offset = math.floor(linear_idx / ofmap_w) * ifmap_w * num_channels
                addr_col_offset = linear_idx % ofmap_w * num_channels
                ifmap_addr = row_base_addr[r] + addr_row_offset + addr_col_offset
                rows_used += 1

            row_clk_offset[r] += 1

            if row_clk_offset[r] > 0 and row_clk_offset[r] % px_per_delta_weight == 0: # Completed MAC for one delta weight
                row_filter_idx[r] += dim_rows
                filter_idx = row_filter_idx[r]

                # Update progress bar
                pbar.update(1)

                if filter_idx < filter_size:
                    row_clk_offset[r] = 0

                    base_row_col_id = math.floor(filter_idx / num_channels)
                    base_ch_id  = filter_idx % num_channels
                    base_row_id = math.floor(base_row_col_id / filt_w) * stride
                    base_col_id = base_row_col_id % filt_w * stride
                    base_addr = base_row_id * ifmap_row_offset + base_col_id * num_channels + base_ch_id
                    row_base_addr[r]  = base_addr

                else:
                    # Reinitialize for next group of column filters

                    v_fold_row[r] += 1

                    if v_fold_row[r] < num_v_fold:
                        row_filter_idx[r] = r

                        base_row_col_id = math.floor(r / num_channels)
                        base_ch_id  = r % num_channels
                        base_row_id = math.floor(base_row_col_id / filt_w) * stride
                        base_col_id = base_row_col_id % filt_w * stride
                        base_addr = base_row_id * ifmap_row_offset + base_col_id * num_channels + base_ch_id
                        row_base_addr[r]  = base_addr

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        if r != 0 and (v_fold_row[r] > v_fold_row[r-1] or v_fold_barrier[r-1] == True):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        for r in range(dim_rows):
            if v_fold_barrier[r] and v_fold_row[r] == v_fold_row[r-1] and v_fold_barrier[r-1] == False:
                v_fold_barrier[r] = False
                row_clk_offset[r] = row_clk_offset[r-1] - 1
                break

        # Check if all input traces are done
        ifmap_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0:
                ifmap_done = False
                break

        # Generate address for ofmap gradients
        for c in range(dim_cols):
            if col_clk_offset[c] >= 0:

                linear_idx = col_clk_offset[c]

                base_row_offset = math.floor(linear_idx / ofmap_w) * ofmap_w * num_ofmap_channels
                base_col_offset = linear_idx % ofmap_w * num_ofmap_channels
                ofmap_addr = col_base_addr[c] + base_row_offset + base_col_offset + ofmap_gradient_base

                cols_used += 1

            col_clk_offset[c] += 1

            if col_clk_offset[c] > 0 and col_clk_offset[c] % px_per_delta_weight == 0:

                # Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Check if all the input traces are generated for the given v fold
                if h_fold_col[c] < num_h_fold:
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    filt_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if v_fold_col[c] < num_v_fold and filt_id < num_filters:
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        col_base_addr[c] = filt_id

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all ofmap gradient traces are generated
        ofmap_done = True
        for c in range(dim_cols):
            if lane_done[c] == False:
                ofmap_done = False
                break

        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util

        # Cycle update
        local_cycle += 1

    pbar.close()

    #calculated_macs = util * dim_rows * dim_cols
    #macs = (ofmap_h * ofmap_w) * filter_size * num_filters
    #accuracy = calculated_macs / macs * 100
    #print('Filter-Gradient: total number of computes: ', calculated_macs, ' (should be: ', macs, ', accuracy: ', accuracy,'%)')

    util_percentile = (util / local_cycle) * 100

    return (local_cycle + cycle), util_percentile
# gen_filter_gradient_read_trace() end

def gen_filter_gradient_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        filt_h = 3, filt_w = 3,
        num_channels = 3,
        num_filters = 4,
        filter_gradient_base = 10000000,
        conv_window_size = 25
        ):

    # Layer specific variables
    filter_size = filt_h * filt_w * num_channels

    # Tracking variables
    id_row = []             # List of filter map ID for each row
    id_col = []             # List of filter ID for each row
    base_addr_col = []      # Starting address of each output filter
    remaining_px  = filter_size
    remaining_filters = num_filters
    active_row = min(dim_rows, filter_size)
    active_col = min(dim_cols, num_filters)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug

    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c * filter_size
        base_addr_col.append(base_col)

    # This is the cycle when all the filter weights in the first col become available
    local_cycle = conv_window_size + active_row - 1 # FIXME

    while remaining_px > 0 or remaining_filters > 0:

        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            for c in range(active_col):
                addr = filter_gradient_base + base_addr_col[c] + local_px * num_channels

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_filters -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle = local_cycle + active_row
            local_cycle -= (active_row + active_col - 1)
            sticky_flag = True

            # There are more Filters to go
            if remaining_filters > 0:
                remaining_px = filter_size
                last_active_col = active_col
                active_col = min(remaining_filters, dim_cols)

                # Re-assign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col * filter_size

                active_row = min(dim_rows, remaining_px)
                # Re-assign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += conv_window_size + active_row # FIXME
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle

            else:   # Restore the local cycle to return to them main function
                local_cycle = last_fold_cycle

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(conv_window_size, active_row)

    return local_cycle + cycle
# gen_filter_gradient_write_trace() end

def gen_input_gradient_read_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, stride = 1,
        ofmap_h = 5, ofmap_w = 5, num_filters = 8,
        ofmap_gradient_base = 8000000, filter_base = 4000000
        ):

    # Layer specific variables
    padded_ofmap_h = (filt_h - 1) + stride * (ofmap_h - 1) + 1
    padded_ofmap_w = (filt_w - 1) + stride * (ofmap_w - 1) + 1
    num_ofmap_channels = num_filters
    px_per_delta_ifmap = filt_h * filt_w * num_filters # for one element of delta_filter
    ofmap_row_offset = ofmap_w * num_ofmap_channels
    filter_row_offset = filt_w * num_channels
    ifmap_size_per_channel = ifmap_h * ifmap_w
    filter_size_per_channel = filt_h * filt_w
    filter_size = filter_size_per_channel * num_channels

    # inner padding and outer padding boundary
    inner_boundary_row_min = filt_h - 1
    inner_boundary_row_max = filt_h - 1 + stride * (ofmap_h - 1) + 1
    inner_boundary_col_min = filt_w - 1
    inner_boundary_col_max = filt_w - 1 + stride * (ofmap_w - 1) + 1

    # Tracking variables
    local_cycle = 0
    remaining_channels  = num_channels
    ofmap_gradient_done = False
    filter_done         = False
    row_base_addr       = []
    row_clk_offset      = []
    row_ifmap_idx       = []
    v_fold_row          = []
    col_base_addr       = []
    col_clk_offset      = []
    v_fold_col          = []
    h_fold_col          = []
    lane_done           = []
    v_fold_barrier      = []
    row_base_padded_row_id = []
    row_base_padded_col_id = []

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0

    # This initialization assumes num_rows << filter_size
    # This assignment logic needs to modified if that is not the case
    for r in range(dim_rows):
        # Calculate base_addr of ifmap for a particular filter element
        base_padded_row_id = math.floor(r / ifmap_w) * stride
        base_padded_col_id = r % ifmap_w * stride
        #base_addr = base_row_id * ifmap_row_offset + base_col_id * num_channels + base_ch_id

        if r < ifmap_size_per_channel:
            clk_offset = r * -1     # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf    # In case filter_size < dim_rows

        #row_base_addr.append(base_addr)
        row_base_padded_row_id.append(base_padded_row_id)
        row_base_padded_col_id.append(base_padded_col_id)
        row_clk_offset.append(clk_offset)
        row_ifmap_idx.append(r)
        v_fold_row.append(0)
        v_fold_barrier.append(False)

    for c in range(dim_cols):
        base_addr = c #filter_size - num_channels + c

        if c < remaining_channels:
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)
        v_fold_col.append(0)
        h_fold_col.append(0)

    # Progress bar
    total = ifmap_size_per_channel * num_v_fold
    pbar = tqdm(total = total)

    # Start computation
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the ofmap traces have been generated
    while ofmap_gradient_done == False or filter_done == False:
        rows_used = 0
        cols_used = 0

        # Address generation for ifmap
        for r in range(dim_rows):

            if row_clk_offset[r] >= 0:

                linear_idx = row_clk_offset[r] % (filt_h * filt_w)
                ofmap_ch_id = math.floor(row_clk_offset[r] / (filt_h * filt_w))

                padded_row_id = row_base_padded_row_id[r] + math.floor(linear_idx / filt_w)
                padded_col_id = row_base_padded_col_id[r] + linear_idx % filt_w
                actual_row_id = -1
                actual_col_id = -1
                if (padded_row_id >= inner_boundary_row_min and \
                        padded_row_id <= inner_boundary_row_max and \
                        padded_col_id >= inner_boundary_col_min and \
                        padded_col_id <= inner_boundary_col_max and \
                        (padded_row_id - (filt_h - 1)) % stride == 0 and \
                        (padded_col_id - (filt_w - 1)) % stride == 0):
                    actual_row_id = math.floor((padded_row_id - (filt_h - 1)) / stride)
                    actual_col_id = math.floor((padded_col_id - (filt_w - 1)) / stride)

                    addr_row_offset = actual_row_id * ofmap_w * num_ofmap_channels
                    addr_col_offset = actual_col_id * num_ofmap_channels
                    ofmap_gradient_addr = addr_row_offset + addr_col_offset + ofmap_ch_id

                else:
                    '''
                    padded zero
                    '''

                rows_used += 1

            row_clk_offset[r] += 1

            if row_clk_offset[r] > 0 and row_clk_offset[r] % px_per_delta_ifmap == 0: # Completed MAC for one delta weight
                row_ifmap_idx[r] += dim_rows
                ifmap_idx = row_ifmap_idx[r]

                # Update progress bar
                pbar.update(1)

                if ifmap_idx < ifmap_size_per_channel:
                    row_clk_offset[r] = 0

                    base_padded_row_id = math.floor(ifmap_idx / ifmap_w) * stride
                    base_padded_col_id = ifmap_idx % ifmap_w * stride
                    row_base_padded_row_id[r] = base_padded_row_id
                    row_base_padded_col_id[r] = base_padded_col_id

                else:
                    # Reinitialize for next group of column filters

                    v_fold_row[r] += 1

                    if v_fold_row[r] < num_v_fold:
                        row_ifmap_idx[r] = r

                        base_padded_row_id = math.floor(r / ifmap_w) * stride
                        base_padded_col_id = r % ifmap_w * stride
                        row_base_padded_row_id[r] = base_padded_row_id
                        row_base_padded_col_id[r] = base_padded_col_id

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        if r != 0 and (v_fold_row[r] > v_fold_row[r-1] or v_fold_barrier[r-1] == True):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        for r in range(dim_rows):
            if v_fold_barrier[r] and v_fold_row[r] == v_fold_row[r-1] and v_fold_barrier[r-1] == False:
                v_fold_barrier[r] = False
                row_clk_offset[r] = row_clk_offset[r-1] - 1
                break

        # Check if all input traces are done
        ofmap_gradient_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0:
                ofmap_gradient_done = False
                break

        # Generate address for ofmap gradients
        for c in range(dim_cols):
            if col_clk_offset[c] >= 0:

                linear_idx = col_clk_offset[c]

                filter_idx = linear_idx / filter_size_per_channel
                row_col_idx = linear_idx % filter_size_per_channel

                row_id = filt_h - math.floor(row_col_idx / filt_w)
                col_id = filt_w - row_col_idx % filt_w
                base_row_offset = row_id * filt_w * num_ofmap_channels
                base_col_offset = col_id * num_ofmap_channels
                base_filter_offset = filter_idx * filter_size

                ofmap_addr = col_base_addr[c] + base_filter_offset + base_row_offset + base_col_offset

                cols_used += 1

            col_clk_offset[c] += 1

            if col_clk_offset[c] > 0 and col_clk_offset[c] % px_per_delta_ifmap == 0:

                # Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Check if all the input traces are generated for the given v fold
                if h_fold_col[c] < num_h_fold:
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    channel_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if v_fold_col[c] < num_v_fold and channel_id < num_channels:
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        col_base_addr[c] = channel_id

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all ofmap gradient traces are generated
        filter_done = True
        for c in range(dim_cols):
            if lane_done[c] == False:
                filter_done = False
                break

        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util

        # Cycle update
        local_cycle += 1

    pbar.close()

    #calculated_macs = util * dim_rows * dim_cols
    #macs = px_per_delta_ifmap * ifmap_size_per_channel * num_channels
    #accuracy = calculated_macs / macs * 100
    #print('Inpute-Gradient: total number of macs: ', calculated_macs, ' (should be: ', macs, ', accuracy: ', accuracy,'%)')

    util_percentile = (util / local_cycle) * 100

    return (local_cycle + cycle), util_percentile
# gen_input_gradient_read_trace() end

def gen_input_gradient_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        ifmap_h = 7, ifmap_w = 7,
        num_channels = 3,
        num_filters = 4,
        ifmap_gradient_base = 6000000,
        conv_window_size = 9
        ):

    # Layer specific variables
    ifmap_size_per_channel = ifmap_h * ifmap_w

    # Tracking variables
    id_row = []             # List of IFMAP ID for each row
    id_col = []             # List of channel ID for each row
    base_addr_col = []      # Starting address of each input channel
    remaining_px  = ifmap_size_per_channel
    remaining_channels = num_channels
    active_row = min(dim_rows, ifmap_size_per_channel)
    active_col = min(dim_cols, num_channels)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug

    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c
        base_addr_col.append(base_col)

    # This is the cycle when all the filter weights in the first col become available
    local_cycle = conv_window_size + active_row - 1 # FIXME

    while remaining_px > 0 or remaining_channels > 0:

        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            for c in range(active_col):
                addr = ifmap_gradient_base + base_addr_col[c] + local_px * num_channels

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_channels -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle = local_cycle + active_row
            local_cycle -= (active_row + active_col - 1)
            sticky_flag = True

            # There are more Filters to go
            if remaining_channels > 0:
                remaining_px = ifmap_size_per_channel
                last_active_col = active_col
                active_col = min(remaining_channels, dim_cols)

                # Re-assign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col

                active_row = min(dim_rows, remaining_px)
                # Re-assign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += conv_window_size + active_row # FIXME
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle

            else:   # Restore the local cycle to return to them main function
                local_cycle = last_fold_cycle

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(conv_window_size, active_row)

    return local_cycle + cycle
# gen_input_gradient_write_trace()

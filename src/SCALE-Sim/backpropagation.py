import math
import dram_trace as dram
import sram_train_os as sram

def backprop(
        array_h = 4,
        array_w = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3,
        strides = 1, num_filt = 8,

        data_flow = 'os',

        word_size_bytes = 1,
        filter_sram_size = 64, ifmap_sram_size = 64, ofmap_sram_size = 64,

        ifmap_base = 0, filter_base = 2000000, ofmap_base = 4000000,
        ifmap_gradient_base = 6000000,
        ofmap_gradient_base = 8000000,
        filter_gradient_base = 10000000,

        sram_read_trace_file = 'sram_read.csv',
        sram_ifmap_gradient_write_trace_file = 'sram_ifmap_gradient_write.csv',
        sram_filter_gradient_write_trace_file = 'sram_filter_gradient_write.csv',

        dram_ifmap_trace_file = 'dram_ifmap_read.csv',
        dram_filter_trace_file = 'dram_filter_read.csv',
        dram_ofmap_gradient_trace_file = 'dram_ofmap_gradient_read.csv',
        dram_filter_gradient_trace_file = 'dram_filter_gradient_write.csv',
        dram_ifmap_gradient_trace_file = 'dram_ifmap_gradient_write.csv'
        ):

    sram_cycles = 0
    util        = 0

    assert data_flow == 'os'

    sram_cycles, util = \
            sram.sram_train(
                    dimension_rows = array_h,
                    dimension_cols = array_w,
                    ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                    filt_h = filt_h, filt_w = filt_w,
                    num_channels = num_channels,
                    strides = strides, num_filters = num_filt,
                    filter_base = filter_base, ifmap_base = ifmap_base,
                    ifmap_gradient_base = ifmap_gradient_base,
                    ofmap_gradient_base = ofmap_gradient_base,
                    filter_gradient_base = filter_gradient_base,
                    sram_read_trace_file = sram_read_trace_file,
                    sram_ifmap_gradient_write_trace_file = sram_ifmap_gradient_write_trace_file,
                    sram_filter_gradient_write_trace_file = sram_filter_gradient_write_trace_file
                    )

    # IFMAP dram read traces
    dram.dram_trace_read_v2(
            sram_sz = ifmap_sram_size,
            word_sz_bytes = word_size_bytes,
            min_addr = ifmap_base, max_addr = filter_base,
            sram_trace_file = sram_read_trace_file,
            dram_trace_file = dram_ifmap_trace_file
            )

    # Filter dram read traces
    # NOTE: use ifmap sram when backpropagate
    dram.dram_trace_read_v2(
            sram_sz = ifmap_sram_size,
            word_sz_bytes = word_size_bytes,
            min_addr = filter_base, max_addr = ofmap_base,
            sram_trace_file = sram_read_trace_file,
            dram_trace_file = dram_filter_trace_file
            )

    # OFMAP gradient dram read traces
    # NOTE: use filter sram when backpropagate
    dram.dram_trace_read_v2(
            sram_sz = filter_sram_size,
            word_sz_bytes = word_size_bytes,
            min_addr = ofmap_gradient_base, max_addr = ifmap_gradient_base,
            sram_trace_file = sram_read_trace_file,
            dram_trace_file = dram_ofmap_gradient_trace_file
            )

    # IFMAP gradient dram write traces
    # NOTE: use ofmap sram for gradient write
    dram.dram_trace_write(
            ofmap_sram_size = ofmap_sram_size,
            data_width_bytes = word_size_bytes,
            sram_write_trace_file = sram_ifmap_gradient_write_trace_file,
            dram_write_trace_file = dram_ifmap_gradient_trace_file
            )

    # Filter gradient dram write traces
    # NOTE: use ofmap sram for gradient write
    dram.dram_trace_write(
            ofmap_sram_size = ofmap_sram_size,
            data_width_bytes = word_size_bytes,
            sram_write_trace_file = sram_filter_gradient_write_trace_file,
            dram_write_trace_file = dram_filter_gradient_trace_file
            )

    print("Average utilization : \t" + str(util) + " %")
    print("Cycles for compute  : \t" + str(sram_cycles) + " cycles")

    bw_numbers, detailed_log = gen_bw_numbers(
            dram_ifmap_trace_file,
            dram_filter_trace_file,
            dram_ofmap_gradient_trace_file,
            dram_ifmap_gradient_trace_file,
            dram_filter_gradient_trace_file,
            sram_read_trace_file,
            sram_ifmap_gradient_write_trace_file,
            sram_filter_gradient_write_trace_file
            )

    return bw_numbers, detailed_log, sram_cycles, util

def gen_bw_numbers( dram_ifmap_trace_file,
                    dram_filter_trace_file,
                    dram_ofmap_gradient_trace_file,
                    dram_ifmap_gradient_trace_file,
                    dram_filter_gradient_trace_file,
                    sram_read_trace_file,
                    sram_ifmap_gradient_write_trace_file,
                    sram_filter_gradient_write_trace_file):
    min_clk = 100000
    max_clk = -1
    detailed_log = ''

    minclk, maxclk, start_clk, stop_clk, num_dram_ifmap_bytes = gen_dram_bw(dram_ifmap_trace_file)
    if minclk < min_clk:
        min_clk = minclk
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_dram_ifmap_bytes) + ',\t'

    minclk, maxclk, start_clk, stop_clk, num_dram_filter_bytes = gen_dram_bw(dram_filter_trace_file)
    if minclk < min_clk:
        min_clk = minclk
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_dram_filter_bytes) + ',\t'

    minclk, maxclk, start_clk, stop_clk, num_dram_ofmap_gradient_bytes = gen_dram_bw(dram_ofmap_gradient_trace_file)
    if minclk < min_clk:
        min_clk = minclk
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_dram_ofmap_gradient_bytes) + ',\t'

    minclk, maxclk, start_clk, stop_clk, num_dram_ifmap_gradient_bytes = gen_dram_bw(dram_ifmap_gradient_trace_file)
    if minclk < min_clk:
        min_clk = minclk
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_dram_ifmap_gradient_bytes) + ',\t'

    minclk, maxclk, start_clk, stop_clk, num_dram_filter_gradient_bytes = gen_dram_bw(dram_filter_gradient_trace_file)
    if minclk < min_clk:
        min_clk = minclk
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_dram_filter_gradient_bytes) + ',\t'

    maxclk, start_clk, stop_clk, num_sram_read_bytes = gen_sram_bw(sram_read_trace_file)
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_sram_read_bytes) + ',\t'

    maxclk, start_clk, stop_clk, num_sram_ifmap_gradient_write_bytes = gen_sram_bw(sram_ifmap_gradient_write_trace_file)
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_sram_ifmap_gradient_write_bytes) + ',\t'

    maxclk, start_clk, stop_clk, num_sram_filter_gradient_write_bytes = gen_sram_bw(sram_filter_gradient_write_trace_file)
    if maxclk > max_clk:
        max_clk = maxclk
    detailed_log += str(start_clk) + '\t' + str(stop_clk) + ',\t' + str(num_sram_filter_gradient_write_bytes) + ',\t'

    delta_clk = max_clk - min_clk

    dram_ifmap_bw           = num_dram_ifmap_bytes / delta_clk
    dram_filter_bw          = num_dram_filter_bytes / delta_clk
    dram_ofmap_gradient_bw  = num_dram_ofmap_gradient_bytes / delta_clk
    dram_ifmap_gradient_bw  = num_dram_ifmap_gradient_bytes / delta_clk
    dram_filter_gradient_bw = num_dram_filter_gradient_bytes / delta_clk
    sram_read_bw            = num_sram_read_bytes / delta_clk
    sram_ifmap_gradient_bw  = num_sram_ifmap_gradient_write_bytes / delta_clk
    sram_filter_gradient_bw = num_sram_filter_gradient_write_bytes / delta_clk

    unit = ' Bytes/cycle'
    print('DRAM IFMAP Read BW            : ' + str(dram_ifmap_bw) + unit)
    print('DRAM Filter Read BW           : ' + str(dram_filter_bw) + unit)
    print('DRAM OFMAP Gradient Read BW   : ' + str(dram_ofmap_gradient_bw) + unit)
    print('DRAM IFMAP Gradient Write BW  : ' + str(dram_ifmap_gradient_bw) + unit)
    print('DRAM Filter Gradient Write BW : ' + str(dram_filter_gradient_bw) + unit)
    print('SRAM Read BW                  : ' + str(sram_read_bw) + unit)
    print('SRAM IFMAP Gradient Write BW  : ' + str(sram_ifmap_gradient_bw) + unit)
    print('SRAM Filter Gradient Write BW : ' + str(sram_filter_gradient_bw) + unit)

    log = str(dram_ifmap_bw) + ',\t' + str(dram_filter_bw) + ',\t' + str(dram_ofmap_gradient_bw) + ',\t' + str(dram_ifmap_gradient_bw) + ',\t' + str(dram_filter_gradient_bw) + ',\t' + str(sram_read_bw) + ',\t' + str(sram_ifmap_gradient_bw) + ',\t' + str(sram_filter_gradient_bw) + ','

    return log, detailed_log


def gen_max_bw_numbers( dram_ifmap_trace_file,
                    dram_filter_trace_file,
                    dram_ofmap_gradient_trace_file,
                    dram_ifmap_gradient_trace_file,
                    dram_filter_gradient_trace_file,
                    sram_read_trace_file,
                    sram_ifmap_gradient_write_trace_file,
                    sram_filter_gradient_write_trace_file):

    max_dram_ifmap_bw = gen_max_dram_bw(dram_ifmap_trace_file)
    max_dram_filter_bw = gen_max_dram_bw(dram_filter_trace_file)
    max_dram_ofmap_gradient_bw = gen_max_dram_bw(dram_ofmap_gradient_trace_file)
    max_dram_ifmap_gradient_bw = gen_max_dram_bw(dram_ifmap_gradient_trace_file)
    max_dram_filter_gradient_bw = gen_max_dram_bw(dram_filter_gradient_trace_file)

    max_sram_read_bw = gen_max_sram_bw(sram_read_trace_file)
    max_sram_ifmap_gradient_bw = gen_max_sram_bw(sram_ifmap_gradient_write_trace_file)
    max_sram_filter_gradient_bw = gen_max_sram_bw(sram_filter_gradient_write_trace_file)

    log = str(max_dram_ifmap_bw) + ',\t' + str(max_dram_filter_bw) + ',\t' + str(max_dram_ofmap_gradient_bw) + ',\t'
    log += str(max_dram_ifmap_gradient_bw) + ',\t' + str(max_dram_filter_gradient_bw) + ',\t'
    log += str(max_sram_read_bw) + ',\t' + str(max_sram_ifmap_gradient_bw) + ',\t' + str(max_sram_filter_gradient_bw) + ','

    return log


def gen_dram_bw(dram_trace_file):
    min_clk = 100000
    max_clk = -1

    num_dram_activation_bytes = 0
    tracefile = open(dram_trace_file, 'r')
    start_clk = 0
    first = True

    for row in tracefile:
        num_dram_activation_bytes += len(row.split(',')) - 2

        elems = row.strip().split(',')
        clk = float(elems[0])

        if first:
            first = False
            start_clk = clk

        if clk < min_clk:
            min_clk = clk

    stop_clk = clk
    if clk > max_clk:
        max_clk = clk

    tracefile.close()

    return min_clk, max_clk, start_clk, stop_clk, num_dram_activation_bytes


def gen_sram_bw(sram_trace_file):
    #min_clk = 100000
    max_clk = -1

    num_sram_bytes = 0
    tracefile = open(sram_trace_file, 'r')
    first = True

    for row in tracefile:
        elems = row.strip().split(',')
        clk = float(elems[0])

        if first:
            first = False
            start_clk = clk

        for i in range(1, len(elems)):
            if elems[i] != ' ':
                num_sram_bytes += 1

    stop_clk = clk
    if clk > max_clk:
        max_clk = clk

    tracefile.close()

    return max_clk, start_clk, stop_clk, num_sram_bytes


def gen_max_dram_bw(dram_trace_file):
    max_dram_bw = 0
    num_bytes = 0
    max_dram_clk = ''
    tracefile = open(dram_trace_file, 'r')

    for row in tracefile:
        clk = row.split(',')[0]
        num_bytes = len(row.split(',')) - 2

        if max_dram_bw < num_bytes:
            max_dram_bw = num_bytes
            max_dram_clk = clk

    tracefile.close()

    return max_dram_bw


def gen_max_sram_bw(sram_trace_file):
    max_sram_bw = 0
    num_bytes = 0
    tracefile = open(sram_trace_file, 'r')

    for row in tracefile:
        num_bytes = len(row.split(',')) - 2

        if max_sram_bw < num_bytes:
            max_sram_bw = num_bytes

    tracefile.close()

    return max_sram_bw

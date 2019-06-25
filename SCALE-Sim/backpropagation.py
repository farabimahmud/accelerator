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

        filt_base = 1000000, ifmap_base = 0, ofmap_base = 2000000,
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
                    filt_base = filt_base, ifmap_base = ifmap_base,
                    ofmap_base = ofmap_base,
                    )

    print("Average utilization : \t" + str(util) + " %")
    print("Cycles for compute  : \t" + str(sram_cycles) + " cycles")

    return sram_cycles, util

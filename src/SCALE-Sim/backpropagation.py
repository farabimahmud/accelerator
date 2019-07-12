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

        filter_base = 4000000, ifmap_base = 0,
        ifmap_gradient_base = 6000000,
        ofmap_gradient_base = 8000000,
        filter_gradient_base = 10000000
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
                    filter_gradient_base = filter_gradient_base
                    )

    print("Average utilization : \t" + str(util) + " %")
    print("Cycles for compute  : \t" + str(sram_cycles) + " cycles")

    return sram_cycles, util

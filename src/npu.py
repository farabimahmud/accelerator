import math

import trace_gen_wrapper as tg
import backpropagation as bp

class NPU:
    def __init__(self, args):
        self.args = args

    def aggregate(self, model_size, group_size):
        # assume sram bw is same as matrix multiplication
        # only first pe row/col is used
        pe_list_size = max(self.args.pe_array_height, self.args.pe_array_width)

        cycles = math.ceil(model_size / pe_list_size) * group_size

        return cycles

    def inference(self, model, training=False):
        avg_bw_file = open(model.name + '_avg_bw.csv', 'w')
        max_bw_file = open(model.name + '_max_bw.csv', 'w')
        cycle_file  = open(model.name + '_cyclces.csv', 'w')
        detail_file = open(model.name + '_detail.csv', 'w')

        avg_bw_file.write('IFMAP SRAM Size,\tFilter SRAM Size,\tOFMAP SRAM Size,\tConv Layer Num,\tDRAM IFMAP Read BW,\tDRAM Filter Read BW,\tDRAM OFMAP Write BW,\tSRAM Read BW,\tSRAM OFMAP WRITE BW,\n')
        max_bw_file.write('IFMAP SRAM Size,\tFilter SRAM Size,\tOFMAP SRAM Size,\tConv Layer Num,\tMax DRAM IFMAP Read BW,\tMax DRAM Filter Read BW,\tMax DRAM OFMAP Write BW,\tMax SRAM Read BW,\tMax SRAM OFMAP Write BW,\n')
        cycle_file.write('Layer,\tCycles,\t% Utilization,\n')
        detail_log = 'Layer,' + \
                     '\tDRAM_IFMAP_start,\tDRAM_IFMAP_stop,\tDRAM_IFMAP_bytes,' + \
                     '\tDRAM_Filter_start,\tDRAM_Filter_stop,\tDRAM_Filter_bytes,' + \
                     '\tDRAM_OFMAP_start,\tDRAM_OFMAP_stop,\tDRAM_OFMAP_bytes,' + \
                     '\tSRAM_read_start,\tSRAM_read_stop,\tSRAM_read_bytes,' + \
                     '\tSRAM_write_start,\tSRAM_write_stop,\tSRAM_write_bytes,\n'
        detail_file.write(detail_log)

        bw_head_str = str(self.args.ifmap_sram_size) + ',\t' + str(self.args.filter_sram_size) + ',\t' + str(self.args.ofmap_sram_size) + ',\t'

        print("\nFeed-Forward ...")

        if training:
            word_size_bytes = 4
        else:
            word_size_bytes = 1

        first = True
        total_cycles = 0
        total_util   = 0

        for l in range(model.num_layers):
            name = model.layers[l]['name']
            print('\nCommencing run for ' + name)

            ifmap_h = model.layers[l]['ifmap_h']
            ifmap_w = model.layers[l]['ifmap_w']

            filt_h = model.layers[l]['filter_h']
            filt_w = model.layers[l]['filter_w']

            num_channels = model.layers[l]['num_channels']
            num_filters = model.layers[l]['num_filters']

            stride = model.layers[l]['stride']

            bw_str, detailed_str, util, cycles = tg.gen_all_traces(
                    array_h = self.args.pe_array_height,
                    array_w = self.args.pe_array_width,
                    ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                    filt_h = filt_h, filt_w = filt_w,
                    num_channels = num_channels,
                    num_filt = num_filters,
                    strides = stride,
                    data_flow = self.args.data_flow,
                    word_size_bytes = word_size_bytes,
                    ifmap_sram_size = self.args.ifmap_sram_size,
                    filter_sram_size = self.args.filter_sram_size,
                    ofmap_sram_size = self.args.ofmap_sram_size,
                    ifmap_base = self.args.ifmap_offset,
                    filt_base = self.args.filter_offset,
                    ofmap_base = self.args.ofmap_offset,
                    sram_read_trace_file = model.name + '_' + name + '_sram_read.csv',
                    sram_write_trace_file = model.name + '_' + name + '_sram_write.csv',
                    dram_filter_trace_file = model.name + '_' + name + '_dram_filter_read.csv',
                    dram_ifmap_trace_file = model.name + '_' + name + '_dram_ifmap_read.csv',
                    dram_ofmap_trace_file = model.name + '_' + name + '_dram_ofmap_write.csv'
                    )

            total_cycles += int(cycles)
            total_util += util * int(cycles)

            avg_bw_log = bw_head_str + name + ',\t' + bw_str + '\n'
            avg_bw_file.write(avg_bw_log)

            detailed_log = name + ',\t' + detailed_str + '\n'
            detail_file.write(detailed_log)

            max_bw_log = bw_head_str + name + ',\t'
            max_bw_log += tg.gen_max_bw_numbers(
                    sram_read_trace_file = model.name + '_' + name + '_sram_read.csv',
                    sram_write_trace_file = model.name + '_' + name + '_sram_write.csv',
                    dram_filter_trace_file = model.name + '_' + name + '_dram_filter_read.csv',
                    dram_ifmap_trace_file = model.name + '_' + name + '_dram_ifmap_read.csv',
                    dram_ofmap_trace_file = model.name + '_' + name + '_dram_ofmap_write.csv'
                    )
            max_bw_file.write(max_bw_log + '\n')

            cycle_file.write(name + ',\t' + cycles + ',\t' + str(util) + ',\n')

        total_util /= total_cycles
        cycle_file.write('Total,\t' + str(total_cycles) + ',\t' + str(total_util))

        avg_bw_file.close()
        max_bw_file.close()
        cycle_file.close()
        detail_file.close()

        return total_cycles
    # inference() end

    def backprop(self, model):
        print('\nBackward Propagation ...')

        total_cycles = 0

        for l in reversed(range(model.num_layers)):
            name = model.layers[l]['name']
            print("")
            print("Commencing back-propagation run for " + name)

            ifmap_h = model.layers[l]['ifmap_h']
            ifmap_w = model.layers[l]['ifmap_w']

            filt_h = model.layers[l]['filter_h']
            filt_w = model.layers[l]['filter_w']

            num_channels = model.layers[l]['num_channels']
            num_filters = model.layers[l]['num_filters']

            stride = model.layers[l]['stride']

            cycles, util = bp.backprop(
                    array_h = self.args.pe_array_height,
                    array_w = self.args.pe_array_width,
                    ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                    filt_h = filt_h, filt_w = filt_w,
                    num_channels = num_channels,
                    strides = stride,
                    num_filt = num_filters,
                    word_size_bytes = 4,
                    filter_sram_size = self.args.filter_sram_size,
                    ifmap_sram_size = self.args.ifmap_sram_size,
                    ofmap_sram_size = self.args.ofmap_sram_size,
                    filter_base = self.args.filter_offset,
                    ifmap_base = self.args.ifmap_offset,
                    ifmap_gradient_base = self.args.ifmap_grad_offset,
                    ofmap_gradient_base = self.args.ofmap_grad_offset,
                    filter_gradient_base = self.args.filter_grad_offset
                    )

            total_cycles += cycles

        return total_cycles
    # end of backprop()

    def train(self, model):
        print('Start a training epoch ...')

        total_cycles = 0

        inference_cycles = self.inference(model, training=True)
        backprop_cycles = self.backprop(model)

        total_cycles = inference_cycles + backprop_cycles

        return total_cycles
    # end of train()


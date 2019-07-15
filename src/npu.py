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
        print("\nFeed-Forward ...")

        if training:
            word_size_bytes = 4
        else:
            word_size_bytes = 1

        first = True
        total_cycles = 0

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
                    ofmap_base = self.args.ofmap_offset
                    )

            total_cycles += int(cycles)

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


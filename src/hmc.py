import trace_gen_wrapper as tg
import backpropagation as bp
from file_read_backwards import FileReadBackwards

class HMC:
    def __init__(self, args):
        self.args = args

    def inference(self, training=False):
        print("\nFeed-Forward ...")

        param_file = open(self.args.network, 'r')

        if training:
            word_size_bytes = 4
        else:
            word_size_bytes = 1

        first = True
        num_layers = 0
        total_cycles = 0

        for row in param_file:
            if first:
                first = False
                continue

            elems = row.strip().split(',')

            # Do not continue if incomplete line
            if len(elems) < 9:
                continue

            num_layers += 1

            name = elems[0]
            print('\nCommencing run for ' + name)

            ifmap_h = int(elems[1])
            ifmap_w = int(elems[2])

            filt_h = int(elems[3])
            filt_w = int(elems[4])

            num_channels = int(elems[5])
            num_filters = int(elems[6])

            stride = int(elems[7])

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

        if training:
            return total_cycles, num_layers
        else:
            return total_cycles
    # inference() end

    def backprop(self, num_layers=0):
        print('\nBackward Propagation ...')
        assert num_layers > 0

        total_cycles = 0

        with FileReadBackwards(self.args.network) as reversed_param_file:
            for row in reversed_param_file:
                if num_layers > 0:

                    elems = row.strip().split(',')
                    #print(len(elems))

                    # Do not continue if incomplete line
                    if len(elems) < 9:
                        continue

                    num_layers -= 1

                    name = elems[0]
                    print("")
                    print("Commencing back-propagation run for " + name)

                    ifmap_h = int(elems[1])
                    ifmap_w = int(elems[2])

                    filt_h = int(elems[3])
                    filt_w = int(elems[4])

                    num_channels = int(elems[5])
                    num_filters = int(elems[6])

                    stride = int(elems[7])

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

    def train(self):
        print('Start a training epoch ...')

        total_cycles = 0

        inference_cycles, num_layers = self.inference(training=True)
        backprop_cycles = self.backprop(num_layers=num_layers)

        total_cycles = inference_cycles + backprop_cycles

        return total_cycles
    # end of train()


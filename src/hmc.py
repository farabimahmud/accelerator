import trace_gen_wrapper as tg
import backpropagation as bp
import math
from file_read_backwards import FileReadBackwards

class HMC:
    def __init__(self, args):
        self.args = args

    # parameters: cycles: inference_cycles/input/npu
    def inference(self, cycles):
        assert cycles > 0

        total_cycles = cycles * math.ceil(self.args.mini_batch_size / self.args.num_vaults)

        return total_cycles
    # inference() end

    def aggregate_weights(self):
        return

    # parameters: cycles: training_cycles/input/npu
    def train(self, cycles=0):
        assert cycles > 0

        total_cycles = cycles * math.ceil(self.args.mini_batch_size / self.args.num_vaults)

        return total_cycles
    # end of train()


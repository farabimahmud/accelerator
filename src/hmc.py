import math

from npu import NPU

class HMC:
    def __init__(self, args):
        self.args = args
        self.npu = NPU(args)
        self.num_npus = self.args.num_vaults
        self.mini_batch_per_npu = math.ceil(self.args.mini_batch_size / self.args.num_vaults)

    def aggregate(self, model):
        partial_model_per_npu = math.ceil(model.size / self.num_npus)
        cycles = self.npu.aggregate(partial_model_per_npu, self.num_npus)

        return cycles

    # parameters: cycles: inference_cycles/input/npu
    def inference(self, model):

        npu_cycles = self.npu.inference(model)
        cycles = npu_cycles * self.mini_batch_per_npu

        return cycles
    # inference() end

    # parameters: cycles: training_cycles/input/npu
    def train(self, model):

        npu_cycles = self.npu.train(model)
        cycles = npu_cycles * self.mini_batch_per_npu

        return cycles
    # end of train()


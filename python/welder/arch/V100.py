import tvm

class V100:
    def __init__(self):
        self.reg_cap = 64*1024
        self.smem_cap = 48*1024
        self.compute_max_core = 80
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 96*1024
        self.bandwidth = [750, 12080]
        self.platform = "CUDA"
        self.compute_capability = "70"
        self.target = tvm.target.cuda(model="V100", arch="sm_70")
        self.cutlass_mma = [32, 32, 4]

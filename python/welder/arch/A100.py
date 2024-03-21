import tvm

class A100:
    def __init__(self):
        self.reg_cap = 64*1024
        self.smem_cap = 48*1024
        self.compute_max_core = 108
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 164*1024
        self.bandwidth = [1555, 18153]      # based on ampere-architecture-white-paper
        self.platform = "CUDA"
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A100", arch="sm_80")
        self.cutlass_mma = [16, 8, 16]

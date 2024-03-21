import tvm

class A6000:
    def __init__(self):
        self.reg_cap = 64*1024
        self.smem_cap = 48*1024
        self.compute_max_core = 84
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 100*1024      # up to 128-28=100 KB for shared memory
        self.bandwidth = [768, 18024]       # based on ampere-ga-102-gpu-architecture-whitepaper
        self.platform = "CUDA"
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A6000", arch="sm_80")
        self.cutlass_mma = [16, 8, 16]

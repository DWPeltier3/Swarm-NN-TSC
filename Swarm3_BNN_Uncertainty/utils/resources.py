import os
import torch

def print_resources():
    CPUs = int(os.environ.get("SLURM_CPUS_PER_TASK"))
    GPUs = torch.cuda.device_count()
    print('\n*** RESOURCES ***')
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Is distributed training supported? {torch.distributed.is_available()}")  # True if distributed training is supported
    print(f"Is GPU supported? {torch.backends.cudnn.is_available()}")  # True if GPU is supported
    print(f"GPUs available: {GPUs}")
    print(f"CPUs available: {CPUs}")

    # Conditionally set float32 precision for Tensor Cores to improve performance
    # set to 'high' if poor convergence
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.set_float32_matmul_precision('medium')
        print("Tensor Core precision set to 'medium' for CUDA devices.")

    return GPUs, CPUs

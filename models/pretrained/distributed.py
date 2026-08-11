"""Distributed training utilities."""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, Any


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def reduce_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone().detach()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    
    return tensor


def synchronize():
    """Synchronize all processes."""
    if not dist.is_available() or not dist.is_initialized():
        return
    
    dist.barrier()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    
    return dist.get_rank()


def get_world_size() -> int:
    """Get number of processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1


def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_rank() == 0


def run_distributed(rank: int, world_size: int, fn: Callable,
                   args: tuple = (), kwargs: dict = None):
    """Run distributed training function."""
    if kwargs is None:
        kwargs = {}
    
    setup_distributed(rank, world_size)
    try:
        fn(*args, **kwargs)
    finally:
        cleanup_distributed()


def spawn_distributed(fn: Callable, args: tuple = (), 
                     nprocs: int = None, kwargs: dict = None):
    """Spawn distributed processes."""
    if nprocs is None:
        nprocs = torch.cuda.device_count()
    
    if kwargs is None:
        kwargs = {}
    
    mp.spawn(run_distributed, 
            args=(nprocs, fn, args, kwargs),
            nprocs=nprocs)

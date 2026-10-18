# NCCL all-reduce latency, 1-100 KB (Volta roadmap Task 0.3, window C).
#   CUDA_VISIBLE_DEVICES=<gpus> torchrun --nproc_per_node N benchmarks/ar_latency.py
# Run with NCCL_DEBUG=INFO and grep the log for "via P2P/IPC" / "via SHM" to see which transport NCCL picked.
# Note: in-pair (2,3)/(0,1) production TP2 uses vLLM's P2P custom all-reduce, not NCCL — compare in-pair numbers
# against the AR_EVT arm of window B; the cross-pair rows are the ones that price TP4 / PP2xTP2.
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for nbytes in (1024, 4096, 10240, 32768, 102400):
    x = torch.zeros(nbytes // 2, dtype=torch.float16, device="cuda")  # zeros: ones overflow fp16 after ~16 reduces
    for _ in range(50): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(500): dist.all_reduce(x)
    torch.cuda.synchronize()
    if dist.get_rank() == 0: print(f"{os.environ['CUDA_VISIBLE_DEVICES']} {nbytes}B {(time.perf_counter()-t)/500*1e6:.1f}us", flush=True)
dist.destroy_process_group()

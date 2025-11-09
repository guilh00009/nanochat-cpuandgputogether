import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import Optimizer
from functools import partial

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class DistMuon(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, dampening=0.0, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)
        super().__init__(params, defaults)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                # All-reduce to average gradients across all ranks
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Momentum update
                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Parameter update
                p.data.add_(update, alpha=-lr)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=100, emb_size=32, nhead=4, nlayers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        encoder_layer = TransformerEncoderLayer(d_model=32, nhead=nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    model = TinyTransformer().to(torch.device(f"cuda:{local_rank}"))
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1_000)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=local_rank, sync_module_states=True)
    dist_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, dist_config)

    optimizer = DistMuon(model.parameters(), lr=1e-3)

    batch = torch.randint(0, 100, (8, 16), device=torch.device(f"cuda:{local_rank}"))
    target = batch.clone()

    criterion = nn.CrossEntropyLoss()
    model.train()
    output = model(batch)
    loss = criterion(output.view(-1, 100), target.view(-1))

    print(f"[Rank {rank}] Loss before backward: {loss.item():.4f}")
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[Rank {rank}] NaN/Inf loss detected!")
        return

    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.detach()
            print(f"[Rank {rank}] Grad {name}: mean={g.mean().item():.3e}, max={g.max().item():.3e}")

    optimizer.step()
    print(f"[Rank {rank}] Step complete.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

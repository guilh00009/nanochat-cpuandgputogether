import torch
import torch.distributed as dist
from torch.optim import Optimizer

class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, dampening=0.0, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if group["weight_decay"] != 0:
                    d_p = d_p.add(p.data, alpha=group["weight_decay"])

                if group["momentum"] != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(d_p, alpha=1 - group["dampening"])
                    if group["nesterov"]:
                        d_p = d_p.add(buf, alpha=group["momentum"])
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group["lr"])

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

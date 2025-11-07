"""
Muon optimizer from Keller et al.
(robust to 1D params; bf16 on GPU; dynamo-safe)
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.distributed as dist

# ---- Newton–Schulz orthogonalization (no torch.compile aqui) ----
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton–Schulz iteration to approx. orthogonalize G (2D or batched 2D).
    Runs safely in bf16 on GPU. Exige G.ndim >= 2.
    """
    if G.ndim < 2:
        # Não deveria chegar aqui (filtramos no step), mas garantimos fallback.
        return G

    a, b, c = (3.4445, -4.7750, 2.0315)

    # Mantém dispositivo, força bf16 se suportado
    if torch.cuda.is_available():
        X = G.to(dtype=torch.bfloat16, device=G.device, non_blocking=True)
    else:
        # Em CPU, bf16 pode não ter kernel ótimo -> use float32
        X = G.to(dtype=torch.bfloat16 if torch.bfloat16.is_floating_point else torch.float32)

    # Trabalha com matrizes "altas" para melhor estabilidade
    need_T = X.size(-2) > X.size(-1)
    if need_T:
        X = X.mT

    # Normaliza pelo espectro (<=1). Usa norma Fro como bound barato para batched.
    # (norm over last two dims; evita underflow com eps)
    eps = torch.tensor(1e-7, dtype=X.dtype, device=X.device)
    denom = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / denom

    # Iterações Newton–Schulz (quintic)
    for _ in range(int(steps)):
        A = X @ X.mT                   # [*, m, m]
        B = b * A + c * (A @ A)        # quintic approx
        X = a * X + B @ X

    if need_T:
        X = X.mT
    return X

# --------------------------- Muon (single-process) ---------------------------
class Muon(torch.optim.Optimizer):
    """
    Muon - SGD-momentum + (optional) Nesterov, then orthogonalize 2D updates.
    ⚠️ Não use em {0,1}D (embedding, bias, etc.). Estes são ignorados aqui.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        # Agrupa apenas por numel para cache-friendly (como você tinha)
        params = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                if g is None:
                    continue

                # Pule params não 2D (bias/embeddings/fc-out). Use outro otimizador para eles.
                if p.ndim < 2 or g.ndim < 2:
                    continue

                # Momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]

                # SGD-momentum + (opcional) Nesterov
                buf.lerp_(g, 1.0 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # Orthogonalize update (bf16 GPU-friendly)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # Aspect-ratio scaling
                scale = float(max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                p.add_(g, alpha=-group["lr"] * scale)

# --------------------------- Muon (Distributed) ---------------------------
class DistMuon(torch.optim.Optimizer):
    """
    Distributed Muon:
      - reduce_scatter(AVG) para grads
      - owner rank aplica Muon update
      - all_gather para replicar pesos atualizados
    Apenas para parâmetros 2D. Outros devem ficar em outro optimizer/grupo.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        assert dist.is_initialized(), "torch.distributed must be initialized before DistMuon."
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        params = [p for p in params if p.ndim == 2]  # garante 2D
        if len(params) == 0:
            raise ValueError("DistMuon received no 2D parameters.")

        # Agrupar por shape para buffers compatíveis
        shapes = sorted({tuple(p.shape) for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if tuple(p.shape) == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            zero_buffer = torch.zeros(shape, device=device, dtype=dtype)
            param_groups.append(dict(params=group_params, zero_buffer=zero_buffer))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world = dist.get_world_size()

        # Verifica grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError("All params must have grads for DistMuon.")

        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]

            # Processa em janelas de tamanho 'world'
            for base in range(0, len(params), world):
                # Índice do dono (owner) nesta janela
                owner_idx = base + rank

                # RS: cada rank fornece sua fatia; preenche com zeros se faltar
                chunk = params[base:base + world]
                rs_inputs = [p.grad for p in chunk] + [zero_buffer] * max(0, world - len(chunk))
                rs_output = params[owner_idx].grad if owner_idx < len(params) else zero_buffer.clone()

                # reduce_scatter AVG
                dist.reduce_scatter(rs_output, rs_inputs, op=dist.ReduceOp.AVG)

                # Owner aplica Muon update
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = rs_output  # já é a média
                    state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = float(max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)

                # all_gather para replicar params atualizados
                ag_in = params[owner_idx] if owner_idx < len(params) else zero_buffer
                # prepara saídas (preenche até world)
                outs = [torch.empty_like(ag_in) for _ in range(world)]
                dist.all_gather(outs, ag_in)

                # grava de volta nos parâmetros locais desta janela
                for i, out in enumerate(outs):
                    idx = base + i
                    if idx < len(params):
                        params[idx].copy_(out)

import torch
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if momentum > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = torch.sign(buf)
                else:
                    update = torch.sign(grad)

                p.add_(-lr * update)

        return loss
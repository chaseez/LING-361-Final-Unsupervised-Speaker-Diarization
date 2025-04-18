import torch 

class FISTA(torch.optim.Optimizer):
    def __init__(self, params, lr, lambda_):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid lambda: {lambda_} - should be >= 0.0")
        
        defaults = dict(lr=lr, lambda_=lambda_)
        super(FISTA, self).__init__(params, defaults)
    
    def shrinkage_operator(self, u, tresh):
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                lr = group['lr']
                lambda_ = group['lambda_']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state = self.state[p]
                    state['x_prev'] = p.data
                    state['y_prev'] = p.data.clone()
                    state['t_prev'] = torch.tensor(1., device=p.device)

                x_prev, y_prev, t_prev = state['x_prev'], state['y_prev'], state['t_prev']

                x_next = self.shrinkage_operator(y_prev - lr * grad, lambda_)
                t_next = (1. + torch.sqrt(1. + 4. * t_prev ** 2)) / 2.
                y_next = x_next + ((t_prev - 1) / t_next) * (x_next - x_prev)

                state['x_prev'], state['y_prev'], state['t_prev'] = x_next, y_next, t_next

                p.data.copy_(x_next)

        return loss
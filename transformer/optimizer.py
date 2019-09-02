class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, warmup_steps=4000, k=0.2):
        self.optimizer = optimizer
        self.k = k
        self.warmup_steps = warmup_steps
        d_model = 512
        self.init_lr = d_model ** (-0.5)
        self.lr = self.init_lr
        self.warmup_steps = warmup_steps
        self.k = k
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        self.lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                              self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

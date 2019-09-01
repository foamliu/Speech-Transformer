import numpy as np


class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, max_lr=1e-3, min_lr=1e-5, warmup_steps=4000, k=0.0004):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.k = k
        self.step_num = 0
        self.lr = self.max_lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        if self.step_num > self.warmup_steps:
            self.lr = self.max_lr * np.exp(-1.0 * self.k * (self.step_num - self.warmup_steps))
            self.lr = max(self.lr, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

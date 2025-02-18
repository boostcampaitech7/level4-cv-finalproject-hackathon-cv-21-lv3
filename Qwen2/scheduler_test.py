import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch_increment=1):
        self.T_cur += epoch_increment
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch += epoch_increment
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def test_scheduler():
    # Test model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Scheduler parameters
    T_0 = 10  # Total epochs for the first cycle
    T_mult = 2
    eta_max = 1e-3
    T_up = 2  # Warmup epochs
    gamma = 0.8
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0, T_mult, eta_max, T_up, gamma)
    
    epochs = 200
    lrs = []

    for epoch in range(epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), lrs, label="Learning Rate")
    plt.title("Cosine Annealing with Warmup Restarts (Epoch-level Update)")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig("scheduler_epoch_test.png")
    plt.show()


if __name__ == "__main__":
    test_scheduler()

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    k = 0.0004
    warmup_steps = 4000
    max_lr = 1e-3
    min_lr = 1e-5

    lr_list = []
    for step_num in range(1, 100000):

        lr = max_lr
        if step_num > warmup_steps:
            lr = max_lr * np.exp(-1.0 * k * (step_num - warmup_steps))
            lr = max(lr, min_lr)
        lr_list.append(lr)

    print(lr_list[:100])
    print(lr_list[-100:])

    plt.plot(lr_list)
    plt.show()

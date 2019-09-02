import matplotlib.pyplot as plt

if __name__ == '__main__':
    k = 0.2
    warmup_steps = 4000
    d_model = 512
    init_lr = d_model ** (-0.5)

    lr_list = []
    for step_num in range(1, 500000):
        lr = k * init_lr * min(step_num ** (-0.5),
                               step_num * (warmup_steps ** (-1.5)))
        lr_list.append(lr)

    print(lr_list[:100])
    print(lr_list[-100:])

    plt.plot(lr_list)
    plt.show()

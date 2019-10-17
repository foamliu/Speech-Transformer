import matplotlib.pyplot as plt

from data_gen import spec_augment, build_LFR_features
from utils import extract_feature

LFR_m = 4
LFR_n = 3


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')


if __name__ == '__main__':
    path = '../audios/audio_0.wav'
    feature = extract_feature(input_file=path, feature='fbank', dim=80, cmvn=True)
    feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)

    # zero mean and unit variance
    feature = (feature - feature.mean()) / feature.std()

    feature_1 = spec_augment(feature)
    #

    plot_data((feature.transpose(), feature_1.transpose()))
    plt.show()

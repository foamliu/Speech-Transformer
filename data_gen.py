import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import num_workers, pickle_file
from utils import extract_feature


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        # print('f.shape: ' + str(f.shape))
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :input_dim] = feature
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=0)
        batch[i] = (padded_input, padded_target, input_length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


# ------------------------------ utils ------------------------------------
def load_inputs_and_targets(batch, LFR_m=1, LFR_n=1):
    # From: espnet/src/asr/asr_utils.py: load_inputs_and_targets
    # load acoustic features and target sequence of token ids
    # for b in batch:
    #     print(b[1]['input'][0]['feat'])
    xs = [kaldi_io.read_mat(b[1]['input'][0]['feat']) for b in batch]
    ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

    if LFR_m != 1 or LFR_n != 1:
        # xs = build_LFR_features(xs, LFR_m, LFR_n)
        xs = [build_LFR_features(x, LFR_m, LFR_n) for x in xs]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        print("warning: Target sequences include empty tokenid")

    # remove zero-lenght samples
    xs = [xs[i] for i in nonzero_sorted_idx]
    ys = [np.fromiter(map(int, ys[i]), dtype=np.int64)
          for i in nonzero_sorted_idx]

    return xs, ys


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


class AiShellDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['wave']
        trn = sample['trn']

        feature = extract_feature(input_file=wave, feature='fbank', dim=self.args.d_input)
        feature = build_LFR_features(feature, m=self.args.LFR_m, n=self.args.LFR_n)

        return feature, trn

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from utils import parse_args

    args = parse_args()
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers,
                                               pin_memory=True, collate_fn=pad_collate)

    print(len(train_dataset))
    print(len(train_loader))

    feature = train_dataset[10][0]
    print(feature.shape)

    trn = train_dataset[10][1]
    print(trn)

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    IVOCAB = data['IVOCAB']

    print([IVOCAB[idx] for idx in trn])

    for data in train_loader:
        print(data)
        break

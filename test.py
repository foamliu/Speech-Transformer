import argparse
import pickle

import torch
from tqdm import tqdm

from config import pickle_file, device, input_dim, LFR_m, LFR_n, sos_id, eos_id
from data_gen import build_LFR_features
from utils import extract_feature
from xer import cer_function


def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=1, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']
    samples = data['test']

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].to(device)
    model.eval()

    num_samples = len(samples)

    total_cer = 0

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample['wave']
        trn = sample['trn']

        feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(device)
        with torch.no_grad():
            nbest_hyps = model.recognize(input, input_length, char_list, args)

        hyp_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print(hyp_list)

        gt = [char_list[idx] for idx in trn if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        gt_list = [gt]

        print(gt_list)

        cer = cer_function(gt_list, hyp_list)
        total_cer += cer

    avg_cer = total_cer / num_samples

    print('avg_cer: ' + str(avg_cer))

# Usage : python pre_process.py --n_samples train:800,dev:100,test:100
#         python pre_process.py --n_samples train:800,dev:100
#         ...


import os
import pickle

from tqdm import tqdm

from config import wav_folder, tran_file, pickle_file
from utils import ensure_folder, parse_args


def get_data(split, n_samples):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    #n_samples = 5000
    rest = n_samples 
    
    folder = os.path.join(wav_folder, split)
    ensure_folder(folder)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        rest = len(files) if n_samples <= 0 else rest

        for f in files[:rest]:

            wave = os.path.join(dir, f)

            key = f.split('.')[0]

            if key in tran_dict:
                trn = tran_dict[key]
                trn = list(trn.strip()) + ['<eos>']

                for token in trn:
                    build_vocab(token)

                trn = [VOCAB[token] for token in trn]

                samples.append({'trn': trn, 'wave': wave})
        
        rest = rest - len(files) if n_samples > 0 else rest
        if rest <= 0 :
            break  

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":

    # number of examples to use 
    global args
    args = parse_args()
    tmp = args.n_samples.split(",")
    tmp = [a.split(":") for a in tmp]
    tmp = {a[0]:int(a[1]) for a in tmp}
    args.n_samples = {"train":-1, "dev":-1,"test":-1}
    args.n_samples.update(tmp)
    
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train', args.n_samples["train"])
    data['dev'] = get_data('dev', args.n_samples["dev"])
    data['test'] = get_data('test', args.n_samples["test"])
    
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))

import pickle

import numpy as np

from config import pickle_file, sos_id, eos_id

print('loading {}...'.format(pickle_file))
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
VOCAB = data['VOCAB']
IVOCAB = data['IVOCAB']

print('loading bigram_freq.pkl...')
with open('bigram_freq.pkl', 'rb') as file:
    bigram_freq = pickle.load(file)

OUT_LIST = ['<sos>比赛很快便城像一边到的局面第二规合<eos>', '<sos>比赛很快便呈像一边到的局面第二规合<eos>', '<sos>比赛很快便城向一边到的局面第二规合<eos>',
            '<sos>比赛很快便呈向一边到的局面第二规合<eos>', '<sos>比赛很快便城像一边到的局面第二回合<eos>']
GT = '比赛很快便呈向一边倒的局面第二回合<eos>'

print('calculating prob...')
prob_list = []
for out in OUT_LIST:
    print(out)
    out = out.replace('<sos>', '').replace('<eos>', '')
    out = [sos_id] + [VOCAB[ch] for ch in out] + [eos_id]
    prob = 1.0
    for i in range(1, len(out)):
        prob *= bigram_freq[(out[i - 1], out[i])]
    prob_list.append(prob)

prob_list = np.array(prob_list)
prob_list = prob_list / np.sum(prob_list)
print(prob_list)

import collections
import pickle

import nltk
import numpy as np
from tqdm import tqdm

from config import pickle_file

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
char_list = data['IVOCAB']
vocab_size = len(char_list)
samples = data['train']
bigram_counter = collections.Counter()

for sample in tqdm(samples):
    text = sample['trn']
    # text = [char_list[idx] for idx in text]
    tokens = list(text)
    bigrm = nltk.bigrams(tokens)
    # print(*map(' '.join, bigrm), sep=', ')

    # get the frequency of each bigram in our corpus
    bigram_counter.update(bigrm)

# what are the ten most popular ngrams in this Spanish corpus?
print(bigram_counter.most_common(10))

temp_dict = dict()
for key, value in bigram_counter.items():
    temp_dict[key] = value

print('smoothing and freq -> prob')
bigram_freq = dict()
for i in tqdm(range(vocab_size)):
    freq_list = []
    for j in range(vocab_size):
        if (i, j) in temp_dict:
            freq_list.append(temp_dict[(i, j)])
        else:
            freq_list.append(1)

    freq_list = np.array(freq_list)
    freq_list = freq_list / np.sum(freq_list)

    assert (len(freq_list) == vocab_size)
    bigram_freq[i] = freq_list

print(len(bigram_freq[0]))
with open('bigram_freq.pkl', 'wb') as file:
    pickle.dump(bigram_freq, file)

import collections
import pickle

import nltk
from tqdm import tqdm

from config import pickle_file

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
char_list = data['IVOCAB']
samples = data['train']
bigram_freq = collections.Counter()

for sample in tqdm(samples):
    text = sample['trn']
    text = [char_list[idx] for idx in text]
    tokens = list(text)
    bigrm = nltk.bigrams(tokens)
    # print(*map(' '.join, bigrm), sep=', ')

    # get the frequency of each bigram in our corpus
    bigram_freq.update(bigrm)

# what are the ten most popular ngrams in this Spanish corpus?
print(bigram_freq.most_common(10))

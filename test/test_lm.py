import nltk
from nltk.tokenize import word_tokenize
text = "to be or not to be"
tokens = nltk.word_tokenize(text)
bigrm = nltk.bigrams(tokens)
print(*map(' '.join, bigrm), sep=', ')
import pickle

from config import pickle_file

if __name__ == '__main__':
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']

    with open('char_list.pkl', 'wb') as file:
        pickle.dump(char_list, file)

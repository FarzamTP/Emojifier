import numpy as np
import pandas as pd
import emoji

def read_glove_vecs(file_path):
    print("Loading Glove Model..")
    f = open(file_path,'r', errors = 'ignore', encoding='utf8')
    gloveModel = {}
    words = set()
    for line in f:
        try:
            splitLines = line.split()
            word = splitLines[0]
            words.add(word)
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        except:
            pass

    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    print(len(gloveModel),"words loaded!")
    return words_to_index, index_to_words, gloveModel

def one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def label_to_emoji(target_label, emoji_dictionary):
    assert type(target_label) == str
    emoji_unicode = emoji_dictionary.get(target_label)
    return emoji.emojize(emoji_unicode, use_aliases=True)

def extract_X_Y(file_path):
    data = pd.read_csv(file_path, error_bad_lines=False)
    # In case of prior manilulations.    
    if 'Unnamed: 2' in data.columns or 'Unnamed: 3' in data.columns:
        data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3'])
    
    data.columns = ['phrase', 'label']
    X = data.get('phrase')
    Y = data.get('label')
    return X, Y

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sentence_to_avg(sentence, word_to_vector):
    words = (sentence.lower()).split()
    avg = np.zeros((50,))
    total = 0
    for w in words:
        total += word_to_vector[w]
    avg = total / float(len(words))
    return avg
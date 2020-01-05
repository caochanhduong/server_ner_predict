import codecs
# from app.alphabet import Alphabet
from alphabet import Alphabet

import numpy as np
import pickle
from os import listdir
from os.path import isfile, join


def token_2_idx(word_list_train, token2id):
    return_list_train = []
    for word_list in word_list_train:
        return_idx_list = []
        for word in word_list:
            return_idx_list.append(token2id[word.lower()])
        return_list_train.append(return_idx_list)
    return return_list_train

def read_conll_format_tgdd(input_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        word_list = []
        tag_list = []
        words = ['\nxbos', 'xfld', '1']
        tags = ['_xbos_', '_xfld_', '_1_']
        num_sent = 0
        max_length = 0
        for line in f:
            line = line.split()
            if (len(line) > 1):
                words.append(map_number_and_punct(line[0].lower()))
                tags.append(line[1])
            elif (len(line) == 0):
                word_list.append(words)
                tag_list.append(tags)
                sent_length = len(words)
                words = ['\nxbos', 'xfld', '1']
                tags = ['_xbos_', '_xfld_', '_1_']
                num_sent += 1
                max_length = max(max_length, sent_length)

        return word_list, tag_list, num_sent, max_length

def map_number_and_punct(word):
    if word.isdigit():
        word = u'<number>'
    elif word in [u',', u'<', u'.', u'>', u'/', u'?', u'..', u'...', u'....', u':', u';', u'"', u"'", u'[', u'{', u']',
                  u'}', u'|', u'\\', u'`', u'~', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'+',
                  u'=']:
        word = u'<punct>'
    return word

def map_string_2_id_open(string_list, name):
    string_id_list = []
    alphabet_string = Alphabet(name)
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    alphabet_string.close()
    return string_id_list, alphabet_string

def map_string_2_id_close(string_list, alphabet_string):
    string_id_list = []
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    return string_id_list

def map_string_2_id(word_list_train, word_list_test,
                    tag_list_train, tag_list_test):
    word_id_list_train, alphabet_word = map_string_2_id_open(
        word_list_train, 'word')
    word_id_list_test = map_string_2_id_close(word_list_test, alphabet_word)
    tag_id_list_train, alphabet_tag = map_string_2_id_open(
        tag_list_train, 'tag')
    tag_id_list_test = map_string_2_id_close(tag_list_test, alphabet_tag)
    return word_id_list_train, word_id_list_test, tag_id_list_train, tag_id_list_test,\
        alphabet_word, alphabet_tag

def create_data_folder(train_dir, test_dir):
    word_list_train = []
    tag_list_train = []
    num_sent_train = 0
    max_length_train = 0
    word_list_test = []
    tag_list_test = []
    num_sent_test = 0
    max_length_test = 0
    train_files = [join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]
    test_files = [join(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]
    for train_file in train_files:
        word_list_train_temp, tag_list_train_temp, num_sent_train_temp, max_length_train_temp = \
            read_conll_format_tgdd(train_file)
        word_list_train.extend(word_list_train_temp)
        tag_list_train.extend(tag_list_train_temp)
        num_sent_train += num_sent_train_temp
        max_length_train = max(max_length_train, max_length_train_temp)

    for test_file in test_files:
        word_list_test_temp, tag_list_test_temp, num_sent_test_temp, max_length_test_temp = \
            read_conll_format_tgdd(test_file)
        word_list_test.extend(word_list_test_temp)
        tag_list_test.extend(tag_list_test_temp)
        num_sent_test += num_sent_test_temp
        max_length_test = max(max_length_test, max_length_test_temp)

    word_id_list_train, word_id_list_test, tag_id_list_train, tag_id_list_test,\
        alphabet_word, alphabet_tag = \
        map_string_2_id(word_list_train, word_list_test, tag_list_train, tag_list_test)
    max_length = max(max_length_train, max_length_test)
    return word_list_train, word_list_test, word_id_list_train, word_id_list_test, tag_id_list_train, tag_id_list_test, \
            max_length, alphabet_word, alphabet_tag, train_files, test_files

def construct_tensor_onehot(feature_sentences, max_length, dim):
    X = np.zeros([len(feature_sentences), max_length, dim])
    for i in range(len(feature_sentences)):
        for j in range(len(feature_sentences[i])):
            if feature_sentences[i][j] > 0:
                X[i, j, feature_sentences[i][j]] = 1
    return X

def append_to_max(in_list, max_length):
    X = np.zeros([len(in_list), max_length])
    for i in range(len(in_list)):
        appendMaxtrix = np.zeros(max_length - len(in_list[i]), dtype=int)
        X[i] = np.concatenate((in_list[i], appendMaxtrix), axis=0)
    return X

def construct_tensor_word(word_sentences, embedd_vectors, embedd_dim, max_length):
    X = np.empty([len(word_sentences), max_length, embedd_dim])
    for i in range(len(word_sentences)):
        words = word_sentences[i]
        length = len(words)
        for j in range(length):
            word = words[j].lower()
            embedd = embedd_vectors[word]
            X[i, j, :] = embedd
        # Zero out X after the end of the sequence
        X[i, length:] = np.zeros([1, embedd_dim])
    return X

def create_vector_data(word_list_train, tag_id_list_train,\
                       embedd_vectors, embedd_dim,\
                       max_length, dim_tag):
    word_train = construct_tensor_word(word_list_train, embedd_vectors,\
                                       embedd_dim, max_length)
    tag_train = tag_id_list_train
    # tag_train = append_to_max(tag_id_list_train, max_length)
    # tag_train = tag_train
    input_train = word_train
    output_train = tag_train
    return input_train, output_train

"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import sys
sys.path.insert(1, 'C:/Users/lenovoi7/Documents/GitHub/EEC289AFinalProject/read_data')

from sample_code_shakespeare import get_train_test_n_grams_data

# # download the tiny shakespeare dataset
# input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)
#
# with open(input_file_path, 'r') as f:
#     data = f.read()
# print(f"length of dataset in characters: {len(data):,}")
N = 3
all_works_sentences, train_set, test_set, test_n_minus_1_grams = get_train_test_n_grams_data(N)

# get all the unique characters that occur in this text
#all_works_sentence = [''.join(x) for x in all_works_sentences]

def flatten(xss):
    return [x for xs in xss for x in xs]

#print(all_works_senten)
chars = 'abcdefghijklmnopqrstuvwxyz '
vocab_size = len(chars)
print("all the unique characters:", ''.join(set(chars)))
print(f"vocab size: {vocab_size:,}")
print(f"All n grams sequences: {test_n_minus_1_grams}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
#n = len(data)
# train_data = data[:int(n*0.9)]
# test_data = data[int(n*0.9):]

# encode both to integers
train_set_1 = flatten(train_set)
train_set_2 = flatten(train_set_1)
train_set_flat = ''.join(train_set_2)

test_set_1 = flatten(test_set)
test_set_2 = flatten(test_set_1)
test_set_flat = ''.join(test_set_2)

train_ids = encode(train_set_flat)
test_ids = encode(test_set_flat)
print(f"train has {len(train_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

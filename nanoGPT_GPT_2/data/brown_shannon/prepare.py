import os
import pickle
import requests
import numpy as np
import sys
sys.path.insert(1, 'C:/Users/delli7/Documents/GitHub/EEC-289A---Final-Project/read_data')

from sample_code_brown import get_train_test_data, extract_unique_n_minus_1_grams


train_set, test_set = get_train_test_data()

def flatten(list_in):
    return [x for xs in list_in for x in xs]

chars = 'abcdefghijklmnopqrstuvwxyz '
vocab_size = len(chars)

print("all the unique characters:", chars)
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encode both to integers
train_set_1 = flatten(train_set)
train_set_2 = flatten(train_set_1)
train_set_flat = ''.join(train_set_2)

test_set_1 = flatten(test_set)
test_set_2 = flatten(test_set_1)
test_set_flat = ''.join(test_set_2)

N = [2, 3]
n_gram_list = []
for n_gram in N:
    test_n_minus_1_grams = extract_unique_n_minus_1_grams(test_set, n=n_gram)
    test_n_minus_1_grams_list = list(test_n_minus_1_grams)

    n_gram_ids = []
    for n_gram_string in test_n_minus_1_grams_list:
        n_gram_ids.append(encode(n_gram_string))
    print(f"test_n_minus_1_grams for n-gram size of {n_gram} has {len(n_gram_ids):,} tokens")
    n_gram_list.append(n_gram_ids)
os_path = os.path.dirname(__file__)
path = os.path.join(os_path, 'n_gram_list.pkl')
with open(path, 'wb') as f:
    pickle.dump(n_gram_list, f)

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

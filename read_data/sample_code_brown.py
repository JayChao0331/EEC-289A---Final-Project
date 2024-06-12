from nltk.corpus import brown
from read_brown import preprocess_word, preprocess_corpus, tokenizer, train_test_split
import json

def extract_unique_n_minus_1_grams(corpus, n):
    n_minus_1_grams = set()
    for sentence in corpus:
        sentence = ''.join(sentence)
        for i in range(len(sentence) - n + 2):
            context = sentence[i:i + n - 1]
            n_minus_1_grams.add(context)
    return n_minus_1_grams

def get_train_test_data():
    # Read Data
    corpus = brown.sents()
    corpus_processed = preprocess_corpus(corpus)
    corpus_tokenized = tokenizer(corpus_processed)
    train_set, test_set = train_test_split(corpus_tokenized, ratio=0.7)

    return train_set, test_set

def process_json_find_average_entropy():
    init_from = ['scratch', 'gpt2']
    for init_mode in init_from:
        with open(f'entropy_data_{dataset}_{init_mode}.json', 'r') as fp:
            lists = json.load(fp=fp)
            for n_gram_size in n_gram_
                print(lists[0]['entropy'])
if __name__ == '__main__':
    dataset = 'brown_shannon'
    process_json_find_average_entropy(dataset)
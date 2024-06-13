from nltk.corpus import brown
from read_brown import preprocess_word, preprocess_corpus, tokenizer, train_test_split
<<<<<<< HEAD

=======
import json
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648

def extract_unique_n_minus_1_grams(corpus, n):
    n_minus_1_grams = set()
    for sentence in corpus:
        sentence = ''.join(sentence)
        for i in range(len(sentence) - n + 2):
            context = sentence[i:i + n - 1]
            n_minus_1_grams.add(context)
    return n_minus_1_grams

<<<<<<< HEAD


if __name__ == '__main__':
    # Read Data
    N=3
=======
def get_train_test_data():
    # Read Data
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648
    corpus = brown.sents()
    corpus_processed = preprocess_corpus(corpus)
    corpus_tokenized = tokenizer(corpus_processed)
    train_set, test_set = train_test_split(corpus_tokenized, ratio=0.7)
<<<<<<< HEAD
    test_n_minus_1_grams = extract_unique_n_minus_1_grams(test_set, n=N)
=======

    return train_set, test_set
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648

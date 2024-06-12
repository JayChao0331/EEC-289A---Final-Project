from nltk.corpus import brown
from read_brown import preprocess_word, preprocess_corpus, tokenizer, train_test_split

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
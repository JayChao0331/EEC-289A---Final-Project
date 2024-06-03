from nltk.corpus import brown
from read_brown import preprocess_word, preprocess_corpus, tokenizer, train_test_split



if __name__ == '__main__':
    # Read Data
    corpus = brown.sents()
    corpus_processed = preprocess_corpus(corpus)
    corpus_tokenized = tokenizer(corpus_processed)
    train_set, test_set = train_test_split(corpus_tokenized, ratio=0.7)

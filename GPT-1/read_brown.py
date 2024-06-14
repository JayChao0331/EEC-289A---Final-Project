import re
import nltk

nltk.download('brown')


def preprocess_word(word):
    # Use regular expression to keep only a-z, A-Z, and spaces
    cleaned_word = re.sub(r'[^a-zA-Z]', '', word)
    # Convert the word to lowercase
    cleaned_word = cleaned_word.lower()
    return cleaned_word


def preprocess_corpus(corpus):
    # Process each word in each sentence
    cleaned_corpus = []
    for sentence in corpus:
        cleaned_sentence = [preprocess_word(word) for word in sentence if preprocess_word(word)]
        cleaned_corpus.append(cleaned_sentence)
    return cleaned_corpus


def tokenizer(corpus):
    corpus_char = []
    for sentence in corpus:
        sentence_str = ' '.join(sentence)
        sentence_char = list(sentence_str)
        corpus_char.append(sentence_char)
    return corpus_char


def train_test_split(corpus, ratio=0.7):
    corpus_len = len(corpus)
    train_id = int(corpus_len * ratio)
    train_set = corpus[:train_id]
    test_set = corpus[train_id:]
    return train_set, test_set

import math
from nltk.corpus import shakespeare
from collections import defaultdict, Counter

from read_data.read_shakespeare import extract_sentences_from_work, clean_text, validate_text, train_test_split


def extract_unique_n_minus_1_grams(corpus, n):
    n_minus_1_grams_freq = Counter()
    total_n_minus_1_grams = 0
    for sentence in corpus:
        sentence = ''.join(sentence)
        for i in range(len(sentence) - n + 2):
            context = sentence[i:i + n - 1]
            n_minus_1_grams_freq[context] += 1
            total_n_minus_1_grams += 1
    
    # Normalize the frequencies
    n_minus_1_grams_freq_normalized = {context: freq / total_n_minus_1_grams for context, freq in n_minus_1_grams_freq.items()}
    
    return n_minus_1_grams_freq_normalized


def build_ngrams(corpus, n):
    ngrams = defaultdict(Counter)
    for sentence in corpus:
        sentence = ''.join(sentence)
        for i in range(len(sentence) - n + 1):
            context = sentence[i:i + n - 1]
            target = sentence[i + n - 1]
            ngrams[context][target] += 1
    return ngrams


def read_ngrams(ngrams):
    for context, counter in ngrams.items():
        print(f"Context: '{context}'")
        for letter, count in counter.items():
            print(f"    '{letter}': {count}")
        break


def get_probability_distribution(ngrams, context, n):
    context = context[-(n-1):]
    possible_next_letters = ngrams[context]
    total = sum(possible_next_letters.values())
    if total == 0:
        return {letter: 0.0 for letter in 'abcdefghijklmnopqrstuvwxyz '}
    prob_distribution = {letter: (count / total) for letter, count in possible_next_letters.items()}
    
    # Ensure all 27 characters are included in the dictionary
    for letter in 'abcdefghijklmnopqrstuvwxyz ':
        if letter not in prob_distribution:
            prob_distribution[letter] = 0.0
    
    # Sort the dictionary by probability in descending order
    prob_distribution = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True))
    
    return prob_distribution


def calculate_upper_bound(prob_distributions, n_minus_1_grams_freq_normalized):
    q_lst = []
    for id in range(27):
        q=0
        for n_minus_1_gram, prob_distribution in prob_distributions.items():
            letter_id = list(prob_distribution.keys())[id]
            q += n_minus_1_grams_freq_normalized[n_minus_1_gram] * prob_distribution[letter_id]
        q_lst.append(q)
    
    entropy_sum = 0
    for q in q_lst:
        entropy_q = None
        if q == 0:
            entropy_q = 0
        else:
            entropy_q = q * math.log2(q)
        entropy_sum += entropy_q
    
    return -entropy_sum


def calculate_lower_bound(prob_distributions, n_minus_1_grams_freq_normalized):
    q_lst = []
    for id in range(27):
        q=0
        for n_minus_1_gram, prob_distribution in prob_distributions.items():
            letter_id = list(prob_distribution.keys())[id]
            q += n_minus_1_grams_freq_normalized[n_minus_1_gram] * prob_distribution[letter_id]
        q_lst.append(q)

    entropy_sum = 0
    for i in range(27):
        if i == 26:
            entropy_q = (i+1) * (q_lst[i] - 0) * math.log2(i+1)
        else:
            entropy_q = (i+1) * (q_lst[i] - q_lst[i+1]) * math.log2(i+1)
            entropy_sum += entropy_q
    
    return entropy_sum




if __name__ == '__main__':
    works = shakespeare.fileids()

    all_works_sentences = []
    for work in works:
        sentences = extract_sentences_from_work(work)
        cleaned_sentences = [list(clean_text(sentence)) for sentence in sentences]
        all_works_sentences.append(cleaned_sentences)

    if validate_text(all_works_sentences):
        print('Dictionary size is correct!')
    else:
        print('Dictionary size is wrong!')

    train_set, test_set = train_test_split(all_works_sentences)

    print('\nCorpus dataset: {}'.format(len(all_works_sentences)))
    print('Training set: {}'.format(len(train_set)))
    print('Testing set: {}\n'.format(len(test_set)))

    # Build Ngram Model
    N = 2
    ngrams = build_ngrams(train_set, n=N)
    n_minus_1_grams_freq_normalized = extract_unique_n_minus_1_grams(test_set, n=N)

    prob_distributions = {}
    for context in n_minus_1_grams_freq_normalized:
        prob_distributions[context] = get_probability_distribution(ngrams, context, N)
    
    upper_bound = calculate_upper_bound(prob_distributions, n_minus_1_grams_freq_normalized)
    print('{}-gram upper bound: {}'.format(N, upper_bound))

    lower_bound = calculate_lower_bound(prob_distributions, n_minus_1_grams_freq_normalized)
    print('{}-gram lower bound: {}'.format(N, lower_bound))


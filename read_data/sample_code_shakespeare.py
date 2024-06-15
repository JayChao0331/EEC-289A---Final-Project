<<<<<<< HEAD
from nltk.corpus import shakespeare
from read_shakespeare import extract_sentences_from_work, clean_text, validate_text, train_test_split


=======
import nltk
from nltk.corpus import shakespeare
from read_shakespeare import extract_sentences_from_work, clean_text, validate_text, train_test_split

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
=======
def get_train_test_data():
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648
    works = shakespeare.fileids()

    all_works_sentences = []
    for work in works:
        sentences = extract_sentences_from_work(work)
        cleaned_sentences = [list(clean_text(sentence)) for sentence in sentences]
        all_works_sentences.append(cleaned_sentences)

    if validate_text(all_works_sentences):
<<<<<<< HEAD
        print('Dictionary size is correct!')
    else:
        print('Dictionary size is wrong!')

    N=3
    train_set, test_set = train_test_split(all_works_sentences)
    test_n_minus_1_grams = extract_unique_n_minus_1_grams(test_set, n=N)
=======
        print(f'Dictionary size is correct!')
    else:
        print(f'Dictionary size is wrong!')
    train_set, test_set = train_test_split(all_works_sentences)

    return all_works_sentences, train_set, test_set
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648

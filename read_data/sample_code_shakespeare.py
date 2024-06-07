from nltk.corpus import shakespeare
from read_shakespeare import extract_sentences_from_work, clean_text, validate_text, train_test_split



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
